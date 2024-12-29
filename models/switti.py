import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from diffusers.models.embeddings import GaussianFourierProjection

import dist
from models.basic_switti import AdaLNBeforeHead, AdaLNSelfCrossAttn
from models.rope import compute_axial_cis
from utils.arg_util import RESOLUTION_PATCH_NUMS_MAPPING

def get_crop_condition(
    heights: list, 
    widths: list, 
    base_size=512
):
    if type(heights[0]) == type(widths[0]) == str:
        heights = [int(h) for h in heights]
        widths = [int(w) for w in widths]
    h = torch.tensor(heights, dtype=torch.int).unsqueeze(1)
    w = torch.tensor(widths, dtype=torch.int).unsqueeze(1)
    hw = torch.cat([h, w], dim=1)
    
    ratio = base_size / hw.min(-1)[0]
    orig_size = (hw * ratio[:, None]).to(torch.int)
    crop_coords = ((orig_size - base_size) // 2).clamp(min=0)
    crop_cond = torch.cat([orig_size, crop_coords], dim=1)

    return crop_cond


class Switti(nn.Module):
    def __init__(
        self,
        Cvae=32,
        V=4096,
        rope=True,
        rope_theta=10000,
        rope_size=128,
        depth=16,
        embed_dim=1024,
        num_heads=16,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_eps=1e-6,
        attn_l2_norm=True,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 10 steps by default
        fused_if_available=True,
        use_swiglu_ffn=True,
        use_ar=False,
        use_crop_cond=True,
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.depth, self.C, self.D, self.num_heads = (
            depth,
            embed_dim,
            embed_dim,
            num_heads,
        )
        self.Cvae, self.V = Cvae, V

        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn**2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.rope = rope

        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())

        # 1. input (word) embedding
        self.word_embed = nn.Linear(self.Cvae, self.C)

        # 2. text embedding
        self.pooled_embed_size = 1280
        self.context_dim = 1280 + 768
        self.text_pooler = nn.Linear(self.pooled_embed_size, self.D)

        init_std = math.sqrt(1 / self.C / 3)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)

        # 3. position embedding
        if not self.rope:
            # absolute position embedding
            pos_1LC = []
            for i, pn in enumerate(self.patch_nums):
                pe = torch.empty(1, pn * pn, self.C)
                nn.init.trunc_normal_(pe, mean=0, std=init_std)
                pos_1LC.append(pe)
            pos_1LC = torch.cat(pos_1LC, dim=1)  # 1, L, C
            assert tuple(pos_1LC.shape) == (1, self.L, self.C)
            self.pos_1LC = nn.Parameter(pos_1LC)
            self.freqs_cis = None
        else:
            # RoPE position embedding
            assert (
                self.C // self.num_heads
            ) % 4 == 0, "2d rope needs head dim to be divisible by 4"
            patch_nums_m1 = tuple(pn - 1 if pn > 1 else 1 for pn in self.patch_nums)
            self.compute_cis = partial(compute_axial_cis, dim=self.C // self.num_heads)
            freqs_cis = []
            for i, pn in enumerate(self.patch_nums):
                norm_coeff = rope_size / patch_nums_m1[i]
                cur_freqs = self.compute_cis(
                    end_x=pn, end_y=pn, theta=rope_theta, norm_coeff=norm_coeff
                )
                freqs_cis.append(cur_freqs[None, ...])
            self.freqs_cis = torch.cat(freqs_cis, dim=1)  # 1, L, C // 2 -- complex

        # level embedding (similar to GPT's segment embedding, 
        # used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)

        # 4. backbone blocks
        self.drop_path_rate = drop_path_rate
        # stochastic depth decay rule (linearly increasing)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([])
        for block_idx in range(depth):
            self.blocks.append(
                AdaLNSelfCrossAttn(
                    cond_dim=self.D,
                    block_idx=block_idx,
                    embed_dim=self.C,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[block_idx],
                    last_drop_p=0 if block_idx == 0 else dpr[block_idx - 1],
                    qk_norm=attn_l2_norm,
                    context_dim=self.context_dim,
                    use_swiglu_ffn=use_swiglu_ffn,
                    norm_eps=norm_eps,
                    use_crop_cond=use_crop_cond,
                )
            )

        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f"\n[constructor]  ==== fused_if_available={fused_if_available} "
            f"(fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, "
            f"fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n"
            f"    [Switti config ] embed_dim={embed_dim}, num_heads={num_heads}, "
            f"depth={depth}, mlp_ratio={mlp_ratio}\n"
            f"    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, "
            f"drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})",
            end="\n\n",
            flush=True,
        )

        # Prepare crop condition embedder
        self.use_crop_cond = use_crop_cond
        if use_crop_cond:
            # crop condition is repredsented with 4 int values. each is embeded to self.D // 4 dim
            assert self.D % 8 == 0
            self.crop_embed = GaussianFourierProjection(
                self.D // 2 // 4, set_W_to_weight=False, log=False, flip_sin_to_cos=False
            )
            self.crop_proj = nn.Linear(self.D, self.D)

        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        self.use_ar = use_ar
        d: torch.Tensor = torch.cat(
            [torch.full((pn * pn,), i) for i, pn in enumerate(self.patch_nums)]
        ).view(1, self.L, 1)
        dT = d.transpose(1, 2)  # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer("lvl_1L", lvl_1L)
        
        if self.use_ar:
            attn_bias_for_masking = torch.where(d >= dT, 0.0, -torch.inf)
        else:
            attn_bias_for_masking = torch.where(d == dT, 0.0, -torch.inf)
            
        attn_bias_for_masking = attn_bias_for_masking.reshape(1, 1, self.L, self.L)
        self.register_buffer(
            "attn_bias_for_masking", attn_bias_for_masking.contiguous()
        )

        # 6. classifier head
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

        # By default disable gradient checkpointing
        self.use_gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.use_gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.use_gradient_checkpointing = False

    def get_logits(
        self,
        h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        cond_BD: Optional[torch.Tensor],
    ):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual  # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:  # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h, cond_BD))


    def forward(
        self,
        x_BLCv_wo_first_l: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        prompt_attn_bias: torch.Tensor,
        batch_height: list[int] | None = None,
        batch_width: list[int] | None = None,
    ) -> torch.Tensor:  # returns logits_BLV
        """
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :param prompt_embeds (B, context_len, self.context_dim):
        text features from pipe.text_encoder and pipe.text_encoder_2,
        concatenated along dim=-1, padded to longest along dim=1
        :param pooled_prompt_embeds (B, self.pooled_embed_size):
        pooled text features from pipe.text_encoder_2
        :param prompt_attn_bias (B, context_len):
        boolean mask to specify which tokens are not padding
        :param batch_height (B,): original height of images in a batch.
        :param batch_width (B,): original width of images in a batch.
        Only used when self.use_crop_cond = True
        :return: logits BLV, V is vocab_size
        """
        bg, ed = 0, self.L
        B = x_BLCv_wo_first_l.shape[0]
        with torch.amp.autocast('cuda', enabled=False):
            pooled_prompt_embeds = self.text_pooler(pooled_prompt_embeds)

            sos = cond_BD = pooled_prompt_embeds
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(
                B, self.first_l, -1
            )

            x_BLC = torch.cat(
                (sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1
            )
            x_BLC += self.lvl_embed(
                self.lvl_1L[:, :ed].expand(B, -1)
            )  # lvl: BLC;  pos: 1LC
            if not self.rope:
                x_BLC += self.pos_1LC[:, :ed]
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        
        if self.use_crop_cond:
            crop_coords = get_crop_condition(batch_height, batch_width).to(cond_BD.device)
            crop_embed = self.crop_embed(crop_coords.view(-1)).reshape(B, self.D)
            crop_cond = self.crop_proj(crop_embed)
        else:
            crop_cond = None

        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype

        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD = cond_BD.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        for block in self.blocks:
            if self.use_gradient_checkpointing:
                x_BLC = torch.utils.checkpoint.checkpoint(
                    block,
                    x=x_BLC,
                    cond_BD=cond_BD,
                    attn_bias=attn_bias,
                    context=prompt_embeds,
                    freqs_cis=self.freqs_cis,
                    context_attn_bias=prompt_attn_bias,
                    crop_cond=crop_cond,
                    use_reentrant=False,
                )
            else:
                x_BLC = block(
                    x=x_BLC,
                    cond_BD=cond_BD,
                    attn_bias=attn_bias,
                    context=prompt_embeds,
                    freqs_cis=self.freqs_cis,
                    context_attn_bias=prompt_attn_bias,
                    crop_cond=crop_cond,
                )

        with torch.amp.autocast('cuda', enabled=not self.training):
            x_BLC = self.get_logits(x_BLC, cond_BD.float())

        return x_BLC  # logits BLV, V is vocab_size

    def init_weights(
        self,
        init_adaln=0.5,
        init_adaln_gamma=1e-5,
        init_head=0.02,
        init_std=0.02,
    ):
        if init_std < 0:
            init_std = (1 / self.C / 3) ** 0.5  # init_std < 0: automated

        print(f"[init_weights] {type(self).__name__} with {init_std=:g}")
        for m in self.modules():
            with_weight = hasattr(m, "weight") and m.weight is not None
            with_bias = hasattr(m, "bias") and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
            elif isinstance(
                m,
                (
                    nn.LayerNorm,
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                    nn.SyncBatchNorm,
                    nn.GroupNorm,
                    nn.InstanceNorm1d,
                    nn.InstanceNorm2d,
                    nn.InstanceNorm3d,
                ),
            ):
                if with_weight:
                    m.weight.data.fill_(1.0)
                if with_bias:
                    m.bias.data.zero_()

        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()

        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if (
                hasattr(self.head_nm.ada_lin[-1], "bias")
                and self.head_nm.ada_lin[-1].bias is not None
            ):
                self.head_nm.ada_lin[-1].bias.data.zero_()

        depth = len(self.blocks)
        for block in self.blocks:
            block.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            block.cross_attn.proj.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(block.ffn, "fc2"):
                block.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))

            if hasattr(block, "ada_lin"):
                block.ada_lin[-1].weight.data[2 * self.C :].mul_(init_adaln)
                block.ada_lin[-1].weight.data[: 2 * self.C].mul_(init_adaln_gamma)
                if (
                    hasattr(block.ada_lin[-1], "bias")
                    and block.ada_lin[-1].bias is not None
                ):
                    block.ada_lin[-1].bias.data.zero_()
            elif hasattr(block, "ada_gss"):
                block.ada_gss.data[:, :, 2:].mul_(init_adaln)
                block.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)

    def extra_repr(self):
        return f"drop_path_rate={self.drop_path_rate:g}"


class SwittiHF(Switti, PyTorchModelHubMixin):
    # tags=["image-generation"]):
    def __init__(
        self,
        depth=30,
        rope=True,
        rope_theta=10000,
        rope_size=128,
        use_swiglu_ffn=True,
        use_ar=False,
        use_crop_cond=True,
        reso=512,
    ):
        heads = depth
        width = depth * 64
        patch_nums = tuple([int(x) for x in RESOLUTION_PATCH_NUMS_MAPPING[reso].split("_")])
        super().__init__(
            depth=depth,
            embed_dim=width,
            num_heads=heads,
            patch_nums=patch_nums,
            rope=rope,
            rope_theta=rope_theta,
            rope_size=rope_size,
            use_swiglu_ffn=use_swiglu_ffn,
            use_ar=use_ar,
            use_crop_cond=use_crop_cond,
        )
