import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid

import dist
from models import Switti, VQVAE
from models.pipeline import SwittiPipeline
from utils.amp_sc import AmpOptimizer
from utils.misc import TensorboardLogger

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor

EVAL_PROMPTS = [
    "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
    "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
    "A sad puppy with large eyes",
    "A girl with pale blue hair and a cami tank top",
    "cute girl, Kyoto animation, 4k, high resolution",
    "A person laying on a surfboard holding his dog",
    "Green commercial building with refrigerator and refrigeration units outside",
    "An airplane with two propellor engines flying in the sky",
    "Four cows in a pen on a sunny day",
    "Three dogs sleeping together on an unmade bed",
    "a deer with bird feathers, highly detailed, full body",
    "A city in 4-dimensional space-time",
    "A black dog sitting on a wooden chair. A white cat with black ears is standing up with its paws on the chair.",
    "a cat patting a crystal ball with the number 7 written on it in black marker",
    "a barred owl peeking out from dense tree branches",
    "a cat sitting on a stairway railing",
    "a cat drinking a pint of beer",
    "a bat landing on a baseball bat",
    "a black dog sitting between a bush and a pair of green pants standing up with nobody inside them",
    "a close-up of a blue dragonfly on a daffodil",
    "A close-up of two beetles wearing karate uniforms and fighting, jumping over a waterfall."
]


class SwittiTrainer(object):
    def __init__(
        self,
        dataloader,
        device,
        patch_nums: Tuple[int, ...],
        resos: Tuple[int, ...],
        pipe: SwittiPipeline,
        vae_local: VQVAE,
        switti_wo_ddp: Switti,
        switti: DDP,
        optimizer: AmpOptimizer,
        label_smooth: float,
        args=None,
    ):
        super().__init__()
        self.dataloader = iter(dataloader)
        self.args = args

        self.switti, self.vae_local, self.quantize_local = (
            switti,
            vae_local,
            vae_local.quantize,
        )
        self.switti_wo_ddp: Switti = switti_wo_ddp  # after torch.compile
        self.optimizer = optimizer
        self.pipe = pipe
        self.switti_wo_ddp.rng = torch.Generator(device=device)

        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smooth, reduction="none"
        )
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction="mean")
        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L

        self.patch_nums, self.resos = patch_nums, resos
        self.begin_ends = []
        cur = 0
        for pn in patch_nums:
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn * pn
        self.device = device
        self.grad_accum = args.grad_accum
        self.embed_noise_std = args.embed_noise_std

    def train_step(
        self,
        g_it: int,
        tb_lg: TensorboardLogger
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        # forward
        self.switti.train()
        for accum_iter in range(self.grad_accum):
            image, prompt = next(self.dataloader)

            inp_B3HW = image.to(self.device, non_blocking=True)
            inp_B3HW = F.interpolate(
                inp_B3HW, size=(self.resos[-1], self.resos[-1]), mode="bicubic",
            )

            B, V = inp_B3HW.size(0), self.vae_local.vocab_size

            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(
                inp_B3HW, noise_std=self.embed_noise_std
            )
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_switti_input(gt_idx_Bl)
            if self.args.uncond_proba > 0:
                cond_uncond_choice = torch.bernoulli(
                    torch.full((B, ), self.args.uncond_proba)
                )
                for i_, p_ in enumerate(cond_uncond_choice):
                    if p_ == 1:
                        prompt[i_] = ""
            (prompt_embeds,
             pooled_prompt_embeds,
             prompt_attn_bias,
             ) = self.pipe.encode_prompt(prompt, encode_null=False)

            with self.optimizer.amp_ctx:
                logits_BLV = self.switti(
                    x_BLCv_wo_first_l,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    prompt_attn_bias=prompt_attn_bias,
                )
                loss = self.train_loss(logits_BLV.view(-1, V),
                                       gt_BL.view(-1),
                                       ).view(B, -1)
                loss = loss.mul(self.loss_weight).sum(dim=-1).mean()

            # backward
            is_stepping = (accum_iter + 1) == self.grad_accum
            grad_norm, scale_log2 = self.optimizer.backward_clip_step(
                loss=loss,
                is_stepping=is_stepping,
                )

        # log to tensorboard
        if g_it > 0 and g_it % self.args.log_iters == 0:
            # recalculate logits in .eval() mode to log acc
            self.switti.eval()
            if self.args.use_gradient_checkpointing:
                self.switti.disable_gradient_checkpointing()
            with torch.no_grad(), self.optimizer.amp_ctx:
                logits_BLV = self.switti(
                    x_BLCv_wo_first_l,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    prompt_attn_bias=prompt_attn_bias,
                )

            # Compute cluster usage
            pred_BL = logits_BLV.data.argmax(dim=-1)
            prob_per_class_is_chosen = pred_BL.view(-1).bincount(minlength=V).float().cuda()
            dist.allreduce(prob_per_class_is_chosen)
            prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
            cluster_usage = (
                prob_per_class_is_chosen > 0.001 / V
            ).float().mean().item() * 100

            logits_lg = dict()
            kw = dict(z_voc_usage=cluster_usage, acc_total=0.0, L_total=0.0)
            for si, (bg, ed) in enumerate(self.begin_ends):
                pred = logits_BLV.data[:, bg:ed].reshape(-1, V)
                tar = gt_BL[:, bg:ed].reshape(-1)
                top5 = torch.topk(pred, 5, dim=-1)[1]

                acc = (pred.argmax(dim=-1) == tar).float().mean().item() * 100
                acc_top5 = torch.eq(tar[:, None], top5).any(dim=1).float().mean().item() * 100
                ce = self.val_loss(pred, tar).item()
                std = pred.std(dim=-1).mean().item()
                norm = pred.norm(dim=-1).mean().item()

                stats = torch.tensor([acc, acc_top5, ce, std, norm], device=dist.get_device())
                dist.allreduce(stats)
                stats /= dist.get_world_size()
                acc, acc_top5, ce, std, norm = stats.tolist()

                logits_lg[f"logits_std_{self.resos[si]}"] = std
                logits_lg[f"logits_norm_{self.resos[si]}"] = norm
                kw[f"acc_{self.resos[si]}"] = acc
                kw[f"acc_top5_{self.resos[si]}"] = acc_top5
                kw[f"L_{self.resos[si]}"] = ce
                kw[f"acc_total"] += acc / len(self.begin_ends)
                kw[f"L_total"] += ce / len(self.begin_ends)

            if g_it % self.args.log_images_iters == 0:
                with FSDP.summon_full_params(self.switti, writeback=False):
                    torch.cuda.empty_cache()
                    for cfg in [0, 6]:
                        subprompt = prompt[:16]
                        imgs = self.pipe(subprompt,
                                         cfg=cfg,
                                         top_k=self.args.top_k,
                                         top_p=self.args.top_p,
                                         return_pil=False,
                                         )
                        imgs = make_grid(imgs, nrow=math.ceil(math.sqrt(len(imgs))))
                        tb_lg.log_image(
                            f"train_imgs_top_k={self.args.top_k}_top_p={self.args.top_p}_cfg={cfg}",
                            imgs,
                            step=g_it,
                            )

                        imgs = self.pipe(
                            EVAL_PROMPTS,
                            cfg=cfg,
                            top_k=self.args.top_k,
                            top_p=self.args.top_p,
                            return_pil=False,
                        )
                        imgs = make_grid(imgs, nrow=math.ceil(math.sqrt(len(imgs))))
                        tb_lg.log_image(
                            f"eval_imgs_topk={self.args.top_k}_top={self.args.top_p}_cfg={cfg}",
                            imgs,
                            step=g_it,
                            )

                        imgs = self.pipe(
                            EVAL_PROMPTS,
                            top_k=1,
                            cfg=cfg,
                            return_pil=False,
                        )
                        imgs = make_grid(imgs, nrow=math.ceil(math.sqrt(len(imgs))))
                        tb_lg.log_image(f"eval_imgs_topk_1_cfg{cfg}", imgs, step=g_it)
                        del imgs

            if dist.is_master():
                tb_lg.update(head="Logits_stats", **logits_lg, step=g_it)
                tb_lg.update(head="AR_iter_loss", **kw, step=g_it)
            print(f"LOGGING {g_it} FINISHED")
            if self.args.use_gradient_checkpointing:
                self.switti.enable_gradient_checkpointing()
            self.switti.train()
            dist.barrier()

        return grad_norm.item(), scale_log2

    def get_config(self):
        return {
            "patch_nums": self.patch_nums,
            "resos": self.resos,
            "label_smooth": self.label_smooth,
        }
