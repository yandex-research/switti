import torch.nn as nn

from .clip import FrozenCLIPEmbedder
from .switti import Switti
from .vqvae import VQVAE
from .pipeline import SwittiPipeline


def build_models(
    # Shared args
    device,
    patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 10 steps by default
    # VQVAE args
    V=4096,
    Cvae=32,
    ch=160,
    share_quant_resi=4,
    # Switti args
    depth=16,
    rope=True,
    rope_theta=10000,
    rope_size=128,
    use_swiglu_ffn=True,
    use_ar=False,
    use_crop_cond=True,
    attn_l2_norm=True,
    init_adaln=0.5,
    init_adaln_gamma=1e-5,
    init_head=0.02,
    init_std=-1,  # init_std < 0: automated
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dpr=0,
    norm_eps=1e-6,
    # pipeline args
    text_encoder_path="openai/clip-vit-large-patch14",
    text_encoder_2_path="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
) -> tuple[VQVAE, Switti]:
    heads = depth
    width = depth * 64
    if dpr > 0:
        dpr = dpr * depth / 24

    # disable built-in initialization for speed
    for clz in (
        nn.Linear,
        nn.LayerNorm,
        nn.BatchNorm2d,
        nn.SyncBatchNorm,
        nn.Conv1d,
        nn.Conv2d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
    ):
        setattr(clz, "reset_parameters", lambda self: None)

    # build models
    vae_local = VQVAE(
        vocab_size=V,
        z_channels=Cvae,
        ch=ch,
        test_mode=True,
        share_quant_resi=share_quant_resi,
        v_patch_nums=patch_nums,
    ).to(device)
    
    switti_wo_ddp = Switti(
        depth=depth,
        embed_dim=width,
        num_heads=heads,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=dpr,
        norm_eps=norm_eps,
        attn_l2_norm=attn_l2_norm,
        patch_nums=patch_nums,
        rope=rope,
        rope_theta=rope_theta,
        rope_size=rope_size,
        use_swiglu_ffn=use_swiglu_ffn,
        use_ar=use_ar,
        use_crop_cond=use_crop_cond,
    ).to(device)
    
    switti_wo_ddp.init_weights(
        init_adaln=init_adaln,
        init_adaln_gamma=init_adaln_gamma,
        init_head=init_head,
        init_std=init_std,
    )
    text_encoder = FrozenCLIPEmbedder(text_encoder_path)
    text_encoder_2 = FrozenCLIPEmbedder(text_encoder_2_path)
    pipe = SwittiPipeline(switti_wo_ddp, vae_local, text_encoder, text_encoder_2, device)

    return vae_local, switti_wo_ddp, pipe
