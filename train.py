import gc
import os
import sys
import time

import torch
from trainer import SwittiTrainer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data import DataLoader

import dist
from calculate_metrics import distributed_metrics_with_csv, to_PIL_image
from models import Switti, VQVAE, build_models
from models.basic_switti import AdaLNSelfCrossAttn
from utils import arg_util, misc
from utils.amp_sc import AmpOptimizer
from utils.fsdp import load_model_state, save_model_state
from utils.lr_control import filter_params, lr_wd_annealing
from utils.data import build_dataset, coco_collate_fn
from utils.data_sampler import DistInfiniteBatchSampler
from utils.fid_score_in_memory import calculate_fid


DEFAULT_VAE_CKPT = "vae_ch160v4096z32.pth"

def build_everything(args: arg_util.Args):
    # create tensorboard logger
    tb_lg: misc.TensorboardLogger
    if dist.is_master():
        os.makedirs(args.tb_log_dir_path, exist_ok=True)
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(
            misc.TensorboardLogger(
                log_dir=args.tb_log_dir_path,
                filename_suffix=f'__{misc.time_str("%m%d_%H%M")}',
            ),
            verbose=True,
        )
        tb_lg.flush()
    else:
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(None, verbose=False)

    # log args
    print(f"initial args:\n{str(args)}")

    # build models
    vae_local, switti_wo_ddp, pipe = build_models(
        # VQVAE hyperparameters
        V=args.vqvae_vocab_size,
        Cvae=args.vqvae_channel_dim,
        ch=args.vqvae_n_channels,
        share_quant_resi=args.vqvae_share_quant_resi,
        # train hyperparameters
        device=dist.get_device(),
        patch_nums=args.patch_nums,
        depth=args.depth,
        attn_l2_norm=args.anorm,
        init_adaln=args.aln,
        init_adaln_gamma=args.alng,
        init_head=args.hd,
        init_std=args.ini,
        text_encoder_path=args.text_encoder_path,
        text_encoder_2_path=args.text_encoder_2_path,
        rope=args.rope,
        rope_theta=args.rope_theta,
        rope_size=args.rope_size,
        dpr=args.drop_path_rate,
        use_swiglu_ffn=args.use_swiglu_ffn,
        use_crop_cond=args.use_crop_cond,
    )
    # Load VAE and Switti checkpoints
    if args.vae_ckpt is None and not os.path.exists(DEFAULT_VAE_CKPT):
        if dist.is_local_master():
            os.system(f'wget https://huggingface.co/FoundationVision/var/resolve/main/{DEFAULT_VAE_CKPT}')
    dist.barrier()
    vae_local.load_state_dict(torch.load(args.vae_ckpt, map_location="cpu"), strict=True)
    start_it = load_model_state(args, switti_wo_ddp)
    vae_local: VQVAE = args.compile_model(vae_local, args.vfast)
    switti_wo_ddp: Switti = args.compile_model(switti_wo_ddp, args.tfast)
    if args.use_gradient_checkpointing:
        switti_wo_ddp.enable_gradient_checkpointing()

    print(f"[INIT] Switti model = {switti_wo_ddp}\n\n")
    count_p = lambda m: f"{sum(p.numel() for p in m.parameters())/1e6:.2f}"
    print(f"[INIT][#para] "
        + ", ".join([f"{k}={count_p(m)}"
        for k, m in (
            ("VAE", vae_local),
            ("VAE.enc", vae_local.encoder),
            ("VAE.dec", vae_local.decoder),
            ("VAE.quant", vae_local.quantize),
    )]))
    print(
        f"[INIT][#para] "
        + ", ".join([f"{k}={count_p(m)}" for k, m in (("Switti", switti_wo_ddp),)])
        + "\n\n"
    )

    # FSDP wrapper
    switti: FSDP = (FSDP if dist.initialized() else NullDDP)(
        switti_wo_ddp,
        auto_wrap_policy=lambda module, recurse, **_etc: recurse or isinstance(module, AdaLNSelfCrossAttn),
        device_id=dist.get_local_rank(),
        sharding_strategy=ShardingStrategy.HYBRID_SHARD if args.use_fsdp else ShardingStrategy.NO_SHARD, #FULL_SHARD,
        use_orig_params=True,
        forward_prefetch=True,
        limit_all_gathers=True,
    )
    # build optimizer
    names, paras, para_groups = filter_params(switti, nowd_keys={
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta',
        'ada_gss', 'moe_bias',
        'scale_mul',
    })

    optimizer = torch.optim.AdamW(
        params=para_groups,
        lr=args.tlr, weight_decay=0.0,
        betas=(args.adam_beta1, args.adam_beta2),
        fused=args.afuse if not args.use_fsdp else False,
    )

    switti_optimizer = AmpOptimizer(
        mixed_precision=args.fp16,
        optimizer=optimizer,
        names=names,
        paras=paras,
        grad_clip=args.tclip,
    )
    del names, paras, para_groups

    # build data
    print(f"[build PT data] ...\n")
    print(f"global bs={args.glb_batch_size}, local bs={args.batch_size}")
    dataset_train = build_dataset(
        args.data_path, final_reso=args.data_load_reso, hflip=args.hflip, mid_reso=args.mid_reso,
    )
    ld_train = DataLoader(
        dataset=dataset_train, num_workers=args.workers, pin_memory=True,
        generator=args.get_different_generator_for_each_rank(), # worker_init_fn=worker_init_fn,
        collate_fn=coco_collate_fn,
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(dataset_train), glb_batch_size=args.glb_batch_size, same_seed_for_all_ranks=args.same_seed_for_all_ranks,
            shuffle=True, fill_last=True, rank=dist.get_rank(), world_size=dist.get_world_size(), start_it=start_it,
        ),
    )
    del dataset_train

    # build trainer
    trainer = SwittiTrainer(
        dataloader=ld_train,
        device=args.device,
        patch_nums=args.patch_nums,
        resos=args.resos,
        pipe=pipe,
        vae_local=vae_local,
        switti_wo_ddp=switti_wo_ddp,
        switti=switti,
        optimizer=switti_optimizer,
        label_smooth=args.ls,
        args=args,
    )
    torch.cuda.empty_cache()

    return (tb_lg, trainer, start_it)


def main_training():
    torch.set_num_threads(32)
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    (tb_lg, trainer, start_it) = build_everything(args)
    dist.barrier()

    # train
    for cur_iter in range(start_it, args.max_iters):
        tb_lg.set_step(cur_iter)

        # get current lr, wd
        min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(
            args.sche,
            trainer.optimizer.optimizer,
            args.tlr,
            args.twd,
            args.twde,
            cur_iter,
            args.wp,
            args.max_iters,
            wp0=args.wp0,
            wpe=args.wpe,
            wp_start_it=start_it,
        )
        args.cur_lr, args.cur_wd = max_tlr, max_twd

        # model forward-backward
        grad_norm, scale_log2 = trainer.train_step(g_it=cur_iter, tb_lg=tb_lg)

        tb_lg.update(head="AR_opt_lr/lr_min", sche_tlr=min_tlr)
        tb_lg.update(head="AR_opt_lr/lr_max", sche_tlr=max_tlr)
        tb_lg.update(head='AR_opt_wd/wd_max', sche_twd=max_twd)
        tb_lg.update(head='AR_opt_wd/wd_min', sche_twd=min_twd)
        tb_lg.update(head="AR_opt_grad/fp16", scale_log2=scale_log2)
        if args.tclip > 0:
            tb_lg.update(head="AR_opt_grad/grad", grad_norm=grad_norm)
            tb_lg.update(head="AR_opt_grad/grad", grad_clip=args.tclip)

        if cur_iter % args.save_iters == 0 and cur_iter > start_it:
            save_model_state(cur_iter, args, trainer.switti)
            # Calculate metrics
            trainer.pipe.switti.eval()
            for eval_set_name in ['coco', 'mjhq']:
                if eval_set_name == "coco":
                    eval_prompts_path = 'eval_prompts/coco.csv'
                    fid_stats_path = args.coco_ref_stats_path
                else:
                    eval_prompts_path = 'eval_prompts/mjhq.csv'
                    fid_stats_path = args.mjhq_ref_stats_path

                with FSDP.summon_full_params(trainer.switti, writeback=False):
                    local_images, local_pick_score, local_clip_score, local_image_reward = distributed_metrics_with_csv(
                        trainer.pipe,
                        eval_prompts_path,
                        args,
                    )

                dist.allreduce(local_pick_score)
                pick_score = local_pick_score.item() / dist.get_world_size()

                dist.allreduce(local_clip_score)
                clip_score = local_clip_score.item() / dist.get_world_size()

                dist.allreduce(local_image_reward)
                image_reward = local_image_reward.item() / dist.get_world_size()

                gathered_images = dist.allgather(local_images)
                images = [to_PIL_image(image) for image in gathered_images]

                if dist.is_master():
                    print("Evaluating FID score...")
                    fid_score = calculate_fid(
                        images, fid_stats_path, inception_path=args.inception_path
                    )

                    eval_metrics = {
                        "CLIP score": clip_score,
                        "FID": fid_score,
                        "Pickscore": pick_score,
                        "ImageReward": image_reward,
                    }
                    tb_lg.update(
                        head=f"{eval_set_name}_metrics_top_k={args.top_k}_top_p={args.top_p}_cfg={args.guidance}",
                        **eval_metrics,
                        step=cur_iter,
                    )

                del local_images, images, gathered_images 
                gc.collect(), torch.cuda.empty_cache()

                dist.barrier()
                print("Finished metrics calculation...")
                args.dump_log()
                tb_lg.flush()
            trainer.pipe.switti.train()

    gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    args.remain_time, args.finish_time = "-", time.strftime(
        "%Y-%m-%d %H:%M", time.localtime(time.time() - 60)
    )
    print(f"final args:\n\n{str(args)}")
    args.dump_log()
    tb_lg.flush()
    tb_lg.close()
    dist.barrier()



class NullDDP(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


if __name__ == "__main__":
    try:
        main_training()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(
            sys.stderr, misc.SyncPrint
        ):
            sys.stdout.close(), sys.stderr.close()
