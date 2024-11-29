import json
import os
import random
import subprocess
import sys
import time
from collections import OrderedDict
from typing import Optional, Union

import numpy as np
import torch

from utils import misc

try:
    from tap import Tap
except ImportError as e:
    print(
        f"`>>>>>>>> from tap import Tap` failed, please run:      pip3 install typed-argument-parser     <<<<<<<<",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"`>>>>>>>> from tap import Tap` failed, please run:      pip3 install typed-argument-parser     <<<<<<<<",
        file=sys.stderr,
        flush=True,
    )
    time.sleep(5)
    raise e

import dist

RESOLUTION_PATCH_NUMS_MAPPING = {
    256: "1_2_3_4_5_6_8_10_13_16",
    512: "1_2_3_4_6_9_13_18_24_32",
    1024: "1_2_3_4_5_7_9_12_16_21_27_36_48_64",
}

class Args(Tap):
    data_path: str = "path_to_your_dataset"
    text_encoder_path: str = "openai/clip-vit-large-patch14"
    text_encoder_2_path: str = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    vae_ckpt: str = "vae_checkpoint_ft.pt"
    exp_name: str = "experiment_name"

    # eval sampling args
    num_images_for_metrics: int = 1
    num_images_for_sbs: int = 4
    eval_batch_size: int = 8
    metrics_max_count: int = 12800
    clip_model_name_or_path: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    pickscore_model_name_or_path: str = "yuvalkirstain/PickScore_v1"
    image_reward_path: str = "ImageReward-v1.0"
    coco_ref_stats_path: str = "stats/fid_stats_mscoco256_val.npz"
    mjhq_ref_stats_path: str = "stats/fid_stats_mjhq256_val.npz"
    inception_path: str = "stats/pt_inception-2015-12-05-6726825d.pth"
    guidance: float = 4
    top_k: int = 400
    top_p: float = 0.95

    # RoPE
    rope: bool = True
    rope_theta: float = 10000
    rope_size: int = 128

    # architecture args
    use_swiglu_ffn = True
    use_ar = False  # False: using non-autoregressive switti
    use_crop_cond = False
    
    # FSDP and gradient checkpointing
    use_fsdp: bool = True
    use_gradient_checkpointing: bool = False

    # logging args
    save_iters: int = 4000
    global_save_iters: int = 40000
    log_iters: int = 200
    log_images_iters: int = 1000
    uncond_proba: float = 0.1

    # VAE
    vfast: int = (
        0  # torch.compile VAE; =0: not compile; 1: compile with 'reduce-overhead'; 2: compile with 'max-autotune'
    )
    vqvae_vocab_size: int = 4096
    vqvae_channel_dim: int = 32
    vqvae_n_channels: int = 160
    vqvae_share_quant_resi: int = 4
    embed_noise_std: float | None = None
    
    # Switti
    tfast: int = (
        0  # torch.compile Switti; =0: not compile; 1: compile with 'reduce-overhead'; 2: compile with 'max-autotune'
    )
    depth: int = 16  # number of blocks in Switti transformer
    # Switti initialization
    ini: float = -1  # -1: automated model parameter initialization
    hd: float = 0.02  # head.w *= hd
    aln: float = 0.5  # the multiplier of ada_lin.w's initialization
    alng: float = 1e-5  # the multiplier of ada_lin.w[gamma channels]'s initialization
    drop_path_rate: float = 0  # dropout after residual in xattn
    
    # Optimization
    fp16: int = 0  # 1: using fp16, 2: bf16
    tblr: float = 1e-4  # base lr
    tlr: float = None  # lr = base lr * (bs / 256)
    twd: float = 0  # initial wd
    twde: float = 0  # final wd, =twde or twd
    tclip: float = 1.0  # <=0 for not using grad clip
    ls: float = 0.0  # label smooth

    bs: int = 768  # global batch size
    grad_accum: int = 1
    batch_size: int = (
        0  # [automatically set; don't specify this] batch size per GPU = round(args.bs / args.ac / dist.get_world_size() / 8) * 8
    )
    glb_batch_size: int = (
        0  # [automatically set; don't specify this] global batch size = args.batch_size * dist.get_world_size()
    )

    max_iters: int = 1000000
    wp: float = 0
    wp0: float = 0.005  # initial lr ratio at the begging of lr warm up
    wpe: float = 1  # final lr ratio at the end of training
    sche: str = "lin0"  # lr schedule

    # optimizer args
    afuse: bool = True  # fused adamw
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95

    # other hps
    saln: bool = False  # whether to use shared adaln
    anorm: bool = True  # whether to use L2 normalized attention

    # data
    pn: str = "1_2_3_4_5_6_8_10_13_16"
    patch_size: int = 16
    patch_nums: tuple = (
        None  # [automatically set; don't specify this] = tuple(map(int, args.pn.replace('-', '_').split('_')))
    )
    resos: tuple = (
        None  # [automatically set; don't specify this] = tuple(pn * args.patch_size for pn in args.patch_nums)
    )

    data_load_reso: int = (
        None  # [automatically set; don't specify this] would be max(patch_nums) * patch_size
    )
    mid_reso: float = (
        1.125  # aug: first resize to mid_reso = 1.125 * data_load_reso, then crop to data_load_reso
    )
    hflip: bool = False  # augmentation: horizontal flip
    workers: int = (
        0  # num workers; 0: auto, -1: don't use multiprocessing in DataLoader
    )

    # would be automatically set in runtime
    cmd: str = " ".join(sys.argv[1:])  # [automatically set; don't specify this]
    branch: str = (
        subprocess.check_output(
            f"git symbolic-ref --short HEAD 2>/dev/null || git rev-parse HEAD",
            shell=True,
        )
        .decode("utf-8")
        .strip()
        or "[unknown]"
    )  # [automatically set; don't specify this]
    commit_id: str = (
        subprocess.check_output(f"git rev-parse HEAD", shell=True)
        .decode("utf-8")
        .strip()
        or "[unknown]"
    )  # [automatically set; don't specify this]
    commit_msg: str = (
        subprocess.check_output(f"git log -1", shell=True)
        .decode("utf-8")
        .strip()
        .splitlines()
        or ["[unknown]"]
    )[
        -1
    ].strip()  # [automatically set; don't specify this]
    acc_mean: float = None  # [automatically set; don't specify this]
    acc_tail: float = None  # [automatically set; don't specify this]
    L_mean: float = None  # [automatically set; don't specify this]
    L_tail: float = None  # [automatically set; don't specify this]
    vacc_mean: float = None  # [automatically set; don't specify this]
    vacc_tail: float = None  # [automatically set; don't specify this]
    vL_mean: float = None  # [automatically set; don't specify this]
    vL_tail: float = None  # [automatically set; don't specify this]
    grad_norm: float = None  # [automatically set; don't specify this]
    cur_lr: float = None  # [automatically set; don't specify this]
    cur_wd: float = None  # [automatically set; don't specify this]
    cur_it: str = ""  # [automatically set; don't specify this]
    cur_ep: str = ""  # [automatically set; don't specify this]
    remain_time: str = ""  # [automatically set; don't specify this]
    finish_time: str = ""  # [automatically set; don't specify this]

    # environment
    local_out_dir_path: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "local_output"
    )  # [automatically set; don't specify this]
    tb_log_dir_path: str = "...tb-..."  # [automatically set; don't specify this]
    checkpoint_path: str = "...checkpoints"
    log_txt_path: str = "..."  # [automatically set; don't specify this]
    last_ckpt_path: str = "..."  # [automatically set; don't specify this]

    tf32: bool = True  # whether to use TensorFloat32
    device: str = "cpu"  # [automatically set; don't specify this]
    seed: int = None  # seed

    def seed_everything(self, benchmark: bool):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = benchmark
        if self.seed is None:
            torch.backends.cudnn.deterministic = False
        else:
            torch.backends.cudnn.deterministic = True
            seed = self.seed * dist.get_world_size() + dist.get_rank()
            os.environ["PYTHONHASHSEED"] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

    same_seed_for_all_ranks: int = 0  # this is only for distributed sampler

    def get_different_generator_for_each_rank(
        self,
    ) -> Optional[torch.Generator]:  # for random augmentation
        if self.seed is None:
            return None
        g = torch.Generator()
        g.manual_seed(self.seed * dist.get_world_size() + dist.get_rank())
        return g

    def compile_model(self, m, fast):
        if fast == 0:
            return m
        return (
            torch.compile(
                m,
                mode={
                    1: "reduce-overhead",
                    2: "max-autotune",
                    3: "default",
                }[fast],
            )
            if hasattr(torch, "compile")
            else m
        )

    def state_dict(self, key_ordered=True) -> Union[OrderedDict, dict]:
        d = (OrderedDict if key_ordered else dict)()
        # self.as_dict() would contain methods, but we only need variables
        for k in self.class_variables.keys():
            if k not in {"device"}:  # these are not serializable
                d[k] = getattr(self, k)
        return d

    @staticmethod
    def set_tf32(tf32: bool):
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = bool(tf32)
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high" if tf32 else "highest")
                print(
                    f"[tf32] [precis] torch.get_float32_matmul_precision(): {torch.get_float32_matmul_precision()}"
                )
            print(
                f"[tf32] [ conv ] torch.backends.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}"
            )
            print(
                f"[tf32] [matmul] torch.backends.cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}"
            )

    def dump_log(self):
        if not dist.is_local_master():
            return
        if "1/" in self.cur_ep:  # first time to dump log
            with open(self.log_txt_path, "w") as fp:
                json.dump(
                    {
                        "is_master": dist.is_master(),
                        "name": self.exp_name,
                        "cmd": self.cmd,
                        "commit": self.commit_id,
                        "branch": self.branch,
                        "tb_log_dir_path": self.tb_log_dir_path,
                    },
                    fp,
                    indent=0,
                )
                fp.write("\n")

        log_dict = {}
        for k, v in {
            "it": self.cur_it,
            "ep": self.cur_ep,
            "lr": self.cur_lr,
            "wd": self.cur_wd,
            "grad_norm": self.grad_norm,
            "L_mean": self.L_mean,
            "L_tail": self.L_tail,
            "acc_mean": self.acc_mean,
            "acc_tail": self.acc_tail,
            "vL_mean": self.vL_mean,
            "vL_tail": self.vL_tail,
            "vacc_mean": self.vacc_mean,
            "vacc_tail": self.vacc_tail,
            "remain_time": self.remain_time,
            "finish_time": self.finish_time,
        }.items():
            if hasattr(v, "item"):
                v = v.item()
            log_dict[k] = v
        with open(self.log_txt_path, "a") as fp:
            fp.write(f"{log_dict}\n")

    def __str__(self):
        s = []
        for k in self.class_variables.keys():
            if k not in {"device", "dbg_ks_fp"}:  # these are not serializable
                s.append(f"  {k:20s}: {getattr(self, k)}")
        s = "\n".join(s)
        return f"{{\n{s}\n}}\n"


def init_dist_and_get_args():
    for i in range(len(sys.argv)):
        if sys.argv[i].startswith("--local-rank=") or sys.argv[i].startswith(
            "--local_rank="
        ):
            del sys.argv[i]
            break
    args = Args(explicit_bool=True).parse_args(known_only=True)

    # warn args.extra_args
    if len(args.extra_args) > 0:
        print(
            f"======================================================================================"
        )
        print(
            f"=========================== WARNING: UNEXPECTED EXTRA ARGS ===========================\n{args.extra_args}"
        )
        print(
            f"=========================== WARNING: UNEXPECTED EXTRA ARGS ==========================="
        )
        print(
            f"======================================================================================\n\n"
        )

    # init torch distributed
    os.makedirs(args.local_out_dir_path, exist_ok=True)
    misc.init_distributed_mode(local_out_path=args.local_out_dir_path)

    # set env
    args.set_tf32(args.tf32)
    args.seed_everything(benchmark=True)

    # update args: data loading
    args.device = dist.get_device()
    args.pn = RESOLUTION_PATCH_NUMS_MAPPING.get(int(args.pn), args.pn)
    args.patch_nums = tuple(map(int, args.pn.replace("-", "_").split("_")))
    args.resos = tuple(pn * args.patch_size for pn in args.patch_nums)
    args.data_load_reso = max(args.resos)

    # update args: bs and lr
    bs_per_gpu = round(args.bs / dist.get_world_size())
    args.batch_size = bs_per_gpu
    args.bs = args.glb_batch_size = args.batch_size * dist.get_world_size()
    args.workers = min(max(0, args.workers), args.batch_size)

    args.tlr = args.tblr
    args.twde = args.twde or args.twd

    tb_name = "tb_logs"
    args.tb_log_dir_path = os.path.join(args.local_out_dir_path, tb_name)

    return args
