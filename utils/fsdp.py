import math
import os

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

import dist
from utils.misc import glob_with_latest_modified_first


def bcast_state_dict(state_dict):
    for data in state_dict.values():
        if isinstance(data, torch.Tensor):
            dist.broadcast(data, 0)
        elif isinstance(data, dict):
            bcast_state_dict(data)
        else:
            Exception(f"Unsupported type: {type(data)}")


def save_model_state(cur_iter: int, args, model: torch.nn.Module):
    """Save model, optimizer state dict and training args to be loaded via load_training_state"""

    # Save model state
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=False, rank0_only=True),
    ):
        model_state_dict = model.state_dict()
        if dist.is_master():
            os.makedirs(args.local_out_dir_path, exist_ok=True)
            model_save_path = os.path.join(
                args.local_out_dir_path, "model_state_dict.pt"
            )
            torch.save(model_state_dict, model_save_path)

            metadata = {"iter": cur_iter, "args": args.state_dict()}
            metadata_save_path = os.path.join(args.local_out_dir_path, "metadata.pt")
            torch.save(metadata, metadata_save_path)

            # Save global checkpoints
            if cur_iter % args.global_save_iters == 0:
                model_save_path = os.path.join(
                    args.local_out_dir_path, f"model_{cur_iter}_state_dict.pt"
                )
                torch.save(model_state_dict, model_save_path)

                metadata_save_path = os.path.join(
                    args.local_out_dir_path, f"metadata_{cur_iter}.pt"
                )
                torch.save(metadata, metadata_save_path)

            print(f"Saved model and optimizer state dicts to {args.local_out_dir_path}")


def load_model_state(args, model: torch.nn.Module) -> int:
    """Load model, optimizer state dict and metadata saved via save_training_state; update parameters in-place"""
    model_path = os.path.join(args.local_out_dir_path, f"model_state_dict.pt")
    metadata_path = os.path.join(args.local_out_dir_path, "metadata.pt")

    if not os.path.exists(model_path):
        start_iter = 0
        file = os.path.join(args.local_out_dir_path, "*.pt")
        all_ckpt = glob_with_latest_modified_first(file)
        if dist.is_master():
            print(f".pt files in {args.local_out_dir_path}: {all_ckpt}")
            print(f"[auto_resume failed] start training from scratch {start_iter}")
    else:
        model_state_dict = torch.load(model_path, map_location="cpu")
        metadata = torch.load(metadata_path, map_location="cpu")
        if args.resos[-1] != metadata["args"]["resos"][-1]:
            # rewrite registered buffers for a different target resolution
            L = sum([pn * pn for pn in args.patch_nums])
            C = args.depth * 64
            d = torch.cat(
                [torch.full((pn * pn,), i) for i, pn in enumerate(args.patch_nums)]
            ).view(1, L, 1)
            dT = d.transpose(1, 2)  # dT: 11L
            model_state_dict["lvl_1L"] = dT[:, 0].contiguous()
            attn_bias_for_masking = torch.where(d >= dT, 0.0, -torch.inf).reshape(
                1, 1, L, L
            )
            model_state_dict["attn_bias_for_masking"] = attn_bias_for_masking

            if not args.rope:
                # absolute position embedding
                init_std = math.sqrt(1 / C / 3)
                pos_1LC = []
                for i, pn in enumerate(args.patch_nums):
                    pe = torch.empty(1, pn * pn, C)
                    nn.init.trunc_normal_(pe, mean=0, std=init_std)
                    pos_1LC.append(pe)
                pos_1LC = torch.cat(pos_1LC, dim=1)  # 1, L, C
                assert tuple(pos_1LC.shape) == (1, L, C)
                model_state_dict["pos_1LC"] = pos_1LC

        model.load_state_dict(model_state_dict)

        print(f"Loaded training state from {args.local_out_dir_path}: {metadata}")
        
        start_iter = metadata["iter"] + 1  # start from iter + 1 to avoid double evals
        print(f"[auto_resume success] resume from iteration {start_iter}")

    dist.barrier()
    bcast_state_dict(model.state_dict())
    start_iter_t = torch.tensor(start_iter, device=dist.get_device())
    dist.broadcast(start_iter_t, 0)
    start_iter = start_iter_t.item()
    dist.barrier()
    return start_iter
