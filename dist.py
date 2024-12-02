"""
Helpers for distributed training.
"""
import os
import socket

import torch as th
import torch.distributed as dist
from torch.distributed import barrier, is_initialized, broadcast

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3

import datetime
import os

import socket
from contextlib import closing


def find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def check_if_port_open(port: int) -> bool:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        try:
            s.bind(("", port))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return True
        except OSError:
            return False


def initialized():
    return dist.is_initialized()


def finalize():
    if dist.is_initialized():
        dist.destroy_process_group()

    
def initialize():
    is_mpirun = not (
        "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
        and "MASTER_ADDR" in os.environ
        and "MASTER_PORT" in os.environ
    )
    
    if is_mpirun:
        from mpi4py import MPI
        import subprocess

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()

        master_addr = None
        master_port = None
        if rank == 0:
            hostname_cmd = ["hostname -I"]
            result = subprocess.check_output(hostname_cmd, shell=True)
            master_addr = result.decode("utf-8").split()[0]

            base_port = os.environ.get(
                "MASTER_PORT", "29500"
            )  # TORCH_DISTRIBUTED_DEFAULT_PORT
            if check_if_port_open(int(base_port)):
                master_port = base_port
            else:
                master_port = find_free_port()

        master_addr = comm.bcast(master_addr, root=0)
        master_port = comm.bcast(master_port, root=0)
        # Determine local rank by assuming hostnames are unique
        proc_name = MPI.Get_processor_name()
        all_procs = comm.allgather(proc_name)
        local_rank = sum([i == proc_name for i in all_procs[:rank]])
        uniq_proc_names = set(all_procs)
        host_rank = sorted(uniq_proc_names).index(proc_name)
        
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["HOST_RANK"] = str(host_rank)
        os.environ["NUM_HOSTS"] = str(len(uniq_proc_names))

        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["OMP_NUM_THREADS"] = "1"
    
    # Initialize torch distributed
    backend = "gloo" if not th.cuda.is_available() else "nccl"
    dist.init_process_group(backend=backend, timeout=datetime.timedelta(0, 3600))
    th.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    
    if is_mpirun and dist.get_rank() == 0:
        print("Distributed setup")
        print("LOCAL_RANK", os.environ['LOCAL_RANK'])
        print("HOST_RANK", os.environ['HOST_RANK'])
        print("NUM_HOSTS", os.environ['NUM_HOSTS'])
        print("WORLD_SIZE", os.environ['WORLD_SIZE'])


def local_host_gather(data):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    host_rank = os.environ["HOST_RANK"]
    all_data = comm.allgather((host_rank, data))
    return [d[1] for d in all_data if d[0] == host_rank]


def in_distributed_mode():
    return dist is not None


def is_master():
    return get_rank() == 0


def is_local_master():
    return get_local_rank() == 0


def get_rank():
    return dist.get_rank() if in_distributed_mode() else 0


def get_local_rank():
    return int(os.environ["LOCAL_RANK"])


def worker_host_idx():
    return int(os.environ["HOST_RANK"])


def num_hosts():
    return int(os.environ['NUM_HOSTS'])


def get_world_size():
    return dist.get_world_size() if in_distributed_mode() else 1


def gpu_visible_device_list():
    return str(dist.get_rank()) if in_distributed_mode() else None


def get_device():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)
        
        
def allreduce(t: th.Tensor, async_op=False):
    if dist.is_initialized():
        if not t.is_cuda:
            cu = t.detach().cuda()
            ret = dist.all_reduce(cu, async_op=async_op)
            t.copy_(cu.cpu())
        else:
            ret = dist.all_reduce(t, async_op=async_op)
        return ret
    return None


def allgather(t: th.Tensor, cat=True):
    if dist.is_initialized():
        if not t.is_cuda:
            t = t.cuda()
        ls = [th.empty_like(t) for _ in range(get_world_size())]
        dist.all_gather(ls, t)
    else:
        ls = [t]
    if cat:
        ls = th.cat(ls, dim=0)
    return ls
