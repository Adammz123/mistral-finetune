import logging
import os
from functools import lru_cache
from typing import List, Union

import torch
import torch.distributed as dist

logger = logging.getLogger("distributed")

BACKEND = "nccl"


@lru_cache()
def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


@lru_cache()
def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def visible_devices() -> List[int]:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return [int(d) for d in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    return list(range(torch.cuda.device_count()))


def set_device():
    logger.info(f"torch.cuda.device_count: {torch.cuda.device_count()}")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    logger.info(f"local rank: {int(os.environ['LOCAL_RANK'])}")

    assert torch.cuda.is_available()

    if torch.cuda.device_count() == 1:
        # gpus-per-task set to 1
        torch.cuda.set_device(0)
        return

    local_rank = int(os.environ["LOCAL_RANK"])
    logger.info(f"Set cuda device to {local_rank}")

    assert 0 <= local_rank < torch.cuda.device_count(), (
        local_rank,
        torch.cuda.device_count(),
    )
    torch.cuda.set_device(local_rank)


def avg_aggregate(metric: Union[float, int]) -> Union[float, int]:
    if not dist.is_available() or not dist.is_initialized():
        return metric
    buffer = torch.tensor([metric], dtype=torch.float32, device="cuda")
    dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
    return buffer[0].item() / get_world_size()


def is_torchrun() -> bool:
    return "TORCHELASTIC_RESTART_COUNT" in os.environ
