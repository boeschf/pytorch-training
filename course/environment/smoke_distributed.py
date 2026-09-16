"""Verify a bounded NCCL collective across all local GPUs."""

from __future__ import annotations

import os
from datetime import timedelta

import torch
from torch import distributed


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    distributed.init_process_group(backend="nccl", timeout=timedelta(seconds=60))
    try:
        value = torch.tensor(float(rank), device=f"cuda:{local_rank}")
        distributed.all_reduce(value)
        expected = world_size * (world_size - 1) / 2
        actual = value.item()
        if actual != expected:
            raise RuntimeError(f"NCCL all-reduce returned {actual}, expected {expected}")
        distributed.barrier(device_ids=[local_rank])
        if rank == 0:
            print(
                f"NCCL all-reduce: PASS ({world_size} ranks, sum={actual:g}, "
                f"backend={distributed.get_backend()})"
            )
    finally:
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
