"""Complete one explicit MLP optimization step.

Outcome: implement zero-grad, forward, loss, backward, and optimizer-step operations.
Prerequisite: tensor/device preparation and the introduction to ``torch.nn.Module``.
Expected runtime: under 30 seconds after completing the gaps.
Resources: CPU or one GPU; no network or downloaded dataset.
"""

from __future__ import annotations

import torch
from torch import nn

from lessons.day1.mlp.reference import (
    OptimizationStep,
    TrainingSnapshot,
    run_mlp,
)
from lessons.prep.tensor_device.reference import DeviceRequest


def optimization_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Perform one optimization step and return its loss.

    Replace the exception with the five operations listed in the exercise outcome.
    Keep the order explicit; do not introduce a training framework.
    """
    raise NotImplementedError("complete the five operations in optimization_step")


def _print_snapshot(snapshot: TrainingSnapshot) -> None:
    """Print one row of exercise progress."""
    print(
        f"{snapshot.epoch:>5} | {snapshot.train_loss:>10.6f} | "
        f"{snapshot.eval_loss:>9.6f} | {snapshot.eval_accuracy:>12.1%}"
    )


def run_exercise(
    step: OptimizationStep,
    *,
    requested_device: DeviceRequest = "auto",
) -> int:
    """Run the participant step through the shared training loop."""
    print("epoch | train loss | eval loss | eval accuracy")
    run = run_mlp(
        requested_device=requested_device,
        seed=7,
        epochs=200,
        learning_rate=0.05,
        samples=512,
        snapshot_interval=20,
        training_step=step,
        on_snapshot=_print_snapshot,
    )

    if run.history[-1].eval_accuracy < 0.95:
        print("Status: FAIL — revisit the optimization step")
        return 1
    print("Status: PASS")
    return 0


def main(requested_device: DeviceRequest = "auto") -> int:
    """Run the exercise and report an incomplete implementation without a traceback."""
    try:
        return run_exercise(optimization_step, requested_device=requested_device)
    except NotImplementedError as error:
        print(f"Status: INCOMPLETE — {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
