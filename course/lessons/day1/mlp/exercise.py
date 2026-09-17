"""Complete one explicit MLP optimization step.

Outcome: implement zero-grad, forward, loss, backward, and optimizer-step operations.
Prerequisite: tensor/device preparation and the introduction to ``torch.nn.Module``.
Expected runtime: under 30 seconds after completing the gaps.
Resources: CPU or one GPU; no network or downloaded dataset.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from lessons.day1.mlp.reference import MLP, evaluate, make_dataset
from lessons.prep.tensor_device.reference import DeviceRequest, resolve_device

OptimizationStep = Callable[
    [nn.Module, torch.optim.Optimizer, nn.Module, torch.Tensor, torch.Tensor],
    float,
]


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


def run_exercise(
    step: OptimizationStep,
    *,
    requested_device: DeviceRequest = "auto",
) -> int:
    """Run the participant step repeatedly and check the learned model."""
    torch.manual_seed(7)
    device = resolve_device(requested_device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(7)
    torch.use_deterministic_algorithms(True)

    split = make_dataset(samples=512, seed=7)
    features = split.train_features.to(device)
    targets = split.train_targets.to(device)
    eval_features = split.eval_features.to(device)
    eval_targets = split.eval_targets.to(device)
    model = MLP().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_function = nn.CrossEntropyLoss()

    print("epoch | train loss | eval loss | eval accuracy")
    accuracy = 0.0
    for epoch in range(201):
        if epoch:
            step(model, optimizer, loss_function, features, targets)
        if epoch % 20 == 0:
            train_loss, _ = evaluate(model, features, targets, loss_function)
            eval_loss, accuracy = evaluate(model, eval_features, eval_targets, loss_function)
            print(f"{epoch:>5} | {train_loss:>10.6f} | {eval_loss:>9.6f} | {accuracy:>12.1%}")
            model.train()

    if accuracy < 0.95:
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
