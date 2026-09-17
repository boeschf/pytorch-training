"""Complete solution for the adjacent MLP optimization-step exercise."""

from __future__ import annotations

import torch
from torch import nn

from lessons.day1.mlp.exercise import run_exercise


# region optimization-step-solution
def optimization_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Perform one explicit optimization step and return its loss."""
    # Gradients accumulate by default, so clear the previous epoch first.
    optimizer.zero_grad(set_to_none=True)
    # The forward pass builds the graph and returns one score per class.
    logits = model(features)
    # Cross-entropy compares those raw scores with integer class labels.
    loss = loss_function(logits, targets)
    # Backpropagation writes gradients into every trainable parameter.
    loss.backward()
    # SGD consumes those gradients and mutates the model parameters.
    optimizer.step()
    return loss.item()


# endregion optimization-step-solution


def main() -> int:
    """Run the solution through the same self-check as the participant exercise."""
    return run_exercise(optimization_step)


if __name__ == "__main__":
    raise SystemExit(main())
