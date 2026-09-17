"""Executable tensor and autograd concepts adapted from the legacy overview."""

from __future__ import annotations

from collections.abc import Callable

import torch

from lessons.prep.tensor_device.reference import DeviceRequest, resolve_device

Objective = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def autograd_report(
    objective: Objective,
    requested_device: DeviceRequest = "auto",
) -> dict[str, object]:
    """Differentiate an objective and compare it with its analytical gradient."""
    device = resolve_device(requested_device)
    inputs = torch.tensor([1.0, 2.0, -3.0], device=device, requires_grad=True)
    targets = torch.tensor([0.5, -1.0, 2.0], device=device)
    loss = objective(inputs, targets)
    if loss.ndim:
        raise ValueError("the objective must return one scalar")
    loss.backward()
    expected = 2 * (inputs.detach() - targets)
    gradient = inputs.grad
    correct = gradient is not None and torch.allclose(gradient, expected)
    return {
        "schema_version": 1,
        "requested_device": requested_device,
        "selected_device": device.type,
        "loss": round(loss.detach().item(), 6),
        "gradient": [] if gradient is None else gradient.detach().cpu().tolist(),
        "expected_gradient": expected.cpu().tolist(),
        "status": "pass" if correct else "fail",
    }
