"""Self-checking tensor and device diagnostic for course preparation."""

from __future__ import annotations

from typing import Literal

import torch

DeviceRequest = Literal["auto", "cpu", "cuda"]


def resolve_device(requested: DeviceRequest) -> torch.device:
    """Resolve an explicit or automatic device request."""
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# region tensor-device-diagnostic
def tensor_device_report(requested: DeviceRequest = "auto") -> dict[str, object]:
    """Exercise tensor creation and transfer, returning JSON-safe evidence."""
    device = resolve_device(requested)
    host_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    tensor = host_tensor.to(device)
    identity = torch.eye(2, dtype=tensor.dtype, device=device)
    result = tensor @ identity
    preserved = torch.equal(result.cpu(), host_tensor)

    return {
        "schema_version": 1,
        "requested_device": requested,
        "selected_device": device.type,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "requires_grad": tensor.requires_grad,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "identity_preserved_values": preserved,
        "status": "pass" if preserved else "fail",
    }


# endregion tensor-device-diagnostic
