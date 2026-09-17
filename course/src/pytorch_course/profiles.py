"""Runtime profile names and metadata shared by setup, diagnostics, and kernels."""

from __future__ import annotations

import sys
from pathlib import Path

PROFILE_MARKER = ".course-profile"
PROFILE_LABELS = {
    "cpu": "CPU",
    "cuda": "CUDA",
    "alps-gh200": "Alps GH200",
}
PROFILES = tuple(PROFILE_LABELS)
PORTABLE_TORCH_VERSION = "2.11.0"
PORTABLE_CUDA_VERSION = "12.8"


def read_profile(prefix: str | Path | None = None) -> str | None:
    """Return the profile recorded in a virtual environment, if valid."""
    environment = Path(prefix or sys.prefix)
    marker = environment / PROFILE_MARKER
    if not marker.is_file():
        return None
    profile = marker.read_text(encoding="utf-8").strip()
    return profile if profile in PROFILE_LABELS else None


def require_profile(prefix: str | Path | None = None) -> str:
    """Return the active managed profile or raise a useful setup error."""
    profile = read_profile(prefix)
    if profile is None:
        names = ", ".join(PROFILES)
        raise RuntimeError(
            f"active Python is not a managed course environment; create one of: {names}"
        )
    return profile


def kernel_identity(profile: str) -> tuple[str, str]:
    """Return stable Jupyter kernel and display names for one profile."""
    label = PROFILE_LABELS[profile]
    return f"cscs-pytorch-course-{profile}", f"CSCS PyTorch Course — {label}"
