"""Inspect the Python, PyTorch, CUDA, and course-image environment."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

COURSE_ROOT = Path(__file__).resolve().parents[2]
IMAGE_ENV = COURSE_ROOT / "environment" / "image.env"
RUNTIME_ENV_VARS = (
    "CUDA_HOME",
    "CUDA_VERSION",
    "NCCL_VERSION",
    "NCCL_NET_PLUGIN",
    "SLURM_JOB_ID",
    "SLURM_NNODES",
    "SLURM_NTASKS",
    "SLURM_PROCID",
    "SLURM_LOCALID",
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
)


def read_image_metadata(path: Path = IMAGE_ENV) -> dict[str, str]:
    """Read the checked-in shell-compatible image metadata without executing it."""
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key:
            raise ValueError(f"invalid image metadata line: {raw_line!r}")
        values[key] = value
    return values


def safely(call: Callable[[], Any]) -> Any:
    """Return a probe result without letting optional runtime probes abort the report."""
    try:
        return call()
    except Exception as error:  # noqa: BLE001 - runtime probes may raise backend errors.
        return {"error": f"{type(error).__name__}: {error}"}


def collect_torch_report() -> dict[str, Any]:
    """Collect PyTorch details, allowing the host authoring environment to lack PyTorch."""
    try:
        import torch
    except Exception as error:  # noqa: BLE001 - imports may fail while loading native libraries.
        return {
            "available": False,
            "import_error": f"{type(error).__name__}: {error}",
        }

    torch_path = Path(torch.__file__).resolve()
    course_venv = (COURSE_ROOT / ".venv").resolve()
    cuda_available = safely(torch.cuda.is_available)
    device_count = safely(torch.cuda.device_count)
    devices: list[dict[str, Any]] = []
    if cuda_available is True and isinstance(device_count, int):
        for index in range(device_count):
            properties = safely(lambda index=index: torch.cuda.get_device_properties(index))
            if isinstance(properties, dict):
                devices.append({"index": index, **properties})
            else:
                devices.append(
                    {
                        "index": index,
                        "name": properties.name,
                        "total_memory_bytes": properties.total_memory,
                        "compute_capability": f"{properties.major}.{properties.minor}",
                    }
                )

    distributed_available = safely(torch.distributed.is_available)
    nccl_available: Any = False
    if distributed_available is True:
        nccl_available = safely(torch.distributed.is_nccl_available)

    return {
        "available": True,
        "version": torch.__version__,
        "location": str(torch_path),
        "inside_course_venv": course_venv in torch_path.parents,
        "compiled_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "device_count": device_count,
        "devices": devices,
        "distributed_available": distributed_available,
        "nccl_available": nccl_available,
        "nccl_version": safely(torch.cuda.nccl.version),
    }


def collect_report() -> dict[str, Any]:
    """Collect a serializable environment report."""
    return {
        "course": {
            "root": str(COURSE_ROOT),
            "image": read_image_metadata(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "prefix": sys.prefix,
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "hostname": platform.node(),
        },
        "runtime_environment": {
            name: value for name in RUNTIME_ENV_VARS if (value := os.environ.get(name)) is not None
        },
        "torch": collect_torch_report(),
    }


def validate_report(
    report: dict[str, Any],
    *,
    require_torch: bool,
    require_cuda: bool,
    minimum_gpus: int,
    require_image_torch: bool,
) -> list[str]:
    """Return unmet environment requirements."""
    errors: list[str] = []
    python_version = tuple(int(part) for part in report["python"]["version"].split(".")[:2])
    if python_version != (3, 12):
        errors.append(f"Python 3.12 required, found {report['python']['version']}")

    torch_report = report["torch"]
    image = report["course"]["image"]
    if require_torch and not torch_report["available"]:
        errors.append("PyTorch import required")
    if require_cuda and torch_report.get("cuda_available") is not True:
        errors.append("CUDA-enabled PyTorch runtime required")

    device_count = torch_report.get("device_count", 0)
    if minimum_gpus and (not isinstance(device_count, int) or device_count < minimum_gpus):
        errors.append(f"at least {minimum_gpus} CUDA devices required, found {device_count}")

    if require_image_torch:
        if not torch_report["available"]:
            errors.append("image-provided PyTorch required, but PyTorch import failed")
        else:
            if torch_report.get("inside_course_venv"):
                errors.append(
                    "PyTorch resolves inside course/.venv; use the image-provided installation"
                )
            expected_version = image.get("COURSE_IMAGE_PYTORCH_VERSION")
            if expected_version and not str(torch_report["version"]).startswith(expected_version):
                errors.append(
                    f"expected image PyTorch {expected_version}, found {torch_report['version']}"
                )

    for key in ("COURSE_IMAGE_REPOSITORY", "COURSE_IMAGE_TAG", "COURSE_IMAGE_DIGEST"):
        if not image.get(key):
            errors.append(f"missing pinned image field {key}")
    return errors


def render_human(report: dict[str, Any], errors: list[str]) -> str:
    """Render the concise interactive report."""
    torch_report = report["torch"]
    image = report["course"]["image"]
    lines = [
        f"Python: {report['python']['version']} ({report['python']['executable']})",
        f"Platform: {report['host']['platform']}",
        f"Course image: {image['COURSE_IMAGE_REPOSITORY']}:{image['COURSE_IMAGE_TAG']}",
        f"Image digest: {image['COURSE_IMAGE_DIGEST']}",
    ]
    if torch_report["available"]:
        lines.extend(
            (
                f"PyTorch: {torch_report['version']} ({torch_report['location']})",
                f"Compiled CUDA: {torch_report['compiled_cuda']}",
                f"CUDA available: {torch_report['cuda_available']}",
                f"CUDA devices: {torch_report['device_count']}",
                f"NCCL available: {torch_report['nccl_available']}",
            )
        )
        for device in torch_report["devices"]:
            lines.append(
                "GPU {index}: {name}, {compute_capability}, {total_memory_bytes} bytes".format(
                    **device
                )
            )
    else:
        lines.append(f"PyTorch: unavailable ({torch_report['import_error']})")

    lines.append("Status: PASS" if not errors else "Status: FAIL")
    lines.extend(f"- {error}" for error in errors)
    return "\n".join(lines)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add environment-doctor arguments to a command parser."""
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--require-torch", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--minimum-gpus", type=int, default=0)
    parser.add_argument(
        "--require-image-torch",
        action="store_true",
        help="reject a PyTorch installation located inside course/.venv",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return parser.parse_args(argv)


def run_doctor(args: argparse.Namespace) -> int:
    """Collect, validate, and print one environment report."""
    report = collect_report()
    errors = validate_report(
        report,
        require_torch=args.require_torch,
        require_cuda=args.require_cuda,
        minimum_gpus=args.minimum_gpus,
        require_image_torch=args.require_image_torch,
    )
    if args.json:
        print(json.dumps({**report, "errors": errors}, indent=2, sort_keys=True))
    else:
        print(render_human(report, errors))
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    return run_doctor(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
