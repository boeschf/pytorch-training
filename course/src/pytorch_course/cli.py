"""Unified participant, instructor, and developer commands."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from pytorch_course import doctor

COURSE_ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT_DIR = COURSE_ROOT / "environment"


def run(command: list[str], *, environment: dict[str, str] | None = None) -> int:
    """Run one checked-in course command from the course root."""
    completed = subprocess.run(
        command,
        cwd=COURSE_ROOT,
        env=environment,
        check=False,
    )
    return completed.returncode


def run_slides(action: str) -> int:
    """Set up or execute the pinned Slidev toolchain."""
    setup = ENVIRONMENT_DIR / "create-slide-env.sh"
    runner = ENVIRONMENT_DIR / "run-slides.sh"
    if action == "setup":
        return run([str(setup)])

    if (
        not (COURSE_ROOT / ".nodeenv" / "bin" / "node").is_file()
        or not (COURSE_ROOT / "slides" / "node_modules" / ".package-lock.json").is_file()
    ):
        setup_result = run([str(setup)])
        if setup_result:
            return setup_result

    if action == "audit":
        return run([str(runner), "audit"])
    return run([str(runner), "run", action])


def install_kernel() -> int:
    """Install the current course environment as a user Jupyter kernel."""
    return run(
        [
            sys.executable,
            "-m",
            "ipykernel",
            "install",
            "--user",
            "--name",
            "cscs-pytorch-course",
            "--display-name",
            "CSCS PyTorch Course",
        ]
    )


def run_alps_smoke(args: argparse.Namespace) -> int:
    """Request a bounded Alps allocation and verify the canonical runtime."""
    environment = os.environ.copy()
    environment["COURSE_GPUS_PER_NODE"] = str(args.gpus)
    smoke = "smoke-node.sh" if args.gpus > 1 else "smoke-gpu.sh"
    return run(
        [
            str(ENVIRONMENT_DIR / "run-alps.sh"),
            f"--account={args.account}",
            f"--partition={args.partition}",
            "--nodes=1",
            "--ntasks=1",
            f"--gpus-per-node={args.gpus}",
            f"--time={args.time}",
            str(ENVIRONMENT_DIR / smoke),
        ],
        environment=environment,
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="course", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    doctor_parser = commands.add_parser(
        "doctor", help="inspect Python, PyTorch, CUDA, and image metadata"
    )
    doctor.add_arguments(doctor_parser)

    kernel = commands.add_parser("kernel", help="manage the participant Jupyter kernel")
    kernel.add_argument("action", choices=("install",))

    slides = commands.add_parser("slides", help="manage the pinned Slidev environment")
    slides.add_argument("action", choices=("setup", "build", "dev", "preview", "export", "audit"))

    alps = commands.add_parser("alps-smoke", help="verify the canonical image on an Alps node")
    alps.add_argument("--account", required=True)
    alps.add_argument("--partition", default="debug")
    alps.add_argument("--gpus", type=int, choices=range(1, 5), default=4)
    alps.add_argument("--time", default="00:05:00")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        return doctor.run_doctor(args)
    if args.command == "kernel":
        return install_kernel()
    if args.command == "slides":
        return run_slides(args.action)
    if args.command == "alps-smoke":
        return run_alps_smoke(args)
    parser.error(f"unknown command: {args.command}")
