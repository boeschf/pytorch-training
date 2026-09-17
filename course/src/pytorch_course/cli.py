"""Unified participant, instructor, and developer commands."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

from pytorch_course import doctor
from pytorch_course.profiles import kernel_identity, require_profile

COURSE_ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT_DIR = COURSE_ROOT / "environment"
COMPLETION_SCRIPT = files("pytorch_course").joinpath("completions").joinpath("course.bash")


def emit_report(
    report: dict[str, object],
    *,
    as_json: bool,
    output: Path | None,
    human_lines: list[str],
) -> int:
    """Print a report, optionally persisting the same JSON contract."""
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if output is not None:
        output_path = output if output.is_absolute() else COURSE_ROOT / output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded + "\n", encoding="utf-8")
    print(encoded if as_json else "\n".join(human_lines))
    return 0 if report["status"] == "pass" else 1


def run_tensor_diagnostic(args: argparse.Namespace) -> int:
    """Run the preparation tensor and device diagnostic."""
    from lessons.prep.tensor_device.reference import tensor_device_report

    try:
        report = tensor_device_report(args.device)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    return emit_report(
        report,
        as_json=args.json,
        output=args.output,
        human_lines=[
            f"Device: {report['selected_device']} (requested {report['requested_device']})",
            f"Tensor: shape={report['shape']}, dtype={report['dtype']}",
            f"CUDA: available={report['cuda_available']}, devices={report['cuda_device_count']}",
            f"Status: {str(report['status']).upper()}",
        ],
    )


def run_python_numpy(args: argparse.Namespace) -> int:
    """Run the preparation NumPy broadcasting diagnostic."""
    from lessons.prep.python_numpy.solution import python_numpy_report

    report = python_numpy_report()
    return emit_report(
        report,
        as_json=args.json,
        output=args.output,
        human_lines=[
            f"Table: {report['samples']} samples × {report['features']} features",
            f"Input preserved: {report['input_preserved']}",
            f"Status: {str(report['status']).upper()}",
        ],
    )


def run_autograd(args: argparse.Namespace) -> int:
    """Run the preparation analytical-gradient diagnostic."""
    from lessons.prep.tensors_autograd.reference import autograd_report
    from lessons.prep.tensors_autograd.solution import squared_error

    try:
        report = autograd_report(squared_error, args.device)
    except (RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return emit_report(
        report,
        as_json=args.json,
        output=args.output,
        human_lines=[
            f"Device: {report['selected_device']} (requested {report['requested_device']})",
            f"Gradient: {report['gradient']}",
            f"Status: {str(report['status']).upper()}",
        ],
    )


def run_data_loader(args: argparse.Namespace) -> int:
    """Run the preparation Dataset and DataLoader diagnostic."""
    from lessons.prep.datasets_loaders.reference import dataset_loader_report
    from lessons.prep.datasets_loaders.solution import PairDataset

    report = dataset_loader_report(PairDataset)
    return emit_report(
        report,
        as_json=args.json,
        output=args.output,
        human_lines=[
            f"Epoch: {report['samples']} samples in {report['batches']} batches",
            f"Deterministic shuffle: {report['deterministic_shuffle']}",
            f"Status: {str(report['status']).upper()}",
        ],
    )


def run_mlp_training(args: argparse.Namespace) -> int:
    """Run the deterministic Day 1 training path."""
    from lessons.day1.mlp.solution import optimization_step
    from lessons.day1.mlp.training import train_mlp

    try:
        report = train_mlp(
            training_step=optimization_step,
            requested_device=args.device,
            seed=args.seed,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            samples=args.samples,
        )
    except (RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return emit_report(
        report,
        as_json=args.json,
        output=args.output,
        human_lines=[
            f"Device: {report['selected_device']} (requested {report['requested_device']})",
            (
                f"Training loss: {report['initial_train_loss']:.6f} "
                f"→ {report['final_train_loss']:.6f}"
            ),
            f"Evaluation accuracy: {report['eval_accuracy']:.1%}",
            f"Status: {str(report['status']).upper()}",
        ],
    )


def run_cnn_training(args: argparse.Namespace) -> int:
    """Run the deterministic Day 1 CNN training path."""
    from lessons.day1.cnn.training import train_cnn

    try:
        report = train_cnn(
            requested_device=args.device,
            seed=args.seed,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            samples=args.samples,
            batch_size=args.batch_size,
        )
    except (RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return emit_report(
        report,
        as_json=args.json,
        output=args.output,
        human_lines=[
            f"Device: {report['selected_device']} (requested {report['requested_device']})",
            (
                f"Evaluation loss: {report['initial_eval_loss']:.6f} "
                f"→ {report['final_eval_loss']:.6f}"
            ),
            f"Evaluation accuracy: {report['eval_accuracy']:.1%}",
            f"Status: {str(report['status']).upper()}",
        ],
    )


def run_mlp_exercise(args: argparse.Namespace) -> int:
    """Run the participant MLP exercise self-check."""
    from lessons.day1.mlp.exercise import main

    return main(args.device)


def run_cnn_exercise(args: argparse.Namespace) -> int:
    """Run the participant CNN architecture self-check."""
    from lessons.day1.cnn.exercise import main

    return main(args.device)


def run_debugging_exercise() -> int:
    """Run the participant debugging and evaluation self-check."""
    from lessons.day1.debugging_evaluation.exercise import main

    return main()


def run_evaluation_diagnostic(args: argparse.Namespace) -> int:
    """Run the completed debugging and evaluation diagnostic."""
    from lessons.day1.debugging_evaluation.reference import debugging_report
    from lessons.day1.debugging_evaluation.solution import repair_batch

    report = debugging_report(repair_batch)
    return emit_report(
        report,
        as_json=args.json,
        output=args.output,
        human_lines=[
            f"Issues before repair: {len(report['issues_before'])}",
            f"Issues after repair: {len(report['issues_after'])}",
            f"Status: {str(report['status']).upper()}",
        ],
    )


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
    """Install the active managed profile as a user Jupyter kernel."""
    try:
        profile = require_profile()
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    kernel_name, display_name = kernel_identity(profile)
    return run(
        [
            sys.executable,
            "-m",
            "ipykernel",
            "install",
            "--user",
            "--name",
            kernel_name,
            "--display-name",
            display_name,
        ]
    )


def emit_bash_completion() -> int:
    """Write the packaged static Bash completion script to stdout."""
    sys.stdout.write(COMPLETION_SCRIPT.read_text(encoding="utf-8"))
    return 0


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

    prep = commands.add_parser("prep", help="run pre-course diagnostics")
    prep_commands = prep.add_subparsers(dest="prep_command", required=True)
    tensor_device = prep_commands.add_parser(
        "tensor-device", help="check tensor creation and device transfer"
    )
    tensor_device.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    tensor_device.add_argument("--json", action="store_true")
    tensor_device.add_argument("--output", type=Path)
    python_numpy = prep_commands.add_parser(
        "python-numpy", help="check NumPy reductions and broadcasting"
    )
    python_numpy.add_argument("--json", action="store_true")
    python_numpy.add_argument("--output", type=Path)
    tensors_autograd = prep_commands.add_parser(
        "tensors-autograd", help="check tensor gradients analytically"
    )
    tensors_autograd.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    tensors_autograd.add_argument("--json", action="store_true")
    tensors_autograd.add_argument("--output", type=Path)
    data_loader = prep_commands.add_parser(
        "data-loader", help="check dataset indexing and deterministic batching"
    )
    data_loader.add_argument("--json", action="store_true")
    data_loader.add_argument("--output", type=Path)

    exercise = commands.add_parser("exercise", help="run participant exercise self-checks")
    exercise_commands = exercise.add_subparsers(dest="exercise_command", required=True)
    exercise_mlp = exercise_commands.add_parser(
        "mlp", help="check the Day 1 optimization-step exercise"
    )
    exercise_mlp.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    exercise_cnn = exercise_commands.add_parser(
        "cnn", help="check the Day 1 CNN architecture exercise"
    )
    exercise_cnn.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    exercise_commands.add_parser(
        "debugging-evaluation", help="check the Day 1 batch-repair exercise"
    )

    train = commands.add_parser("train", help="run tested training paths")
    train_commands = train.add_subparsers(dest="train_command", required=True)
    mlp = train_commands.add_parser("mlp", help="train the deterministic Day 1 MLP")
    mlp.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    mlp.add_argument("--seed", type=int, default=7)
    mlp.add_argument("--epochs", type=int, default=200)
    mlp.add_argument("--learning-rate", type=float, default=0.05)
    mlp.add_argument("--samples", type=int, default=512)
    mlp.add_argument("--json", action="store_true")
    mlp.add_argument("--output", type=Path)
    cnn = train_commands.add_parser("cnn", help="train the deterministic Day 1 CNN")
    cnn.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    cnn.add_argument("--seed", type=int, default=11)
    cnn.add_argument("--epochs", type=int, default=6)
    cnn.add_argument("--learning-rate", type=float, default=0.03)
    cnn.add_argument("--samples", type=int, default=256)
    cnn.add_argument("--batch-size", type=int, default=32)
    cnn.add_argument("--json", action="store_true")
    cnn.add_argument("--output", type=Path)

    diagnose = commands.add_parser("diagnose", help="run focused learning diagnostics")
    diagnose_commands = diagnose.add_subparsers(dest="diagnose_command", required=True)
    evaluation = diagnose_commands.add_parser(
        "evaluation", help="check classification shape, dtype, and state invariants"
    )
    evaluation.add_argument("--json", action="store_true")
    evaluation.add_argument("--output", type=Path)

    kernel = commands.add_parser("kernel", help="manage the participant Jupyter kernel")
    kernel.add_argument("action", choices=("install",))

    completion = commands.add_parser("completion", help="emit shell completion code")
    completion.add_argument("shell", choices=("bash",))

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
    if args.command == "prep" and args.prep_command == "tensor-device":
        return run_tensor_diagnostic(args)
    if args.command == "prep" and args.prep_command == "python-numpy":
        return run_python_numpy(args)
    if args.command == "prep" and args.prep_command == "tensors-autograd":
        return run_autograd(args)
    if args.command == "prep" and args.prep_command == "data-loader":
        return run_data_loader(args)
    if args.command == "exercise" and args.exercise_command == "mlp":
        return run_mlp_exercise(args)
    if args.command == "exercise" and args.exercise_command == "cnn":
        return run_cnn_exercise(args)
    if args.command == "exercise" and args.exercise_command == "debugging-evaluation":
        return run_debugging_exercise()
    if args.command == "train" and args.train_command == "mlp":
        return run_mlp_training(args)
    if args.command == "train" and args.train_command == "cnn":
        return run_cnn_training(args)
    if args.command == "diagnose" and args.diagnose_command == "evaluation":
        return run_evaluation_diagnostic(args)
    if args.command == "kernel":
        return install_kernel()
    if args.command == "completion":
        return emit_bash_completion()
    if args.command == "slides":
        return run_slides(args.action)
    if args.command == "alps-smoke":
        return run_alps_smoke(args)
    parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
