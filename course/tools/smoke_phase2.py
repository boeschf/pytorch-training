"""Run the complete bounded preparation and Day 1 reference path."""

from __future__ import annotations

import argparse
import json

from lessons.day1.cnn.training import train_cnn
from lessons.day1.debugging_evaluation.reference import debugging_report
from lessons.day1.debugging_evaluation.solution import repair_batch
from lessons.day1.mlp.solution import optimization_step
from lessons.day1.mlp.training import train_mlp
from lessons.prep.datasets_loaders.reference import dataset_loader_report
from lessons.prep.datasets_loaders.solution import PairDataset
from lessons.prep.python_numpy.solution import python_numpy_report
from lessons.prep.tensor_device.reference import tensor_device_report
from lessons.prep.tensors_autograd.reference import autograd_report
from lessons.prep.tensors_autograd.solution import squared_error


def require_pass(name: str, report: dict[str, object]) -> None:
    """Raise with complete evidence when one bounded stage fails."""
    if report["status"] != "pass":
        raise RuntimeError(f"{name} failed: {report}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()

    reports = {
        "tensor_device": tensor_device_report(args.device),
        "python_numpy": python_numpy_report(),
        "tensors_autograd": autograd_report(squared_error, args.device),
        "datasets_loaders": dataset_loader_report(PairDataset),
        "mlp": train_mlp(training_step=optimization_step, requested_device=args.device),
        "cnn": train_cnn(requested_device=args.device),
        "debugging_evaluation": debugging_report(repair_batch),
    }
    for name, report in reports.items():
        require_pass(name, report)

    print(json.dumps(reports, indent=2, sort_keys=True))
    print(f"Phase 2 {args.device}: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
