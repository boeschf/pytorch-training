from __future__ import annotations

import numpy as np
import pytest

from lessons.prep.python_numpy.solution import standardize_columns

torch = pytest.importorskip("torch")

from lessons.day1.cnn.training import train_cnn
from lessons.day1.debugging_evaluation.reference import classification_issues, debugging_report
from lessons.day1.debugging_evaluation.solution import repair_batch
from lessons.prep.datasets_loaders.reference import dataset_loader_report
from lessons.prep.datasets_loaders.solution import PairDataset
from lessons.prep.tensors_autograd.reference import autograd_report
from lessons.prep.tensors_autograd.solution import squared_error


def test_numpy_standardization_obeys_column_contract() -> None:
    values = np.array([[1.0, 10.0], [2.0, 30.0], [6.0, 20.0]])
    original = values.copy()

    normalized = standardize_columns(values)

    assert np.array_equal(values, original)
    assert np.allclose(normalized.mean(axis=0), 0.0)
    assert np.allclose(normalized.std(axis=0), 1.0)


def test_autograd_matches_the_analytical_gradient() -> None:
    report = autograd_report(squared_error, "cpu")

    assert report["status"] == "pass"
    assert report["gradient"] == report["expected_gradient"]


def test_data_loader_epoch_is_complete_and_repeatable() -> None:
    report = dataset_loader_report(PairDataset)

    assert report["status"] == "pass"
    assert report["deterministic_shuffle"] is True
    assert report["complete_epoch"] is True


def test_cpu_cnn_has_a_deterministic_held_out_learning_signal() -> None:
    first = train_cnn(requested_device="cpu")
    second = train_cnn(requested_device="cpu")

    assert first == second
    assert first["status"] == "pass"
    assert first["eval_accuracy"] >= 0.95
    assert first["final_eval_loss"] < first["initial_eval_loss"] * 0.25


def test_debugging_diagnostic_finds_and_repairs_real_batch_errors() -> None:
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64, 2))
    model.eval()
    images = torch.zeros(8, 8, 8, dtype=torch.int32)
    targets = torch.arange(8, dtype=torch.float32) % 2

    issues = classification_issues(model, images, targets)
    report = debugging_report(repair_batch)

    assert issues == [
        "features must have shape [batch, channels, height, width]",
        "features must use a floating-point dtype",
        "CrossEntropyLoss targets must use torch.int64",
    ]
    assert report["status"] == "pass"
    assert report["issues_after"] == []
