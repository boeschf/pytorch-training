from __future__ import annotations

import pytest

pytest.importorskip("torch")

from pytorch_course.foundations.mlp import train_mlp


def test_cpu_training_has_deterministic_learning_signal() -> None:
    first = train_mlp(requested_device="cpu")
    second = train_mlp(requested_device="cpu")

    assert first == second
    assert first["status"] == "pass"
    assert first["final_train_loss"] < first["initial_train_loss"] * 0.35
    assert first["eval_accuracy"] >= 0.95
