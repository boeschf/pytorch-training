from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from lessons.day1.mlp.reference import MLP, make_dataset, train_mlp
from lessons.day1.mlp.solution import optimization_step


def test_cpu_training_has_deterministic_learning_signal() -> None:
    first = train_mlp(requested_device="cpu")
    second = train_mlp(requested_device="cpu")

    assert first == second
    assert first["status"] == "pass"
    assert first["final_train_loss"] < first["initial_train_loss"] * 0.35
    assert first["eval_accuracy"] >= 0.95


def test_solution_step_updates_model_parameters() -> None:
    torch.manual_seed(7)
    split = make_dataset(samples=64, seed=7)
    model = MLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    loss = optimization_step(
        model,
        optimizer,
        torch.nn.CrossEntropyLoss(),
        split.train_features,
        split.train_targets,
    )

    assert loss > 0
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, model.parameters(), strict=True)
    )
