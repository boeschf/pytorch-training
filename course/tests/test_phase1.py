from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from lessons.day1.mlp.solution import optimization_step
from lessons.day1.mlp.training import MLP, evaluate, make_dataset, run_mlp, train_mlp


def test_cpu_training_has_deterministic_learning_signal() -> None:
    first = train_mlp(requested_device="cpu")
    second = train_mlp(requested_device="cpu")

    assert first == second
    assert first["status"] == "pass"
    assert first["final_train_loss"] < first["initial_train_loss"] * 0.35
    assert first["eval_accuracy"] >= 0.95


def test_training_history_includes_boundaries_and_progress() -> None:
    run = run_mlp(
        requested_device="cpu",
        epochs=25,
        samples=64,
        snapshot_interval=10,
    )

    assert [snapshot.epoch for snapshot in run.history] == [0, 10, 20, 25]
    assert run.history[-1].train_loss < run.history[0].train_loss
    assert not run.model.training


def test_evaluate_preserves_the_selected_model_mode() -> None:
    split = make_dataset(samples=64, seed=7)
    model = MLP()
    loss_function = torch.nn.CrossEntropyLoss()

    model.train()
    evaluate(model, split.eval_features, split.eval_targets, loss_function)
    assert model.training

    model.eval()
    evaluate(model, split.eval_features, split.eval_targets, loss_function)
    assert not model.training


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
