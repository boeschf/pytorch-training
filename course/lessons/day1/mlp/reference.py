"""Deterministic two-class MLP with an explicit training loop."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from lessons.prep.tensor_device.reference import DeviceRequest, resolve_device


@dataclass(frozen=True)
class DatasetSplit:
    """Deterministic train and evaluation tensors."""

    train_features: torch.Tensor
    train_targets: torch.Tensor
    eval_features: torch.Tensor
    eval_targets: torch.Tensor


# region mlp-model
class MLP(nn.Module):
    """Small classifier for a nonlinear two-dimensional problem."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return unnormalized class scores for each sample."""
        return self.layers(features)


# endregion mlp-model


@dataclass(frozen=True)
class TrainingSnapshot:
    """Metrics captured at one epoch for diagnostics and visualization."""

    epoch: int
    train_loss: float
    train_accuracy: float
    eval_loss: float
    eval_accuracy: float


@dataclass(frozen=True)
class TrainingRun:
    """Trained state plus serializable metrics and learning history."""

    model: MLP
    split: DatasetSplit
    device: torch.device
    history: tuple[TrainingSnapshot, ...]
    report: dict[str, object]


# region xor-dataset
def make_dataset(*, samples: int, seed: int) -> DatasetSplit:
    """Create a balanced XOR-like dataset without network access."""
    if samples < 64 or samples % 4:
        raise ValueError("samples must be a multiple of four and at least 64")

    # region xor-generation
    generator = torch.Generator().manual_seed(seed)
    centers = torch.tensor(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ]
    )
    center_ids = torch.arange(samples) % 4
    features = centers[center_ids] + 0.30 * torch.randn(samples, 2, generator=generator)
    targets = torch.tensor([0, 1, 1, 0], dtype=torch.long)[center_ids]
    order = torch.randperm(samples, generator=generator)
    features = features[order]
    targets = targets[order]
    train_size = 3 * samples // 4
    # endregion xor-generation
    return DatasetSplit(
        train_features=features[:train_size],
        train_targets=targets[:train_size],
        eval_features=features[train_size:],
        eval_targets=targets[train_size:],
    )


# endregion xor-dataset


# region evaluation
def evaluate(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    loss_function: nn.Module,
) -> tuple[float, float]:
    """Return loss and accuracy without recording gradients."""
    model.eval()
    with torch.no_grad():
        logits = model(features)
        loss = loss_function(logits, targets).item()
        accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
    return loss, accuracy


# endregion evaluation


def run_mlp(
    *,
    requested_device: DeviceRequest = "auto",
    seed: int = 7,
    epochs: int = 200,
    learning_rate: float = 0.05,
    samples: int = 512,
    snapshot_interval: int = 10,
) -> TrainingRun:
    """Train the MLP and retain bounded history for diagnostics and plots."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if snapshot_interval < 1:
        raise ValueError("snapshot_interval must be positive")

    device = resolve_device(requested_device)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    split = make_dataset(samples=samples, seed=seed)
    train_features = split.train_features.to(device)
    train_targets = split.train_targets.to(device)
    eval_features = split.eval_features.to(device)
    eval_targets = split.eval_targets.to(device)

    # region training-components
    model = MLP().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # endregion training-components

    train_loss, train_accuracy = evaluate(model, train_features, train_targets, loss_function)
    eval_loss, eval_accuracy = evaluate(model, eval_features, eval_targets, loss_function)
    history = [
        TrainingSnapshot(0, train_loss, train_accuracy, eval_loss, eval_accuracy),
    ]
    model.train()

    # region explicit-training-loop
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_features)
        loss = loss_function(logits, train_targets)
        loss.backward()
        optimizer.step()
        # endregion explicit-training-loop

        if epoch % snapshot_interval == 0 or epoch == epochs:
            train_loss, train_accuracy = evaluate(
                model, train_features, train_targets, loss_function
            )
            eval_loss, eval_accuracy = evaluate(model, eval_features, eval_targets, loss_function)
            history.append(
                TrainingSnapshot(
                    epoch,
                    train_loss,
                    train_accuracy,
                    eval_loss,
                    eval_accuracy,
                )
            )
            if epoch != epochs:
                model.train()

    initial = history[0]
    final = history[-1]
    learned = final.train_loss < initial.train_loss * 0.35 and final.eval_accuracy >= 0.95
    report: dict[str, object] = {
        "schema_version": 1,
        "seed": seed,
        "requested_device": requested_device,
        "selected_device": device.type,
        "samples": samples,
        "train_samples": len(train_targets),
        "eval_samples": len(eval_targets),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "initial_train_loss": round(initial.train_loss, 6),
        "final_train_loss": round(final.train_loss, 6),
        "train_accuracy": round(final.train_accuracy, 6),
        "eval_loss": round(final.eval_loss, 6),
        "eval_accuracy": round(final.eval_accuracy, 6),
        "torch_version": torch.__version__,
        "status": "pass" if learned else "fail",
    }
    return TrainingRun(model, split, device, tuple(history), report)


def train_mlp(
    *,
    requested_device: DeviceRequest = "auto",
    seed: int = 7,
    epochs: int = 200,
    learning_rate: float = 0.05,
    samples: int = 512,
) -> dict[str, object]:
    """Train the reference MLP and return stable, machine-readable metrics."""
    return run_mlp(
        requested_device=requested_device,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        samples=samples,
    ).report
