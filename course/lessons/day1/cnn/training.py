"""Deterministic offline image classification for the Day 1 CNN lesson."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lessons.prep.tensor_device.reference import DeviceRequest, resolve_device

ModelBuilder = Callable[[], nn.Module]


@dataclass(frozen=True)
class ImageSplit:
    """Deterministic training and evaluation image tensors."""

    train_images: torch.Tensor
    train_targets: torch.Tensor
    eval_images: torch.Tensor
    eval_targets: torch.Tensor


@dataclass(frozen=True)
class CNNSnapshot:
    """Metrics captured after an evaluation pass."""

    epoch: int
    train_loss: float
    eval_loss: float
    eval_accuracy: float


@dataclass(frozen=True)
class CNNRun:
    """Trained model, data, bounded history, and serializable report."""

    model: nn.Module
    split: ImageSplit
    device: torch.device
    history: tuple[CNNSnapshot, ...]
    report: dict[str, object]


# region stripe-images
def make_image_dataset(*, samples: int = 256, seed: int = 11) -> ImageSplit:
    """Create noisy 8×8 images containing a vertical or horizontal stripe."""
    if samples < 64 or samples % 4:
        raise ValueError("samples must be a multiple of four and at least 64")
    generator = torch.Generator().manual_seed(seed)
    targets = torch.arange(samples) % 2
    images = 0.12 * torch.randn(samples, 1, 8, 8, generator=generator)
    images[targets == 0, :, :, 2:4] += 1.0
    images[targets == 1, :, 4:6, :] += 1.0

    # Shuffle before a 3:1 split so each partition contains both classes.
    order = torch.randperm(samples, generator=generator)
    images = images[order]
    targets = targets[order]
    train_size = 3 * samples // 4
    return ImageSplit(
        images[:train_size],
        targets[:train_size],
        images[train_size:],
        targets[train_size:],
    )


# endregion stripe-images


def make_loaders(
    split: ImageSplit,
    *,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    """Create a seeded training loader and stable evaluation loader."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    train_loader = DataLoader(
        TensorDataset(split.train_images, split.train_targets),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    eval_loader = DataLoader(
        TensorDataset(split.eval_images, split.eval_targets),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, eval_loader


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the caller-selected model state without recording gradients."""
    total_loss = 0.0
    correct = 0
    samples = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            total_loss += loss_function(logits, targets).item() * len(targets)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            samples += len(targets)
    return total_loss / samples, correct / samples


def run_cnn(
    *,
    build_model: ModelBuilder,
    requested_device: DeviceRequest = "auto",
    seed: int = 11,
    epochs: int = 6,
    learning_rate: float = 0.03,
    samples: int = 256,
    batch_size: int = 32,
) -> CNNRun:
    """Train a caller-supplied CNN on a bounded synthetic image problem."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    device = resolve_device(requested_device)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    split = make_image_dataset(samples=samples, seed=seed)
    train_loader, eval_loader = make_loaders(split, batch_size=batch_size, seed=seed)
    model = build_model().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[CNNSnapshot] = []

    # region cnn-training-loop
    model.eval()
    initial_eval_loss, initial_accuracy = evaluate(model, eval_loader, loss_function, device)
    history.append(CNNSnapshot(0, float("nan"), initial_eval_loss, initial_accuracy))

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        seen = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_function(logits, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(targets)
            seen += len(targets)

        model.eval()
        eval_loss, eval_accuracy = evaluate(model, eval_loader, loss_function, device)
        history.append(CNNSnapshot(epoch, total_train_loss / seen, eval_loss, eval_accuracy))
    # endregion cnn-training-loop

    final = history[-1]
    learned = final.eval_accuracy >= 0.95 and final.eval_loss < initial_eval_loss * 0.25
    report: dict[str, object] = {
        "schema_version": 1,
        "seed": seed,
        "requested_device": requested_device,
        "selected_device": device.type,
        "samples": samples,
        "train_samples": len(split.train_targets),
        "eval_samples": len(split.eval_targets),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "initial_eval_loss": round(initial_eval_loss, 6),
        "final_train_loss": round(final.train_loss, 6),
        "final_eval_loss": round(final.eval_loss, 6),
        "eval_accuracy": round(final.eval_accuracy, 6),
        "status": "pass" if learned else "fail",
    }
    return CNNRun(model, split, device, tuple(history), report)


def train_cnn(**kwargs: object) -> dict[str, object]:
    """Train the reference CNN and return JSON-safe metrics."""
    from lessons.day1.cnn.solution import build_cnn

    return run_cnn(build_model=build_cnn, **kwargs).report
