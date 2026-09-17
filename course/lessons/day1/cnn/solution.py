"""Reference solution for the compact CNN exercise."""

from __future__ import annotations

from torch import nn

from lessons.prep.tensor_device.reference import DeviceRequest


# region cnn-model-solution
def build_cnn() -> nn.Module:
    """Return a model mapping `[batch, 1, 8, 8]` images to two logits."""
    return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(4 * 4 * 4, 2),
    )


# endregion cnn-model-solution


def main(requested_device: DeviceRequest = "auto") -> int:
    """Run the completed model through the same bounded training contract."""
    from lessons.day1.cnn.training import run_cnn

    report = run_cnn(build_model=build_cnn, requested_device=requested_device).report
    print(report)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
