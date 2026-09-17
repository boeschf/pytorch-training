"""Verify the deterministic Phase 1 learning signal on one device."""

from __future__ import annotations

import argparse
import json

from lessons.day1.mlp.training import train_mlp
from lessons.prep.tensor_device.reference import tensor_device_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()

    tensor_report = tensor_device_report(args.device)
    first = train_mlp(requested_device=args.device)
    second = train_mlp(requested_device=args.device)

    if tensor_report["status"] != "pass":
        raise RuntimeError(f"tensor diagnostic failed: {tensor_report}")
    if first["status"] != "pass":
        raise RuntimeError(f"learning contract failed: {first}")
    if first != second:
        raise RuntimeError(f"training is not deterministic:\n{first}\n{second}")

    print(json.dumps(first, indent=2, sort_keys=True))
    print(f"Phase 1 {args.device}: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
