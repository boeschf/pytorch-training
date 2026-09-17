"""Pre-course tensor/device diagnostic.

Outcome: create a tensor, select a usable device, and verify a device computation.
Prerequisite: the course environment with image-provided PyTorch.
Expected runtime: under 10 seconds.
Resources: CPU; uses one GPU when available.
"""

from __future__ import annotations

import json

from lessons.prep.tensor_device.reference import tensor_device_report

if __name__ == "__main__":
    report = tensor_device_report("auto")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "pass" else 1)
