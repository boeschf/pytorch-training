"""Reference solution for ``exercises/day1/mlp.py``.

The complete explicit loop lives in ``pytorch_course.foundations.mlp`` so the
solution, notebook, slides, CLI, and smoke runs execute one implementation.
"""

from __future__ import annotations

import json

from pytorch_course.foundations.mlp import train_mlp

if __name__ == "__main__":
    report = train_mlp(requested_device="auto")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "pass" else 1)
