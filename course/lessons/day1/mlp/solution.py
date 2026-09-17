"""Reference solution for the adjacent ``exercise.py``.

The complete explicit loop lives in ``reference.py`` so the solution, notebook,
slides, CLI, and smoke runs execute one implementation.
"""

from __future__ import annotations

import json

from lessons.day1.mlp.reference import train_mlp

if __name__ == "__main__":
    report = train_mlp(requested_device="auto")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "pass" else 1)
