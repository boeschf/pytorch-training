#!/usr/bin/env bash
set -euo pipefail

course_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

uv run --project "$course_root" python -m ipykernel install \
  --user \
  --name cscs-pytorch-course \
  --display-name "CSCS PyTorch Course"
