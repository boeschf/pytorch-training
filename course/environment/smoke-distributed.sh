#!/usr/bin/env bash
set -euo pipefail

course_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
venv_python="$course_root/.venv-alps-gh200/bin/python"
[[ -x "$venv_python" ]] || {
  printf 'error: create the alps-gh200 profile before running Alps smoke checks\n' >&2
  exit 1
}
processes=${COURSE_GPUS_PER_NODE:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

exec "$venv_python" -m torch.distributed.run \
  --standalone \
  --nproc-per-node="$processes" \
  "$course_root/environment/smoke_distributed.py"
