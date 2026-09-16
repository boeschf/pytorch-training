#!/usr/bin/env bash
set -euo pipefail

course_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
processes=${COURSE_GPUS_PER_NODE:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

exec python -m torch.distributed.run \
  --standalone \
  --nproc-per-node="$processes" \
  "$course_root/environment/smoke_distributed.py"
