#!/usr/bin/env bash
set -euo pipefail

course_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
image_store="$course_root/.edf_imagestore"
mkdir -p "$image_store"
export EDF_IMAGESTORE="$image_store"

exec srun \
  --mpi=pmix \
  --network=disable_rdzv_get \
  --environment="$course_root/environment/alps.toml" \
  --distribution=block:cyclic \
  --cpu-bind=sockets \
  --cpus-per-task "${SLURM_CPUS_PER_TASK:-72}" \
  "$@"
