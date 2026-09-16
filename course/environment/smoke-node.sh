#!/usr/bin/env bash
set -euo pipefail

course_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
gpus=${COURSE_GPUS_PER_NODE:-4}

COURSE_MINIMUM_GPUS="$gpus" "$course_root/environment/smoke-gpu.sh"
COURSE_GPUS_PER_NODE="$gpus" "$course_root/environment/smoke-distributed.sh"
