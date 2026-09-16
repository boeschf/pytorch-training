#!/usr/bin/env bash
set -euo pipefail

course_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$course_root/src${PYTHONPATH:+:$PYTHONPATH}"

exec python "$course_root/environment/doctor.py" \
  --require-torch \
  --require-cuda \
  --minimum-gpus "${COURSE_MINIMUM_GPUS:-1}" \
  --require-image-torch
