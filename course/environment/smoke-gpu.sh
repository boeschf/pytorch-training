#!/usr/bin/env bash
set -euo pipefail

course_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
venv_python="$course_root/.venv-alps-gh200/bin/python"
[[ -x "$venv_python" ]] || {
  printf 'error: create the alps-gh200 profile before running Alps smoke checks\n' >&2
  exit 1
}

exec "$venv_python" "$course_root/environment/doctor.py" \
  --require-profile alps-gh200 \
  --require-cuda \
  --minimum-gpus "${COURSE_MINIMUM_GPUS:-1}"
