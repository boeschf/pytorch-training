#!/usr/bin/env bash
set -euo pipefail

course_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

venv_root="$course_root/.venv"
venv_python="$venv_root/bin/python"
if [[ -x "$venv_python" ]]; then
  python_version=$("$venv_python" -c 'import platform; print(platform.python_version())')
  if [[ "$python_version" != 3.12.* ]]; then
    printf 'Expected Python 3.12, found %s in %s\n' \
      "$python_version" "$venv_root" >&2
    exit 1
  fi
  venv_config=$(<"$venv_root/pyvenv.cfg")
  if [[ "$venv_config" != *"include-system-site-packages = true"* ]]; then
    printf '%s must expose the image system packages\n' "$venv_root" >&2
    exit 1
  fi
else
  uv venv --python 3.12 --system-site-packages "$venv_root"
fi
uv sync --project "$course_root" --all-groups --frozen

printf 'Course environment ready: %s\n' "$course_root/.venv"
printf 'Run: uv run --project %q course doctor\n' "$course_root"
