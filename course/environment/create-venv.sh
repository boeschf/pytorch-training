#!/usr/bin/env bash
set -euo pipefail

course_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

[[ $# -eq 1 ]] || fail "usage: $0 {cpu|cuda|alps-gh200}"
profile=$1

case "$profile" in
  cpu | cuda | alps-gh200) ;;
  *) fail "usage: $0 {cpu|cuda|alps-gh200}" ;;
esac

venv_root="$course_root/.venv-$profile"
venv_python="$venv_root/bin/python"
marker="$venv_root/.course-profile"
system_packages=false
[[ "$profile" == alps-gh200 ]] && system_packages=true

if [[ -e "$venv_root" ]]; then
  [[ -x "$venv_python" ]] || fail "$venv_root is not a usable virtual environment"
  [[ -f "$marker" ]] || fail "$venv_root has no course profile marker; remove it and retry"
  recorded_profile=$(<"$marker")
  [[ "$recorded_profile" == "$profile" ]] || fail "$venv_root belongs to another profile"
  [[ "$("$venv_python" -c 'import platform; print(platform.python_version())')" == 3.12.* ]] ||
    fail "$venv_root must use Python 3.12"
  config=$(<"$venv_root/pyvenv.cfg")
  [[ "$config" == *"include-system-site-packages = $system_packages"* ]] ||
    fail "$venv_root has the wrong system-site-packages setting; remove it and retry"
elif [[ "$profile" == alps-gh200 ]]; then
  base_python=$(command -v python3.12 || command -v python3 || command -v python)
  "$base_python" -c 'import torch' 2>/dev/null ||
    fail "alps-gh200 must be created inside the canonical course image"
  "$base_python" -m venv --system-site-packages "$venv_root"
  printf '%s\n' "$profile" >"$marker"
else
  command -v uv >/dev/null || fail "uv is required to create the $profile profile"
  uv venv --python 3.12 "$venv_root"
  printf '%s\n' "$profile" >"$marker"
fi

if [[ "$profile" == alps-gh200 ]]; then
  site_packages=$("$venv_python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
  # Keep the pinned image's Python stack intact; only expose the course source tree.
  printf '%s\n%s\n' "$course_root" "$course_root/src" \
    >"$site_packages/cscs_pytorch_course.pth"
  cat >"$venv_root/bin/course" <<EOF
#!$venv_python
from pytorch_course.cli import main
raise SystemExit(main())
EOF
  chmod +x "$venv_root/bin/course"
else
  UV_PROJECT_ENVIRONMENT="$venv_root" \
    uv sync --project "$course_root" --all-groups --extra "$profile" --link-mode copy --frozen
fi

"$venv_root/bin/course" doctor --require-profile "$profile"
printf 'Course environment ready: %s\n' "$venv_root"
printf 'Activate: source %q\n' "$venv_root/bin/activate"
