#!/usr/bin/env bash
set -euo pipefail

course_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
node_root="$course_root/.nodeenv"

if [[ -x "$node_root/bin/node" ]]; then
  actual_node_version=$("$node_root/bin/node" --version)
  if [[ "$actual_node_version" != "v24.21.0" ]]; then
    printf 'Expected Node v24.21.0, found %s in %s\n' \
      "$actual_node_version" "$node_root" >&2
    exit 1
  fi
else
  uv run --project "$course_root" nodeenv --node=24.21.0 "$node_root"
fi
export PATH="$node_root/bin:$PATH"
"$node_root/bin/node" \
  "$node_root/lib/node_modules/npm/bin/npm-cli.js" \
  --prefix "$course_root/slides" ci

printf 'Slide environment ready: %s\n' "$node_root"
printf 'Run: %s run build\n' "$course_root/environment/run-slides.sh"
