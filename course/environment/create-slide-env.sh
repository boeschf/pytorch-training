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
npm_cli="$node_root/lib/node_modules/npm/bin/npm-cli.js"
actual_npm_version=$("$node_root/bin/node" "$npm_cli" --version)
if [[ "$actual_npm_version" != "12.0.2" ]]; then
  "$node_root/bin/node" "$npm_cli" install \
    --global npm@12.0.2 \
    --no-audit \
    --no-fund
fi
"$node_root/bin/node" "$npm_cli" \
  --prefix "$course_root/slides" ci \
  --no-fund

printf 'Slide environment ready: %s\n' "$node_root"
printf 'Run: uv run --project %q course slides build\n' "$course_root"
