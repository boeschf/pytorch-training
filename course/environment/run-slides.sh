#!/usr/bin/env bash
set -euo pipefail

course_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
node_root="$course_root/.nodeenv"
export PATH="$node_root/bin:$PATH"

exec "$node_root/bin/node" \
  "$node_root/lib/node_modules/npm/bin/npm-cli.js" \
  --prefix "$course_root/slides" "$@"
