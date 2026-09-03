#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART_DIR="$ROOT/wheel_parts"
OUT_DIR="$ROOT/dist"
OUT_WHL="$OUT_DIR/vllm-0.14.1-cp311-cp311-linux_x86_64.whl"

mkdir -p "$OUT_DIR"
cat "$PART_DIR"/vllm-0.14.1-cp311-cp311-linux_x86_64.whl.part-* > "$OUT_WHL"

echo "Verifying wheel SHA256..."
(cd "$ROOT" && sha256sum -c wheel_parts/vllm-0.14.1-cp311-cp311-linux_x86_64.whl.sha256)

echo "Done: $OUT_WHL"
