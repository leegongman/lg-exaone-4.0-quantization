#!/usr/bin/env bash
set -euo pipefail
find checkpoints -type d | while read -r d; do
  parts=$(find "$d" -maxdepth 1 -type f -name 'model.safetensors.part-*' | sort)
  if [[ -n "$parts" ]]; then
    cat $parts > "$d/model.safetensors"
    if [[ -f "$d/model.safetensors.sha256" ]]; then
      (
        cd "$d"
        sha256sum -c model.safetensors.sha256
      )
    fi
  fi
done
