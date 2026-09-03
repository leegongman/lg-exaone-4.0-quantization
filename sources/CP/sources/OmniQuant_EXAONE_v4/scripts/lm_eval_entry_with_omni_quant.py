#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from pathlib import Path


def _remove_repo_root_from_sys_path(repo_root: Path) -> None:
    cleaned = []
    cwd = Path.cwd().resolve()
    for entry in sys.path:
        resolved = cwd if entry == "" else Path(entry).resolve()
        if resolved == repo_root:
            continue
        cleaned.append(entry)
    sys.path[:] = cleaned


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent

    # Import the pip-installed lm_eval package first, before exposing the
    # repository root that contains a different local `lm_eval/` directory.
    _remove_repo_root_from_sys_path(repo_root)
    from lm_eval.__main__ import cli_evaluate

    # Expose the OmniQuant repo for local helpers, but rely on the packaged
    # vLLM omni_activation_real implementation instead of the older local
    # correctness scaffold.
    sys.path.insert(0, str(repo_root))

    return int(cli_evaluate() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
