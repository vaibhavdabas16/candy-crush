from __future__ import annotations

from datetime import datetime
from pathlib import Path


def make_run_dir(base_dir: str | Path, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_dir) / f"{run_name}-{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = Path(base_dir) / f"{run_name}-{timestamp}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
