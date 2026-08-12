from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / 'src'
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from badminton_data_process.core.io import (  # noqa: E402
    ensure_dir,
    ensure_parent,
    read_csv_rows,
    write_csv_rows,
)

__all__ = ['ensure_parent', 'ensure_dir', 'read_csv_rows', 'write_csv_rows']
