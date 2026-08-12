from __future__ import annotations

from typing import Protocol


class TrajectorySmoother(Protocol):
    name: str

    def smooth(self, rows: list[dict], coordinate_columns: list[str]) -> list[dict]:
        """Return smoothed trajectory rows."""

