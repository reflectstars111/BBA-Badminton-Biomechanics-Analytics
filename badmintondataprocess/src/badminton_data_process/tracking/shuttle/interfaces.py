from __future__ import annotations

from typing import Protocol


class ShuttleDetector(Protocol):
    name: str

    def detect_sequence(self, frames: list, context: dict) -> list[dict]:
        """Return shuttle detections for a temporal frame window."""

