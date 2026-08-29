from __future__ import annotations

from typing import Protocol


class PlayerDetector(Protocol):
    name: str

    def detect(self, frame, context: dict) -> list[dict]:
        """Return player detections for one frame."""


class PoseEstimator(Protocol):
    model_name: str

    def infer(self, frame) -> tuple[object, object]:
        """Return COCO-17 coordinates and per-keypoint confidence arrays."""


class PlayerTracker(Protocol):
    name: str

    def update(self, detections: list[dict], frame_index: int) -> list[dict]:
        """Return associated player tracks for one frame."""
