from __future__ import annotations

import json
import gc
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
except ImportError:  # pragma: no cover - runtime dependency
    np = None

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional runtime dependency
    YOLO = None


COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

COCO_SKELETON_EDGES = (
    ("left_ear", "left_eye"),
    ("left_eye", "nose"),
    ("nose", "right_eye"),
    ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)


@dataclass(frozen=True, slots=True)
class PoseKeypoint:
    name: str
    x: float
    y: float
    confidence: float


@dataclass(frozen=True, slots=True)
class PoseObservation:
    """One COCO-17 pose associated with one detector bbox."""

    keypoints: tuple[PoseKeypoint, ...]
    model_name: str
    detection_confidence: float

    def valid_keypoints(self, threshold: float) -> tuple[PoseKeypoint, ...]:
        return tuple(point for point in self.keypoints if point.confidence >= threshold)

    def anchor_keypoints(self) -> dict[str, tuple[float, float, float]]:
        return {
            point.name: (point.x, point.y, point.confidence)
            for point in self.keypoints
        }

    def mean_confidence(self, threshold: float) -> float:
        valid = self.valid_keypoints(threshold)
        return sum(point.confidence for point in valid) / len(valid) if valid else 0.0

    def to_json(self) -> str:
        return json.dumps(
            [
                {
                    "name": point.name,
                    "x": round(point.x, 3),
                    "y": round(point.y, 3),
                    "confidence": round(point.confidence, 4),
                }
                for point in self.keypoints
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class PoseRuntimeConfig:
    model_name: str = "yolo11n-pose.pt"
    keypoint_threshold: float = 0.35
    min_valid_keypoints: int = 5
    rtmpose_mode: str = "balanced"
    rtmpose_backend: str = "onnxruntime"
    rtmpose_device: str = "auto"
    rtmpose_detector_model: str = ""
    rtmpose_pose_model: str = ""
    rtmpose_detector_input_size: tuple[int, int] = (416, 416)
    rtmpose_pose_input_size: tuple[int, int] = (192, 256)


def pose_keypoints_from_json(value: str | None) -> tuple[PoseKeypoint, ...]:
    if not value:
        return ()
    try:
        payload = json.loads(value)
        return tuple(
            PoseKeypoint(
                name=str(item["name"]),
                x=float(item["x"]),
                y=float(item["y"]),
                confidence=float(item["confidence"]),
            )
            for item in payload
        )
    except (TypeError, ValueError, KeyError, json.JSONDecodeError):
        return ()


def build_pose_observation(
    keypoints_xy: Iterable[Iterable[float]],
    keypoint_confidences: Iterable[float] | None,
    *,
    model_name: str,
    detection_confidence: float,
) -> PoseObservation:
    coordinates = list(keypoints_xy)
    if len(coordinates) != len(COCO_KEYPOINT_NAMES):
        raise ValueError(
            f"COCO pose requires {len(COCO_KEYPOINT_NAMES)} keypoints, got {len(coordinates)}"
        )
    confidences = (
        list(keypoint_confidences)
        if keypoint_confidences is not None
        else [1.0] * len(COCO_KEYPOINT_NAMES)
    )
    if len(confidences) != len(COCO_KEYPOINT_NAMES):
        raise ValueError("keypoint confidence count does not match coordinates")
    return PoseObservation(
        keypoints=tuple(
            PoseKeypoint(
                name=name,
                x=float(point[0]),
                y=float(point[1]),
                confidence=min(max(float(confidence), 0.0), 1.0),
            )
            for name, point, confidence in zip(
                COCO_KEYPOINT_NAMES,
                coordinates,
                confidences,
                strict=True,
            )
        ),
        model_name=model_name,
        detection_confidence=min(max(float(detection_confidence), 0.0), 1.0),
    )


def skeleton_segments(
    keypoints: Iterable[PoseKeypoint],
    threshold: float,
) -> list[tuple[PoseKeypoint, PoseKeypoint]]:
    by_name = {
        point.name: point
        for point in keypoints
        if point.confidence >= threshold
    }
    return [
        (by_name[start], by_name[end])
        for start, end in COCO_SKELETON_EDGES
        if start in by_name and end in by_name
    ]


@lru_cache(maxsize=4)
def load_yolo_pose_model(model_name: str) -> Any:
    if YOLO is None:
        raise RuntimeError("ultralytics is required for the YOLO Pose Implementation")
    return YOLO(model_name)


def detect_yolo_pose_candidates(
    frame: Any,
    court_mask: Any,
    *,
    model_name: str,
    confidence_threshold: float,
    image_size: int,
) -> list[dict[str, object]]:
    """Return role-association candidates carrying typed pose Observations."""

    if np is None:
        raise RuntimeError("NumPy is required for the YOLO Pose Implementation")
    model = load_yolo_pose_model(model_name)
    results = model.predict(
        source=frame,
        conf=confidence_threshold,
        classes=[0],
        verbose=False,
        imgsz=image_size,
        max_det=10,
    )
    if not results:
        return []
    result = results[0]
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    if boxes is None or keypoints is None or keypoints.xy is None:
        return []
    xy_values = keypoints.xy.cpu().numpy()
    score_values = keypoints.conf.cpu().numpy() if keypoints.conf is not None else None
    box_values = boxes.xyxy.cpu().numpy()
    box_confidences = boxes.conf.cpu().numpy()
    if len(box_values) != len(xy_values):
        raise RuntimeError("YOLO Pose returned mismatched bbox and keypoint counts")

    candidates: list[dict[str, object]] = []
    for index, (xyxy, confidence) in enumerate(
        zip(box_values, box_confidences, strict=True)
    ):
        x1, y1, x2, y2 = [int(round(value)) for value in xyxy.tolist()]
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1] - 1, x2)
        y2 = min(frame.shape[0] - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        bottom_center = ((x1 + x2) / 2.0, float(y2))
        bx = min(frame.shape[1] - 1, max(0, int(round(bottom_center[0]))))
        by = min(frame.shape[0] - 1, max(0, int(round(bottom_center[1]))))
        if court_mask[by, bx] == 0:
            continue
        pose = build_pose_observation(
            xy_values[index],
            score_values[index] if score_values is not None else None,
            model_name=model_name,
            detection_confidence=float(confidence),
        )
        candidates.append(
            {
                "bbox": (x1, y1, x2, y2),
                "score": float(confidence),
                "bottom_center": bottom_center,
                "area": float((x2 - x1) * (y2 - y1)),
                "pose": pose,
            }
        )
    return candidates


def _auto_rtmpose_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@dataclass(slots=True)
class RtmposeImplementation:
    """Lazy rtmlib-backed RTMPose Implementation with explicit runtime config."""

    mode: str = "balanced"
    backend: str = "onnxruntime"
    device: str = "auto"
    detector_model: str = ""
    pose_model: str = ""
    detector_input_size: tuple[int, int] = (416, 416)
    pose_input_size: tuple[int, int] = (192, 256)
    _model: Any = None

    @property
    def model_name(self) -> str:
        return self.pose_model or f"rtmlib:rtmpose:{self.mode}"

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from rtmlib import Body
        except ImportError as exc:
            raise RuntimeError(
                "RTMPose requires optional dependencies: pip install -e '.[pose]'"
            ) from exc
        selected_device = _auto_rtmpose_device() if self.device == "auto" else self.device
        if self.backend == "onnxruntime" and selected_device == "cuda":
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError(
                    "RTMPose CUDA requires onnxruntime-gpu; the onnxruntime module is missing"
                ) from exc
            providers = set(ort.get_available_providers())
            if "CUDAExecutionProvider" not in providers:
                raise RuntimeError(
                    "RTMPose CUDA was requested, but ONNX Runtime does not expose "
                    "CUDAExecutionProvider. Install onnxruntime-gpu and remove the CPU-only "
                    "onnxruntime package."
                )
        if bool(self.detector_model) != bool(self.pose_model):
            raise RuntimeError(
                "rtmpose_detector_model and rtmpose_pose_model must be configured together"
            )
        if self.detector_model:
            detector_path = Path(self.detector_model)
            pose_path = Path(self.pose_model)
            if not detector_path.is_file() or not pose_path.is_file():
                raise RuntimeError(
                    "configured RTMPose ONNX files are missing: "
                    f"detector={detector_path}, pose={pose_path}"
                )
            self._model = Body(
                det=str(detector_path),
                det_input_size=self.detector_input_size,
                pose=str(pose_path),
                pose_input_size=self.pose_input_size,
                backend=self.backend,
                device=selected_device,
            )
        else:
            self._model = Body(
                mode=self.mode,
                backend=self.backend,
                device=selected_device,
            )
        return self._model

    def infer(self, frame: Any) -> tuple[Any, Any]:
        keypoints, scores = self._load()(frame)
        return keypoints, scores


@lru_cache(maxsize=4)
def load_rtmpose_implementation(
    mode: str,
    backend: str,
    device: str,
    detector_model: str,
    pose_model: str,
    detector_input_size: tuple[int, int],
    pose_input_size: tuple[int, int],
) -> RtmposeImplementation:
    return RtmposeImplementation(
        mode=mode,
        backend=backend,
        device=device,
        detector_model=detector_model,
        pose_model=pose_model,
        detector_input_size=detector_input_size,
        pose_input_size=pose_input_size,
    )


def _bbox_from_pose(
    pose: PoseObservation,
    frame_shape: tuple[int, ...],
    keypoint_threshold: float,
) -> tuple[int, int, int, int] | None:
    valid = pose.valid_keypoints(keypoint_threshold)
    if len(valid) < 3:
        return None
    frame_height, frame_width = frame_shape[:2]
    xs = [point.x for point in valid]
    ys = [point.y for point in valid]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    pad_x = max(4.0, width * 0.15)
    pad_y = max(4.0, height * 0.08)
    x1 = max(0, int(round(min(xs) - pad_x)))
    y1 = max(0, int(round(min(ys) - pad_y)))
    x2 = min(frame_width - 1, int(round(max(xs) + pad_x)))
    y2 = min(frame_height - 1, int(round(max(ys) + pad_y)))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def rtmpose_outputs_to_candidates(
    frame: Any,
    court_mask: Any,
    keypoints: Any,
    scores: Any,
    *,
    model_name: str,
    keypoint_threshold: float,
    min_valid_keypoints: int,
) -> list[dict[str, object]]:
    if np is None:
        raise RuntimeError("NumPy is required for RTMPose")
    coordinates = np.asarray(keypoints) if keypoints is not None else np.empty((0, 17, 2))
    confidences = np.asarray(scores) if scores is not None else None
    if coordinates.ndim == 2:
        coordinates = coordinates[None, ...]
    if confidences is not None and confidences.ndim == 1:
        confidences = confidences[None, ...]
    candidates: list[dict[str, object]] = []
    for index, person_points in enumerate(coordinates):
        person_scores = confidences[index] if confidences is not None else None
        pose = build_pose_observation(
            person_points,
            person_scores,
            model_name=model_name,
            detection_confidence=1.0,
        )
        valid = pose.valid_keypoints(keypoint_threshold)
        if len(valid) < min_valid_keypoints:
            continue
        bbox = _bbox_from_pose(pose, frame.shape, keypoint_threshold)
        if bbox is None:
            continue
        anchors = pose.anchor_keypoints()
        ankles = [
            anchors[name]
            for name in ("left_ankle", "right_ankle")
            if name in anchors and anchors[name][2] >= keypoint_threshold
        ]
        # rtmlib does not expose the internal detector bbox. Without at least
        # one ankle, a pose-extent bottom is not a defensible ground contact.
        if not ankles:
            continue
        bottom_center = (
            sum(point[0] for point in ankles) / len(ankles),
            sum(point[1] for point in ankles) / len(ankles),
        )
        bx = min(frame.shape[1] - 1, max(0, int(round(bottom_center[0]))))
        by = min(frame.shape[0] - 1, max(0, int(round(bottom_center[1]))))
        if court_mask[by, bx] == 0:
            continue
        confidence = pose.mean_confidence(keypoint_threshold)
        candidates.append(
            {
                "bbox": bbox,
                "score": confidence,
                "bottom_center": bottom_center,
                "area": float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
                "pose": pose,
            }
        )
    return candidates


def detect_rtmpose_candidates(
    frame: Any,
    court_mask: Any,
    *,
    mode: str,
    backend: str,
    device: str,
    detector_model: str,
    pose_model: str,
    detector_input_size: tuple[int, int],
    pose_input_size: tuple[int, int],
    keypoint_threshold: float,
    min_valid_keypoints: int,
) -> list[dict[str, object]]:
    implementation = load_rtmpose_implementation(
        mode,
        backend,
        device,
        detector_model,
        pose_model,
        detector_input_size,
        pose_input_size,
    )
    keypoints, scores = implementation.infer(frame)
    return rtmpose_outputs_to_candidates(
        frame,
        court_mask,
        keypoints,
        scores,
        model_name=implementation.model_name,
        keypoint_threshold=keypoint_threshold,
        min_valid_keypoints=min_valid_keypoints,
    )


def release_pose_model_cache() -> None:
    """Release pose sessions before downstream GPU/CPU-heavy Stages."""

    load_yolo_pose_model.cache_clear()
    load_rtmpose_implementation.cache_clear()
    gc.collect()
