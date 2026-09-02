from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


MATCH_FIELDS = [
    "match_id",
    "source",
    "url",
    "tournament",
    "year",
    "discipline",
    "round",
    "player_1",
    "player_2",
    "resolution",
    "fps",
    "camera_type",
    "usable_rallies",
    "notes",
]

REQUIRED_MATCH_FIELDS = {
    "match_id",
    "source",
    "url",
    "tournament",
    "year",
    "discipline",
    "round",
    "player_1",
    "player_2",
    "resolution",
    "fps",
    "camera_type",
}

RALLY_FIELDS = [
    "match_id",
    "rally_id",
    "start_frame",
    "end_frame",
    "start_time",
    "end_time",
    "duration_seconds",
    "frame_interval",
    "source_main_view_segment_id",
    "eligibility",
    "eligibility_reason",
    "evidence_sample_count",
    "output_path",
    "notes",
]

RALLY_DECISION_FIELDS = [
    "candidate_id",
    "source_main_view_segment_id",
    "start_frame",
    "end_frame",
    "start_time",
    "end_time",
    "duration_seconds",
    "frame_interval",
    "status",
    "reason",
    "evidence_sample_count",
    "rally_id",
    "output_path",
]

COURT_CALIBRATION_SUMMARY_FIELDS = [
    "video_path",
    "video_stem",
    "status",
    "frame_index",
    "json_path",
    "preview_path",
    "detector",
    "candidate_count",
    "accepted_candidate_count",
    "stable_candidate_count",
    "quality_score",
    "message",
]

PLAYER_TRACK_FIELDS = [
    "video_path",
    "video_stem",
    "rally_id",
    "frame_id",
    "timestamp",
    "player_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "body_image_x",
    "body_image_y",
    "body_anchor_source",
    "body_anchor_confidence",
    "body_anchor_valid",
    "ground_image_x",
    "ground_image_y",
    "ground_anchor_source",
    "ground_anchor_confidence",
    "ground_anchor_valid",
    "pose_model",
    "pose_keypoints_json",
    "pose_confidence",
    "pose_valid",
    "pose_valid_keypoint_count",
    "pose_keypoint_threshold",
    "image_x",
    "image_y",
    "court_x",
    "court_y",
    "confidence",
    "is_interpolated",
    "detector",
]

SHUTTLE_TRACK_FIELDS = [
    "video_path",
    "video_stem",
    "rally_id",
    "frame_id",
    "timestamp",
    "x",
    "y",
    "confidence",
    "is_interpolated",
    "visibility",
]

MAIN_VIEW_FRAME_FIELDS = [
    "sample_frame",
    "timestamp",
    "main_view_score",
    "court_score",
    "geometry_score",
    "layout_score",
    "stability_score",
    "line_score",
    "reject_score",
    "court_area_ratio",
    "court_span_x",
    "court_span_y",
    "player_candidate_count",
    "player_split_sides",
    "is_main_view",
    "reject_reason",
]

MAIN_VIEW_SEGMENT_FIELDS = [
    "segment_id",
    "start_frame",
    "end_frame",
    "start_time",
    "end_time",
    "duration_seconds",
    "label",
    "main_view_score",
    "court_score",
    "geometry_score",
    "layout_score",
    "stability_score",
    "reject_score",
    "frame_count",
]

MAIN_VIEW_QUALITY_FIELDS = [
    "segment_id",
    "rally_id",
    "start_frame",
    "end_frame",
    "start_time",
    "end_time",
    "duration_seconds",
    "quality_score",
    "main_view_score",
    "court_score",
    "geometry_score",
    "layout_score",
    "projection_outlier_ratio",
    "boundary_stuck_ratio",
    "absurd_y_ratio",
    "accepted",
    "reject_reason",
]

REJECTED_SEGMENT_FIELDS = [
    "segment_id",
    "rally_id",
    "start_frame",
    "end_frame",
    "start_time",
    "end_time",
    "duration_seconds",
    "reject_reason",
    "score",
]

FRAME_INDEX_FIELDS = [
    "clean_frame_id",
    "original_time",
    "original_frame_id",
    "segment_id",
]


class MainViewLabel(str, Enum):
    MAIN_VIEW = "MAIN_VIEW"


_LEGACY_MAIN_VIEW_LABELS = {
    "MAIN_LIVE_VIEW": MainViewLabel.MAIN_VIEW,
    "MAIN_BIRDSEYE_LIVE": MainViewLabel.MAIN_VIEW,
}


def parse_main_view_label(value: object) -> MainViewLabel:
    if isinstance(value, MainViewLabel):
        return value
    text = str(value)
    if text == MainViewLabel.MAIN_VIEW.value:
        return MainViewLabel.MAIN_VIEW
    if text in _LEGACY_MAIN_VIEW_LABELS:
        return _LEGACY_MAIN_VIEW_LABELS[text]
    raise ValueError(f"Unsupported Main View label: {value!r}")


class RallyEligibility(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(slots=True)
class MatchRecord:
    match_id: str
    source: str
    url: str
    tournament: str
    year: str
    discipline: str
    round: str
    player_1: str
    player_2: str
    resolution: str
    fps: str
    camera_type: str
    usable_rallies: str = ""
    notes: str = ""


@dataclass(slots=True)
class RallySegment:
    match_id: str
    rally_id: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration_seconds: float
    output_path: str
    frame_interval: str = "[start_frame,end_frame)"
    source_main_view_segment_id: str = ""
    eligibility: str = RallyEligibility.ACCEPTED.value
    eligibility_reason: str = ""
    evidence_sample_count: int = 0
    notes: str = ""


@dataclass(slots=True)
class CourtCalibration:
    artifact_version: str
    validated: bool
    court_type: str
    coordinate_unit: str
    video_path: str
    frame_index: int
    image_size: dict[str, int]
    image_points_tl_tr_br_bl: list[list[float]]
    court_points_tl_tr_br_bl: list[list[float]]
    homography_image_to_court: list[list[float]]
    candidate_source: str
    quality: dict[str, Any]
    temporal_validation: dict[str, Any]


@dataclass(slots=True)
class PlayerTrackRow:
    video_path: str
    video_stem: str
    rally_id: str
    frame_id: int
    timestamp: float
    player_id: str
    bbox_x1: str
    bbox_y1: str
    bbox_x2: str
    bbox_y2: str
    body_image_x: str
    body_image_y: str
    body_anchor_source: str
    body_anchor_confidence: str
    body_anchor_valid: str
    ground_image_x: str
    ground_image_y: str
    ground_anchor_source: str
    ground_anchor_confidence: str
    ground_anchor_valid: str
    pose_model: str
    pose_keypoints_json: str
    pose_confidence: str
    pose_valid: str
    pose_valid_keypoint_count: str
    pose_keypoint_threshold: str
    image_x: str
    image_y: str
    court_x: str
    court_y: str
    confidence: str
    is_interpolated: str
    detector: str


@dataclass(slots=True)
class ShuttleTrackRow:
    video_path: str
    video_stem: str
    rally_id: str
    frame_id: int
    timestamp: float
    x: str
    y: str
    confidence: str
    is_interpolated: str
    visibility: str


@dataclass(slots=True)
class MainViewSegment:
    segment_id: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration_seconds: float
    label: str
    main_view_score: float
    court_score: float
    geometry_score: float
    layout_score: float
    stability_score: float
    reject_score: float
    frame_count: int


@dataclass(slots=True)
class MainViewFrameScore:
    sample_frame: int
    timestamp: float
    main_view_score: float
    court_score: float
    geometry_score: float
    layout_score: float
    stability_score: float
    line_score: float
    reject_score: float
    court_area_ratio: float
    court_span_x: float
    court_span_y: float
    player_candidate_count: int
    player_split_sides: int
    is_main_view: int
    reject_reason: str


@dataclass(slots=True)
class MainViewQualityRow:
    segment_id: str
    rally_id: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration_seconds: float
    quality_score: float
    main_view_score: float
    court_score: float
    geometry_score: float
    layout_score: float
    projection_outlier_ratio: float
    boundary_stuck_ratio: float
    absurd_y_ratio: float
    accepted: int
    reject_reason: str


@dataclass(slots=True)
class RejectedSegment:
    segment_id: str
    rally_id: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration_seconds: float
    reject_reason: str
    score: float


@dataclass(slots=True)
class FrameIndexRow:
    clean_frame_id: int
    original_time: float
    original_frame_id: int
    segment_id: str


class StageName(str, Enum):
    MAIN_VIEW = "main_view"
    RALLY_SEGMENTATION = "rally_segmentation"
    COURT_CALIBRATION = "court_calibration"
    PLAYER_TRACKING = "player_tracking"
    SHUTTLE_TRACKING = "shuttle_tracking"
    TRAJECTORY_SMOOTHING = "trajectory_smoothing"
    VISUALIZATION = "visualization"
    TACTICAL_ANALYSIS = "tactical_analysis"
    BIOMECHANICS_ANALYSIS = "biomechanics_analysis"
    DEMO_RENDERING = "demo_rendering"


STAGE_ORDER = list(StageName)


class StageStatus(str, Enum):
    SUCCESS = "success"
    REJECTED = "rejected"
    EMPTY = "empty"
    FAILED = "failed"
    SKIPPED = "skipped"


class ArtifactKind(str, Enum):
    FILE = "file"
    CSV = "csv"
    JSON = "json"
    DIRECTORY = "directory"
    VIDEO = "video"
    FILE_SET = "file_set"


class ArtifactStatus(str, Enum):
    VALID = "valid"
    MISSING = "missing"
    EMPTY = "empty"
    INVALID = "invalid"


@dataclass(slots=True)
class ArtifactReport:
    name: str
    path: str
    kind: ArtifactKind
    status: ArtifactStatus
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StageResult:
    """Structured result produced by one pipeline stage execution."""

    status: StageStatus
    exit_code: int | None = None
    message: str = ""
    artifacts: list[ArtifactReport] = field(default_factory=list)


@dataclass(slots=True)
class PipelineStageReport:
    name: StageName
    status: StageStatus
    started_at: str
    finished_at: str
    duration_seconds: float
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    exit_code: int | None = None
    artifacts: list[ArtifactReport] = field(default_factory=list)


def dataclass_to_row(item: object) -> dict[str, object]:
    return asdict(item)


def path_as_posix(path: str | Path) -> str:
    return Path(path).as_posix()
