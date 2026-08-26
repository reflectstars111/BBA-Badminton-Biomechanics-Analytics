from __future__ import annotations

import types
from dataclasses import MISSING, dataclass, fields
from typing import Any, Union, get_args, get_origin, get_type_hints


class ConfigValidationError(ValueError):
    pass


def _check_value(section: str, key: str, value: Any, hint: Any) -> Any:
    origin = get_origin(hint)
    if origin is types.UnionType or origin is Union:
        members = [a for a in get_args(hint) if a is not type(None)]
        if value is None:
            return None
        for member in members:
            try:
                return _check_value(section, key, value, member)
            except ConfigValidationError:
                continue
        raise ConfigValidationError(
            f"{section}.{key}: expected {members}, got {type(value).__name__} ({value!r})"
        )
    if hint is int:
        if type(value) is not int:
            raise ConfigValidationError(
                f"{section}.{key}: expected int, got {type(value).__name__} ({value!r})"
            )
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigValidationError(
                f"{section}.{key}: expected float, got {type(value).__name__} ({value!r})"
            )
        return float(value)
    if hint is bool:
        if type(value) is not bool:
            raise ConfigValidationError(
                f"{section}.{key}: expected bool, got {type(value).__name__} ({value!r})"
            )
        return value
    if hint is str:
        if not isinstance(value, str):
            raise ConfigValidationError(
                f"{section}.{key}: expected str, got {type(value).__name__} ({value!r})"
            )
        return value
    if hint is list:
        if not isinstance(value, list):
            raise ConfigValidationError(
                f"{section}.{key}: expected list, got {type(value).__name__} ({value!r})"
            )
        return value
    return value


def _parse(cls: type, data: dict[str, Any], section: str):
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = sorted(k for k in data if k not in known)
    if unknown:
        raise ConfigValidationError(f"{section}: unknown key(s): {', '.join(unknown)}")
    values: dict[str, Any] = {}
    for f in fields(cls):
        if f.name in data:
            values[f.name] = _check_value(section, f.name, data[f.name], hints[f.name])
        elif f.default is not MISSING:
            values[f.name] = f.default
        else:
            raise ConfigValidationError(f"{section}.{f.name}: required key missing")
    return cls(**values)


@dataclass(slots=True)
class RallySegmentationConfig:
    output_dir: str = "rallies"
    metadata_csv: str = "metadata/rallies.csv"
    sample_every: int = 15
    min_rally_seconds: float = 1.0
    max_rally_seconds: float = 120.0
    max_gap_seconds: float = 3.0
    min_motion_score: float = 0.01
    max_motion_score: float = 0.16
    min_center_green_ratio: float = 0.22
    min_bottom_green_ratio: float = 0.36
    min_line_ratio: float = 0.09
    min_top_green_ratio: float = 0.05
    min_middle_green_ratio: float = 0.20
    max_left_right_green_diff: float = 0.16
    min_top_dark_ratio: float = 0.80
    min_middle_edge_ratio: float = 0.16
    pad_before_seconds: float = 0.4
    pad_after_seconds: float = 0.6
    max_pre_context_seconds: float = 2.2
    max_post_context_seconds: float = 1.4
    allowed_context_drop_samples: int = 1
    enable_manual_review: bool = True
    scoreboard_score_roi: list | None = None
    scoreboard_context_roi: list | None = None
    scoreboard_max_lag_seconds: float = 8.0


@dataclass(slots=True)
class CourtCalibrationConfig:
    court_type: str = "singles"
    output_dir: str = "annotations/court_calibration"
    preview_dir: str = "outputs/court_calibration_debug"
    summary_csv: str = "annotations/court_calibration_summary.csv"
    save_homography: bool = True
    reference_points: list | None = None
    min_line_support: float = 0.15


@dataclass(slots=True)
class PlayerTrackingConfig:
    calibration_dir: str = "annotations/court_calibration"
    output: str = "annotations/player_tracks.csv"
    summary_csv: str = "annotations/player_tracking_summary.csv"
    debug_dir: str = "outputs/player_tracking_debug"
    detector: str = "heuristic"
    tracker: str = "role_association"
    pose_model: str = "none"
    use_ankle_midpoint: bool = False
    yolo_model: str = "yolov8n.pt"
    yolo_confidence: float = 0.12
    yolo_image_size: int = 1280
    near_max_track_distance: float = 120.0
    far_max_track_distance: float = 170.0
    near_max_missing_frames: int = 4
    far_max_missing_frames: int = 10
    role_half_tolerance: float = 48.0


@dataclass(slots=True)
class ShuttleTrackingConfig:
    calibration_dir: str = "annotations/court_calibration"
    output: str = "annotations/shuttle_tracks.csv"
    summary_csv: str = "annotations/shuttle_tracking_summary.csv"
    debug_dir: str = "outputs/shuttle_tracking_debug"
    model: str = "motion_bright_baseline"
    future_model_interface: str = "tracknet"
    tracknet_weights: str = ""
    tracknet_vis_threshold: float = 0.15
    diff_threshold: int = 18
    max_jump: float = 80.0
    max_missing_frames: int = 3
    min_brightness: int = 165
    min_candidate_area: float = 1.0
    max_candidate_area: float = 55.0
    max_candidate_size: int = 14
    direction_weight: float = 24.0
    speed_weight: float = 0.35


@dataclass(slots=True)
class SmoothingConfig:
    min_confidence: float = 0.2
    max_gap_frames: int = 4
    window_size: int = 5
    ema_alpha: float = 0.35


@dataclass(slots=True)
class TacticalAnalysisConfig:
    output_dir: str = "outputs/tactics"
    hit_distance_px: float = 80.0
    turn_angle_deg: float = 100.0
    min_event_gap_frames: int = 15


@dataclass(slots=True)
class PipelineConfig:
    rally_segmentation: RallySegmentationConfig
    court_calibration: CourtCalibrationConfig
    player_tracking: PlayerTrackingConfig
    shuttle_tracking: ShuttleTrackingConfig
    smoothing: SmoothingConfig
    tactical_analysis: TacticalAnalysisConfig


def parse_config(config: dict[str, Any]) -> PipelineConfig:
    return PipelineConfig(
        rally_segmentation=_parse(
            RallySegmentationConfig, config.get("rally_segmentation", {}), "rally_segmentation"
        ),
        court_calibration=_parse(
            CourtCalibrationConfig, config.get("court_calibration", {}), "court_calibration"
        ),
        player_tracking=_parse(
            PlayerTrackingConfig, config.get("player_tracking", {}), "player_tracking"
        ),
        shuttle_tracking=_parse(
            ShuttleTrackingConfig, config.get("shuttle_tracking", {}), "shuttle_tracking"
        ),
        smoothing=_parse(SmoothingConfig, config.get("smoothing", {}), "smoothing"),
        tactical_analysis=_parse(
            TacticalAnalysisConfig,
            config.get("tactical_analysis", {}),
            "tactical_analysis",
        ),
    )
