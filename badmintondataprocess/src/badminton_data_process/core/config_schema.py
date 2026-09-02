from __future__ import annotations

import math
import types
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints


class ConfigValidationError(ValueError):
    """Raised with every configuration problem found before a run starts."""


def _type_error(path: str, expected: str, value: Any) -> ConfigValidationError:
    return ConfigValidationError(
        f"{path}: expected {expected}, got {type(value).__name__} ({value!r})"
    )


def _check_value(path: str, value: Any, hint: Any) -> Any:
    origin = get_origin(hint)
    if origin is types.UnionType or origin is Union:
        members = [member for member in get_args(hint) if member is not type(None)]
        if value is None and len(members) != len(get_args(hint)):
            return None
        if len(members) == 1:
            return _check_value(path, value, members[0])
        for member in members:
            try:
                return _check_value(path, value, member)
            except ConfigValidationError:
                continue
        expected = " or ".join(getattr(member, "__name__", str(member)) for member in members)
        raise _type_error(path, expected, value)
    if origin is list:
        if not isinstance(value, list):
            raise _type_error(path, "list", value)
        (member_hint,) = get_args(hint) or (Any,)
        return [
            _check_value(f"{path}[{index}]", item, member_hint)
            for index, item in enumerate(value)
        ]
    if hint is Any:
        return value
    if hint is int:
        if type(value) is not int:
            raise _type_error(path, "int", value)
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _type_error(path, "float", value)
        return float(value)
    if hint is bool:
        if type(value) is not bool:
            raise _type_error(path, "bool", value)
        return value
    if hint is str:
        if not isinstance(value, str):
            raise _type_error(path, "str", value)
        return value
    return value


def _parse(cls: type, data: Any, section: str, errors: list[str]):
    if not isinstance(data, dict):
        errors.append(f"{section}: expected mapping, got {type(data).__name__} ({data!r})")
        data = {}
    hints = get_type_hints(cls)
    known = {field.name for field in fields(cls)}
    for key in sorted(key for key in data if key not in known):
        errors.append(f"{section}.{key}: unknown key")

    values: dict[str, Any] = {}
    for field in fields(cls):
        if field.name in data:
            try:
                values[field.name] = _check_value(
                    f"{section}.{field.name}", data[field.name], hints[field.name]
                )
            except ConfigValidationError as exc:
                errors.append(str(exc))
        elif field.default is not MISSING:
            values[field.name] = field.default
        else:
            errors.append(f"{section}.{field.name}: required key missing")
    return cls(**values)


@dataclass(slots=True)
class DataConfig:
    raw_videos_dir: str = "raw_videos"
    rallies_dir: str = "rallies"
    metadata_dir: str = "metadata"
    annotations_dir: str = "annotations"
    outputs_dir: str = "outputs"
    runs_dir: str = "runs"


@dataclass(slots=True)
class MainViewConfig:
    sample_every: int = 30
    threshold: float = 0.75
    max_reject_score: float = 0.4
    min_segment_seconds: float = 3.0
    max_gap_seconds: float = 2.0


@dataclass(slots=True)
class RallySegmentationConfig:
    sample_every: int = 15
    min_rally_seconds: float = 1.0
    max_rally_seconds: float = 120.0
    max_gap_seconds: float = 3.0
    pre_context_seconds: float = 2.2
    post_context_seconds: float = 1.4
    min_active_samples: int = 2
    min_motion_score: float = 0.01
    max_motion_score: float = 0.16
    min_center_green_ratio: float = 0.22
    min_bottom_green_ratio: float = 0.36
    min_line_ratio: float = 0.09
    min_top_green_ratio: float = 0.05
    min_middle_green_ratio: float = 0.20
    max_left_right_green_diff: float = 0.16
    min_top_dark_ratio: float = 0.15
    min_middle_edge_ratio: float = 0.16
    scoreboard_score_roi: list[float] | None = None
    scoreboard_context_roi: list[float] | None = None
    scoreboard_max_lag_seconds: float = 8.0


@dataclass(slots=True)
class CourtCalibrationConfig:
    reference_points: list[float] | None = None
    detector: str = "contour"
    min_line_support: float = 0.15
    min_area_ratio: float = 0.08
    max_out_of_bounds_ratio: float = 0.0
    max_condition_number: float = 1.0e10
    max_reprojection_error_px: float = 1.0
    stability_corner_rmse_ratio: float = 0.04
    min_stable_candidates: int = 2


@dataclass(slots=True)
class PlayerTrackingConfig:
    detector: str = "yolo"
    roles: list[str] | None = None
    court_mask_margin_ratio: float = 0.0
    yolo_model: str = "yolov8n.pt"
    yolo_confidence: float = 0.25
    yolo_image_size: int = 640
    pose_model: str = "yolo11n-pose.pt"
    pose_keypoint_confidence: float = 0.35
    pose_min_keypoints: int = 5
    rtmpose_mode: str = "balanced"
    rtmpose_backend: str = "onnxruntime"
    rtmpose_device: str = "auto"
    rtmpose_detector_model: str = ""
    rtmpose_pose_model: str = ""
    rtmpose_detector_input_size: list[int] | None = None
    rtmpose_pose_input_size: list[int] | None = None
    near_max_track_distance: float = 120.0
    far_max_track_distance: float = 170.0
    near_max_missing_frames: int = 4
    far_max_missing_frames: int = 10
    role_half_tolerance: float = 48.0

    def __post_init__(self) -> None:
        if self.roles is None:
            self.roles = ["near", "far"]
        if self.rtmpose_detector_input_size is None:
            self.rtmpose_detector_input_size = [416, 416]
        if self.rtmpose_pose_input_size is None:
            self.rtmpose_pose_input_size = [192, 256]


@dataclass(slots=True)
class ShuttleTrackingConfig:
    model: str = "motion_bright_baseline"
    tracknet_weights: str = ""
    tracknet_device: str = "auto"
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
    shuttle_max_interpolation_displacement_px: float = 80.0
    window_size: int = 5
    ema_alpha: float = 0.35


@dataclass(slots=True)
class TacticalAnalysisConfig:
    hit_distance_px: float = 80.0
    turn_angle_deg: float = 100.0
    min_event_gap_frames: int = 15


@dataclass(slots=True)
class BiomechanicsAnalysisConfig:
    enabled: bool = True
    classification_backend: str = "none"
    keypoint_confidence: float = 0.35
    min_keypoint_coverage_ratio: float = 0.35
    min_contiguous_frames: int = 5
    event_pre_frames: int = 12
    event_post_frames: int = 20
    wrist_speed_threshold_norm_s: float = 1.0
    shuttle_turn_angle_deg: float = 45.0
    shuttle_proximity_ratio: float = 0.75
    min_shuttle_proximity_score: float = 0.35
    shuttle_turn_span_observations: int = 3
    min_event_gap_frames: int = 8
    min_candidate_score: float = 0.60
    angle_smoothing_window: int = 5
    bst_repository: str = "../third_party/BST-Badminton-Stroke-type-Transformer"
    bst_weights: str = ""
    bst_device: str = "auto"
    bst_min_confidence: float = 0.45
    bst_model_name: str = "BST_AP"
    bst_pose_style: str = "JnB_bone"
    bst_seq_len: int = 30
    bst_num_classes: int = 25
    enable_far_player: bool = True


@dataclass(slots=True)
class DemoRenderingConfig:
    enabled: bool = True
    output_filename: str = "badminton_analysis_demo.mp4"
    max_rallies: int | None = None
    trail_length: int = 18
    event_hold_frames: int = 15
    show_topdown: bool = True
    show_stats: bool = True
    codec: str = "mp4v"
    browser_compatible: bool = True
    preserve_audio: bool = False


@dataclass(slots=True)
class PipelineConfig:
    data: DataConfig
    main_view: MainViewConfig
    rally_segmentation: RallySegmentationConfig
    court_calibration: CourtCalibrationConfig
    player_tracking: PlayerTrackingConfig
    shuttle_tracking: ShuttleTrackingConfig
    smoothing: SmoothingConfig
    tactical_analysis: TacticalAnalysisConfig
    biomechanics_analysis: BiomechanicsAnalysisConfig
    demo_rendering: DemoRenderingConfig


def _positive(errors: list[str], path: str, value: int | float) -> None:
    if value <= 0:
        errors.append(f"{path}: must be > 0, got {value!r}")


def _nonnegative(errors: list[str], path: str, value: int | float) -> None:
    if value < 0:
        errors.append(f"{path}: must be >= 0, got {value!r}")


def _ratio(errors: list[str], path: str, value: float, *, open_lower: bool = False) -> None:
    valid = 0.0 < value <= 1.0 if open_lower else 0.0 <= value <= 1.0
    if not valid:
        interval = "(0, 1]" if open_lower else "[0, 1]"
        errors.append(f"{path}: must be in {interval}, got {value!r}")


def _validate_roi(errors: list[str], path: str, roi: list[float] | None) -> None:
    if roi is None:
        return
    if len(roi) != 4:
        errors.append(f"{path}: expected [x, y, width, height], got {len(roi)} value(s)")
        return
    x, y, width, height = roi
    for index, value in enumerate(roi):
        _ratio(errors, f"{path}[{index}]", value)
    if width <= 0 or height <= 0:
        errors.append(f"{path}: width and height must be > 0")
    if x + width > 1 or y + height > 1:
        errors.append(f"{path}: normalized ROI must fit inside the frame")


def _validate(cfg: PipelineConfig, errors: list[str]) -> None:
    for field in fields(cfg.data):
        value = getattr(cfg.data, field.name)
        if not value.strip():
            errors.append(f"data.{field.name}: path must not be blank")

    _positive(errors, "main_view.sample_every", cfg.main_view.sample_every)
    _ratio(errors, "main_view.threshold", cfg.main_view.threshold)
    _ratio(errors, "main_view.max_reject_score", cfg.main_view.max_reject_score)
    _positive(errors, "main_view.min_segment_seconds", cfg.main_view.min_segment_seconds)
    _nonnegative(errors, "main_view.max_gap_seconds", cfg.main_view.max_gap_seconds)

    rally = cfg.rally_segmentation
    _positive(errors, "rally_segmentation.sample_every", rally.sample_every)
    _positive(errors, "rally_segmentation.min_rally_seconds", rally.min_rally_seconds)
    _positive(errors, "rally_segmentation.max_rally_seconds", rally.max_rally_seconds)
    if rally.max_rally_seconds < rally.min_rally_seconds:
        errors.append(
            "rally_segmentation.max_rally_seconds: must be >= "
            "rally_segmentation.min_rally_seconds"
        )
    _nonnegative(errors, "rally_segmentation.max_gap_seconds", rally.max_gap_seconds)
    _nonnegative(errors, "rally_segmentation.pre_context_seconds", rally.pre_context_seconds)
    _nonnegative(errors, "rally_segmentation.post_context_seconds", rally.post_context_seconds)
    _positive(errors, "rally_segmentation.min_active_samples", rally.min_active_samples)
    _nonnegative(errors, "rally_segmentation.min_motion_score", rally.min_motion_score)
    _nonnegative(errors, "rally_segmentation.max_motion_score", rally.max_motion_score)
    if rally.max_motion_score < rally.min_motion_score:
        errors.append(
            "rally_segmentation.max_motion_score: must be >= "
            "rally_segmentation.min_motion_score"
        )
    for name in (
        "min_center_green_ratio",
        "min_bottom_green_ratio",
        "min_line_ratio",
        "min_top_green_ratio",
        "min_middle_green_ratio",
        "max_left_right_green_diff",
        "min_top_dark_ratio",
        "min_middle_edge_ratio",
    ):
        _ratio(errors, f"rally_segmentation.{name}", getattr(rally, name))
    _validate_roi(errors, "rally_segmentation.scoreboard_score_roi", rally.scoreboard_score_roi)
    _validate_roi(
        errors, "rally_segmentation.scoreboard_context_roi", rally.scoreboard_context_roi
    )
    if (rally.scoreboard_score_roi is None) != (rally.scoreboard_context_roi is None):
        errors.append(
            "rally_segmentation.scoreboard_score_roi: score and context ROI must be configured together"
        )
    _nonnegative(
        errors,
        "rally_segmentation.scoreboard_max_lag_seconds",
        rally.scoreboard_max_lag_seconds,
    )

    calibration = cfg.court_calibration
    if calibration.detector not in {"contour", "hough", "hybrid", "hough_low_angle"}:
        errors.append(
            "court_calibration.detector: must be one of "
            "['contour', 'hough', 'hybrid', 'hough_low_angle'], "
            f"got {calibration.detector!r}"
        )
    _ratio(errors, "court_calibration.min_line_support", calibration.min_line_support)
    _ratio(errors, "court_calibration.min_area_ratio", calibration.min_area_ratio, open_lower=True)
    _ratio(
        errors,
        "court_calibration.max_out_of_bounds_ratio",
        calibration.max_out_of_bounds_ratio,
    )
    _positive(errors, "court_calibration.max_condition_number", calibration.max_condition_number)
    _nonnegative(
        errors,
        "court_calibration.max_reprojection_error_px",
        calibration.max_reprojection_error_px,
    )
    _ratio(
        errors,
        "court_calibration.stability_corner_rmse_ratio",
        calibration.stability_corner_rmse_ratio,
        open_lower=True,
    )
    _positive(errors, "court_calibration.min_stable_candidates", calibration.min_stable_candidates)
    if calibration.reference_points is not None:
        if len(calibration.reference_points) != 8:
            errors.append(
                "court_calibration.reference_points: expected four normalized (x, y) pairs "
                f"(8 values), got {len(calibration.reference_points)}"
            )
        for index, value in enumerate(calibration.reference_points):
            path = f"court_calibration.reference_points[{index}]"
            if calibration.max_out_of_bounds_ratio == 0.0:
                _ratio(errors, path, value)
            elif not math.isfinite(value) or not -2.0 <= value <= 3.0:
                errors.append(
                    f"{path}: must be finite and in [-2, 3] when off-frame corners are enabled"
                )

    player = cfg.player_tracking
    if player.detector not in {"heuristic", "yolo", "yolo_pose", "rtmpose"}:
        errors.append(
            "player_tracking.detector: must be one of "
            "['heuristic', 'yolo', 'yolo_pose', 'rtmpose'], "
            f"got {player.detector!r}"
        )
    if player.roles not in (["near"], ["near", "far"]):
        errors.append(
            "player_tracking.roles: supported values are ['near'] or ['near', 'far'], "
            f"got {player.roles!r}"
        )
    _ratio(
        errors,
        "player_tracking.court_mask_margin_ratio",
        player.court_mask_margin_ratio,
    )
    if player.detector == "yolo" and not player.yolo_model.strip():
        errors.append("player_tracking.yolo_model: must not be blank for detector='yolo'")
    if player.detector == "yolo_pose" and not player.pose_model.strip():
        errors.append("player_tracking.pose_model: must not be blank for detector='yolo_pose'")
    if player.rtmpose_mode not in {"lightweight", "balanced", "performance"}:
        errors.append("player_tracking.rtmpose_mode: unsupported mode")
    if player.rtmpose_backend not in {"onnxruntime", "opencv"}:
        errors.append("player_tracking.rtmpose_backend: must be 'onnxruntime' or 'opencv'")
    if player.rtmpose_device not in {"auto", "cpu", "cuda", "mps"}:
        errors.append(
            "player_tracking.rtmpose_device: must be one of ['auto', 'cpu', 'cuda', 'mps']"
        )
    if bool(player.rtmpose_detector_model) != bool(player.rtmpose_pose_model):
        errors.append(
            "player_tracking.rtmpose_detector_model and rtmpose_pose_model "
            "must be configured together"
        )
    _ratio(errors, "player_tracking.yolo_confidence", player.yolo_confidence)
    _ratio(errors, "player_tracking.pose_keypoint_confidence", player.pose_keypoint_confidence)
    _positive(errors, "player_tracking.pose_min_keypoints", player.pose_min_keypoints)
    if player.pose_min_keypoints > 17:
        errors.append("player_tracking.pose_min_keypoints: must be <= 17")
    for path, size in (
        ("player_tracking.rtmpose_detector_input_size", player.rtmpose_detector_input_size),
        ("player_tracking.rtmpose_pose_input_size", player.rtmpose_pose_input_size),
    ):
        if size is None or len(size) != 2:
            errors.append(f"{path}: expected [width, height]")
        elif any(value <= 0 for value in size):
            errors.append(f"{path}: dimensions must be > 0")
    _positive(errors, "player_tracking.yolo_image_size", player.yolo_image_size)
    _positive(errors, "player_tracking.near_max_track_distance", player.near_max_track_distance)
    _positive(errors, "player_tracking.far_max_track_distance", player.far_max_track_distance)
    _nonnegative(errors, "player_tracking.near_max_missing_frames", player.near_max_missing_frames)
    _nonnegative(errors, "player_tracking.far_max_missing_frames", player.far_max_missing_frames)
    _nonnegative(errors, "player_tracking.role_half_tolerance", player.role_half_tolerance)

    shuttle = cfg.shuttle_tracking
    if shuttle.model not in {"motion_bright_baseline", "tracknet"}:
        errors.append(
            "shuttle_tracking.model: must be one of ['motion_bright_baseline', 'tracknet'], "
            f"got {shuttle.model!r}"
        )
    if shuttle.model == "tracknet" and not shuttle.tracknet_weights.strip():
        errors.append("shuttle_tracking.tracknet_weights: required for model='tracknet'")
    if shuttle.tracknet_device not in {"auto", "cpu", "cuda"}:
        errors.append(
            "shuttle_tracking.tracknet_device: must be one of ['auto', 'cpu', 'cuda']"
        )
    _ratio(errors, "shuttle_tracking.tracknet_vis_threshold", shuttle.tracknet_vis_threshold)
    _nonnegative(errors, "shuttle_tracking.diff_threshold", shuttle.diff_threshold)
    _positive(errors, "shuttle_tracking.max_jump", shuttle.max_jump)
    _nonnegative(errors, "shuttle_tracking.max_missing_frames", shuttle.max_missing_frames)
    if not 0 <= shuttle.min_brightness <= 255:
        errors.append(
            f"shuttle_tracking.min_brightness: must be in [0, 255], got {shuttle.min_brightness!r}"
        )
    _positive(errors, "shuttle_tracking.min_candidate_area", shuttle.min_candidate_area)
    _positive(errors, "shuttle_tracking.max_candidate_area", shuttle.max_candidate_area)
    if shuttle.max_candidate_area < shuttle.min_candidate_area:
        errors.append(
            "shuttle_tracking.max_candidate_area: must be >= shuttle_tracking.min_candidate_area"
        )
    _positive(errors, "shuttle_tracking.max_candidate_size", shuttle.max_candidate_size)
    _nonnegative(errors, "shuttle_tracking.direction_weight", shuttle.direction_weight)
    _nonnegative(errors, "shuttle_tracking.speed_weight", shuttle.speed_weight)

    smoothing = cfg.smoothing
    _ratio(errors, "smoothing.min_confidence", smoothing.min_confidence)
    _nonnegative(errors, "smoothing.max_gap_frames", smoothing.max_gap_frames)
    _positive(
        errors,
        "smoothing.shuttle_max_interpolation_displacement_px",
        smoothing.shuttle_max_interpolation_displacement_px,
    )
    _positive(errors, "smoothing.window_size", smoothing.window_size)
    if smoothing.window_size % 2 == 0:
        errors.append("smoothing.window_size: must be odd")
    _ratio(errors, "smoothing.ema_alpha", smoothing.ema_alpha, open_lower=True)

    tactics = cfg.tactical_analysis
    _positive(errors, "tactical_analysis.hit_distance_px", tactics.hit_distance_px)
    if not 0 < tactics.turn_angle_deg <= 180:
        errors.append(
            f"tactical_analysis.turn_angle_deg: must be in (0, 180], got {tactics.turn_angle_deg!r}"
        )
    _nonnegative(errors, "tactical_analysis.min_event_gap_frames", tactics.min_event_gap_frames)

    biomechanics = cfg.biomechanics_analysis
    if biomechanics.classification_backend not in {"none", "heuristic", "bst"}:
        errors.append(
            "biomechanics_analysis.classification_backend: supported values are "
            "'none', 'heuristic', and 'bst'"
        )
    _ratio(errors, "biomechanics_analysis.keypoint_confidence", biomechanics.keypoint_confidence)
    _ratio(
        errors,
        "biomechanics_analysis.min_keypoint_coverage_ratio",
        biomechanics.min_keypoint_coverage_ratio,
        open_lower=True,
    )
    _positive(errors, "biomechanics_analysis.min_contiguous_frames", biomechanics.min_contiguous_frames)
    _nonnegative(errors, "biomechanics_analysis.event_pre_frames", biomechanics.event_pre_frames)
    _nonnegative(errors, "biomechanics_analysis.event_post_frames", biomechanics.event_post_frames)
    _positive(
        errors,
        "biomechanics_analysis.wrist_speed_threshold_norm_s",
        biomechanics.wrist_speed_threshold_norm_s,
    )
    if not 0 < biomechanics.shuttle_turn_angle_deg <= 180:
        errors.append(
            "biomechanics_analysis.shuttle_turn_angle_deg: must be in (0, 180]"
        )
    _positive(
        errors,
        "biomechanics_analysis.shuttle_proximity_ratio",
        biomechanics.shuttle_proximity_ratio,
    )
    _ratio(
        errors,
        "biomechanics_analysis.min_shuttle_proximity_score",
        biomechanics.min_shuttle_proximity_score,
        open_lower=True,
    )
    _positive(
        errors,
        "biomechanics_analysis.shuttle_turn_span_observations",
        biomechanics.shuttle_turn_span_observations,
    )
    _nonnegative(
        errors,
        "biomechanics_analysis.min_event_gap_frames",
        biomechanics.min_event_gap_frames,
    )
    _ratio(
        errors,
        "biomechanics_analysis.min_candidate_score",
        biomechanics.min_candidate_score,
        open_lower=True,
    )
    _positive(
        errors,
        "biomechanics_analysis.angle_smoothing_window",
        biomechanics.angle_smoothing_window,
    )
    if biomechanics.angle_smoothing_window % 2 == 0:
        errors.append("biomechanics_analysis.angle_smoothing_window: must be odd")
    if biomechanics.bst_device not in {"auto", "cpu", "cuda"}:
        errors.append(
            "biomechanics_analysis.bst_device: supported values are 'auto', 'cpu', and 'cuda'"
        )
    _ratio(
        errors,
        "biomechanics_analysis.bst_min_confidence",
        biomechanics.bst_min_confidence,
    )
    if biomechanics.bst_model_name not in {"BST", "BST_CG", "BST_AP", "BST_CG_AP"}:
        errors.append(
            "biomechanics_analysis.bst_model_name: unsupported BST architecture"
        )
    if biomechanics.bst_pose_style not in {"J_only", "JnB_bone"}:
        errors.append(
            "biomechanics_analysis.bst_pose_style: supported values are 'J_only' and 'JnB_bone'"
        )
    _positive(errors, "biomechanics_analysis.bst_seq_len", biomechanics.bst_seq_len)
    if biomechanics.bst_num_classes not in {25, 35}:
        errors.append("biomechanics_analysis.bst_num_classes: supported values are 25 and 35")
    if biomechanics.classification_backend == "bst" and not biomechanics.bst_weights.strip():
        errors.append(
            "biomechanics_analysis.bst_weights: required when classification_backend='bst'"
        )

    demo = cfg.demo_rendering
    if (
        not demo.output_filename.strip()
        or Path(demo.output_filename).name != demo.output_filename
    ):
        errors.append("demo_rendering.output_filename: must be a non-empty filename, not a path")
    if demo.max_rallies is not None:
        _positive(errors, "demo_rendering.max_rallies", demo.max_rallies)
    _positive(errors, "demo_rendering.trail_length", demo.trail_length)
    _nonnegative(errors, "demo_rendering.event_hold_frames", demo.event_hold_frames)
    if len(demo.codec) != 4:
        errors.append("demo_rendering.codec: must contain exactly four characters")


def parse_config(config: dict[str, Any]) -> PipelineConfig:
    if not isinstance(config, dict):
        raise ConfigValidationError(
            f"configuration: expected mapping, got {type(config).__name__} ({config!r})"
        )

    errors: list[str] = []
    sections: tuple[tuple[str, type], ...] = (
        ("data", DataConfig),
        ("main_view", MainViewConfig),
        ("rally_segmentation", RallySegmentationConfig),
        ("court_calibration", CourtCalibrationConfig),
        ("player_tracking", PlayerTrackingConfig),
        ("shuttle_tracking", ShuttleTrackingConfig),
        ("smoothing", SmoothingConfig),
        ("tactical_analysis", TacticalAnalysisConfig),
        ("biomechanics_analysis", BiomechanicsAnalysisConfig),
        ("demo_rendering", DemoRenderingConfig),
    )
    known_sections = {name for name, _ in sections}
    for key in sorted(key for key in config if key not in known_sections):
        errors.append(f"{key}: unknown top-level key")

    parsed = {
        name: _parse(section_type, config.get(name, {}), name, errors)
        for name, section_type in sections
    }
    cfg = PipelineConfig(**parsed)
    _validate(cfg, errors)
    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise ConfigValidationError(f"Invalid pipeline configuration:\n{details}")
    return cfg
