from __future__ import annotations

from pathlib import Path

import pytest

from badminton_data_process.core.config import deep_merge, load_config
from badminton_data_process.core.config_schema import ConfigValidationError, parse_config
from badminton_data_process.core.paths import ProjectPaths, RunLayout, discover_project_root


def test_load_default_config_contains_pipeline_sections() -> None:
    root = discover_project_root(Path(__file__))
    config = load_config(root=root)
    assert config["data"]["raw_videos_dir"] == "raw_videos"
    assert config["rally_segmentation"]["sample_every"] == 15
    assert config["player_tracking"]["detector"] == "yolo"
    assert config["player_tracking"]["roles"] == ["near", "far"]
    assert config["player_tracking"]["yolo_confidence"] == 0.25
    assert config["player_tracking"]["yolo_image_size"] == 640
    assert config["player_tracking"]["pose_model"] == "yolo11n-pose.pt"
    assert config["player_tracking"]["rtmpose_mode"] == "balanced"
    assert config["court_calibration"]["detector"] == "contour"
    assert config["demo_rendering"]["output_filename"] == "badminton_analysis_demo.mp4"
    parsed = parse_config(config)
    assert parsed.main_view.threshold == 0.75
    assert parsed.main_view.max_reject_score == 0.4
    assert parsed.court_calibration.min_stable_candidates == 2
    assert parsed.demo_rendering.enabled is True


def test_deep_merge_keeps_nested_defaults() -> None:
    merged = deep_merge(
        {"a": {"b": 1, "c": 2}, "x": 3},
        {"a": {"b": 9}},
    )
    assert merged == {"a": {"b": 9, "c": 2}, "x": 3}


def test_project_paths_resolve_relative_paths() -> None:
    root = discover_project_root(Path(__file__))
    paths = ProjectPaths.from_config(load_config(root=root), root=root)
    assert paths.resolve("metadata/matches.csv") == root / "metadata" / "matches.csv"


def test_strict_config_reports_all_invalid_fields_with_paths() -> None:
    with pytest.raises(ConfigValidationError) as exc_info:
        parse_config(
            {
                "obsolete_outputs": {},
                "main_view": {
                    "sample_every": 0,
                    "threshold": 1.2,
                    "max_reject_score": -0.1,
                },
                "player_tracking": {
                    "roles": ["far"],
                    "court_mask_margin_ratio": 1.1,
                    "tracker": "bytetrack",
                },
                "smoothing": {"max_gap_frames": -1},
            }
        )

    message = str(exc_info.value)
    assert "obsolete_outputs: unknown top-level key" in message
    assert "player_tracking.tracker: unknown key" in message
    assert "main_view.sample_every: must be > 0" in message
    assert "main_view.threshold: must be in [0, 1]" in message
    assert "main_view.max_reject_score: must be in [0, 1]" in message
    assert "player_tracking.roles: supported values" in message
    assert "player_tracking.court_mask_margin_ratio: must be in [0, 1]" in message
    assert "smoothing.max_gap_frames: must be >= 0" in message


def test_rtmpose_configuration_is_typed_and_explicit() -> None:
    parsed = parse_config(
        {
            "player_tracking": {
                "detector": "rtmpose",
                "rtmpose_mode": "balanced",
                "rtmpose_backend": "onnxruntime",
                "rtmpose_device": "cpu",
            }
        }
    )
    assert parsed.player_tracking.detector == "rtmpose"
    assert parsed.player_tracking.rtmpose_pose_input_size == [192, 256]


def test_rtmpose_local_models_must_be_configured_as_a_pair() -> None:
    with pytest.raises(ConfigValidationError, match="must be configured together"):
        parse_config(
            {"player_tracking": {"rtmpose_detector_model": "weights/detector.onnx"}}
        )


def test_all_england_rtmpose_profile_keeps_tracknet_shuttle_model() -> None:
    root = discover_project_root(Path(__file__))
    config = load_config(
        root / "configs/experiments/all_england_2019_rtmpose_validation.yaml",
        root=root,
    )

    assert config["player_tracking"]["detector"] == "rtmpose"
    assert config["player_tracking"]["rtmpose_device"] == "cuda"
    assert config["shuttle_tracking"]["model"] == "tracknet"
    assert config["shuttle_tracking"]["tracknet_weights"] == "weights/TrackNet_best.pt"


def test_low_angle_profile_reuses_gpu_pipeline_with_view_specific_gates() -> None:
    root = discover_project_root(Path(__file__))
    config = load_config(
        root / "configs/experiments/lindan_2026_low_angle.yaml",
        root=root,
    )
    parsed = parse_config(config)

    assert parsed.main_view.max_reject_score == 0.65
    assert parsed.court_calibration.max_out_of_bounds_ratio == 0.25
    assert max(parsed.court_calibration.reference_points or []) > 1.0
    assert parsed.player_tracking.court_mask_margin_ratio == 0.025
    assert parsed.rally_segmentation.min_top_green_ratio == 0.0
    assert parsed.rally_segmentation.max_gap_seconds == 0.75
    assert parsed.player_tracking.rtmpose_device == "cuda"
    assert parsed.shuttle_tracking.model == "tracknet"
    assert config["shuttle_tracking"]["tracknet_device"] == "cuda"


def test_run_layout_resolves_custom_runs_dir_and_owns_artifact_paths(tmp_path) -> None:
    layout = RunLayout.create(tmp_path, "experiment_001", "research/runs")

    assert layout.run_dir == tmp_path / "research" / "runs" / "experiment_001"
    assert layout.annotations_dir == layout.run_dir / "annotations"
    assert layout.player_tracks_csv == layout.annotations_dir / "player_tracks.csv"
    assert layout.tactics_events_csv == layout.outputs_dir / "tactics" / "tactics_events.csv"
    assert layout.demo_output("demo.mp4") == layout.outputs_dir / "demo" / "demo.mp4"


@pytest.mark.parametrize("run_id", ["", "../escape", "nested/run"])
def test_run_layout_rejects_run_id_path_escape(tmp_path, run_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"):
        RunLayout.create(tmp_path, run_id)


def test_run_layout_rejects_demo_path_escape(tmp_path) -> None:
    layout = RunLayout.create(tmp_path, "safe")
    with pytest.raises(ValueError, match="demo filename"):
        layout.demo_output("../escape.mp4")


@pytest.mark.parametrize(
    "relative_path",
    [
        "configs/experiments/synthetic_smoke.yaml",
        "configs/experiments/malaysia_2018_tuned.yaml",
        "configs/experiments/all_england_2019_scoreboard.yaml",
        "configs/experiments/rtmpose_balanced.yaml",
        "configs/experiments/all_england_2019_rtmpose_validation.yaml",
    ],
)
def test_experiment_configs_satisfy_typed_pipeline_schema(relative_path: str) -> None:
    root = discover_project_root(Path(__file__))
    parsed = parse_config(load_config(root / relative_path, root=root))
    assert parsed.rally_segmentation.min_active_samples > 0
