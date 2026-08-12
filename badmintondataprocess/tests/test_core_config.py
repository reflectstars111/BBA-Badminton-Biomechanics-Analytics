from __future__ import annotations

from pathlib import Path

from badminton_data_process.core.config import deep_merge, load_config
from badminton_data_process.core.paths import ProjectPaths, discover_project_root


def test_load_default_config_contains_pipeline_sections() -> None:
    root = discover_project_root(Path(__file__))
    config = load_config(root=root)
    assert config["data"]["raw_videos_dir"] == "raw_videos"
    assert config["rally_segmentation"]["sample_every"] == 15
    assert config["player_tracking"]["detector"] == "heuristic"


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

