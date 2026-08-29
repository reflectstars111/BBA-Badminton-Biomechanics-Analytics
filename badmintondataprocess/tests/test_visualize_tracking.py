from __future__ import annotations

import pytest

from badminton_data_process.legacy import load_legacy_module


pytest.importorskip("matplotlib")


@pytest.mark.parametrize("rally_count", [1, 2])
def test_shuttle_gallery_accepts_small_rally_sets(tmp_path, rally_count: int) -> None:
    visualization = load_legacy_module("visualize_tracking.py")
    summaries = [
        {
            "video_stem": f"match_rally_{index:03d}",
            "rally_id": f"{index:03d}",
            "track_rows": "2",
            "visible_rows": "1",
        }
        for index in range(1, rally_count + 1)
    ]
    tracks = [
        {
            "video_stem": summary["video_stem"],
            "rally_id": summary["rally_id"],
            "frame_id": "0",
            "x": "10",
            "y": "20",
        }
        for summary in summaries
    ]
    output = tmp_path / "gallery.png"

    visualization.save_shuttle_trajectory_samples(summaries, tracks, output)

    assert output.stat().st_size > 0
