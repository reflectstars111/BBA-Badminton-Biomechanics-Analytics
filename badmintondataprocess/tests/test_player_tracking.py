from __future__ import annotations

from badminton_data_process.legacy import load_legacy_module


def test_near_player_role_uses_feet_without_cropping_full_body() -> None:
    tracking = load_legacy_module("player_tracking.py")
    near_bbox = (281, 255, 340, 410)
    candidates = [
        {
            "bbox": near_bbox,
            "score": 0.71,
            "bottom_center": (310.5, 410.0),
            "area": 9145.0,
        },
        {
            "bbox": (366, 194, 413, 281),
            "score": 0.87,
            "bottom_center": (389.5, 281.0),
            "area": 4089.0,
        },
    ]

    selected = tracking.pick_player_boxes(
        candidates=candidates,
        near_threshold_y=354.0,
        track_states=tracking.init_track_states(),
        frame_shape=(480, 852, 3),
        near_max_track_distance=120.0,
        far_max_track_distance=170.0,
        near_max_missing_frames=4,
        far_max_missing_frames=10,
        role_half_tolerance=48.0,
    )

    assert selected["near"][0] == near_bbox
