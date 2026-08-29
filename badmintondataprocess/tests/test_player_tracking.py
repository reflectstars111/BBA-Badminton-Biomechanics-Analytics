from __future__ import annotations

import numpy as np

from badminton_data_process.legacy import load_legacy_module
from badminton_data_process.tracking.player import tracking


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


def test_player_role_uses_court_projection_instead_of_corner_average() -> None:
    # This calibration maps image y=332.55 to the net (court_y=6.7), while
    # averaging the four corner y values incorrectly puts the split at 353.75.
    homography = np.array(
        [
            [0.0684714863, 0.0310984492, -26.24374343],
            [0.0, 0.3806376864, -97.43715506],
            [-0.0000373092, 0.0101215130, 1.0],
        ],
        dtype=np.float32,
    )
    near_bbox = (305, 294, 326, 350)
    far_bbox = (366, 194, 413, 281)
    candidates = [
        {
            "bbox": near_bbox,
            "score": 0.9,
            "bottom_center": (315.5, 350.0),
            "area": 1176.0,
        },
        {
            "bbox": far_bbox,
            "score": 0.6,
            "bottom_center": (389.5, 281.0),
            "area": 4089.0,
        },
    ]

    selected = tracking.pick_player_boxes(
        candidates=candidates,
        near_threshold_y=353.75,
        track_states=tracking.init_track_states(),
        frame_shape=(480, 852, 3),
        near_max_track_distance=120.0,
        far_max_track_distance=170.0,
        near_max_missing_frames=4,
        far_max_missing_frames=10,
        role_half_tolerance=48.0,
        homography=homography,
    )

    assert selected["near"][0] == near_bbox
    assert selected["far"][0] == far_bbox


def test_real_detection_suppresses_overlapping_other_role_prediction() -> None:
    homography = np.array(
        [
            [0.0684714863, 0.0310984492, -26.24374343],
            [0.0, 0.3806376864, -97.43715506],
            [-0.0000373092, 0.0101215130, 1.0],
        ],
        dtype=np.float32,
    )
    near_prediction = (305, 294, 326, 350)
    far_detection = (305, 294, 326, 330)
    track_states = tracking.init_track_states()
    track_states["near"]["bbox"] = near_prediction

    selected = tracking.pick_player_boxes(
        candidates=[
            {
                "bbox": far_detection,
                "score": 0.8,
                "bottom_center": (315.5, 330.0),
                "area": 756.0,
            }
        ],
        near_threshold_y=332.55,
        track_states=track_states,
        frame_shape=(480, 852, 3),
        near_max_track_distance=120.0,
        far_max_track_distance=170.0,
        near_max_missing_frames=4,
        far_max_missing_frames=10,
        role_half_tolerance=48.0,
        homography=homography,
    )

    assert selected["near"][0] is None
    assert selected["far"][0] == far_detection


def test_near_only_tracking_does_not_create_far_role() -> None:
    candidates = [
        {
            "bbox": (281, 255, 340, 410),
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
        player_ids=("near",),
    )

    assert set(selected) == {"near"}
