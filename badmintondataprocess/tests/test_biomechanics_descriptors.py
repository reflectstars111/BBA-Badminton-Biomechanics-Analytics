from __future__ import annotations

from badminton_data_process.analysis.biomechanics.descriptors import enrich_action_events


def _event(player_id: str = "near") -> dict[str, object]:
    return {
        "video_stem": "rally",
        "rally_id": "001",
        "event_id": "001_E001",
        "player_id": player_id,
        "candidate_frame": 5,
        "candidate_score": 0.9,
        "window_start_frame": 0,
        "window_end_frame": 11,
    }


def _players(player_id: str = "near") -> list[dict[str, object]]:
    positions = [0.0, 0.1, 0.3, 0.6, 0.9, 1.2, 0.9, 0.6, 0.3, 0.1, 0.0]
    return [
        {
            "video_stem": "rally",
            "rally_id": "001",
            "player_id": player_id,
            "frame_id": frame,
            "timestamp": frame * 0.1,
            "court_x": x,
            "court_y": 3.0,
            "pose_keypoints_json": "",
        }
        for frame, x in enumerate(positions)
    ]


def _kinematics(player_id: str = "near") -> list[dict[str, object]]:
    return [
        {
            "video_stem": "rally",
            "rally_id": "001",
            "player_id": player_id,
            "frame_id": frame,
            "kinematics_eligibility": "eligible",
            "support_width_ratio": 0.3 + frame * 0.01,
            "body_support_offset_ratio": (-1) ** frame * 0.1,
            "trunk_lean_deg": frame - 5,
            "left_knee_angle_deg": 130,
            "right_knee_angle_deg": 125,
            "left_elbow_angle_deg": 100 + frame,
            "right_elbow_angle_deg": 110 + frame,
            "left_shoulder_angle_deg": 80 + frame,
            "right_shoulder_angle_deg": 90 + frame,
        }
        for frame in range(11)
    ]


def test_event_descriptors_report_stability_footwork_and_recovery() -> None:
    enriched = enrich_action_events([_event()], _players(), _kinematics())

    assert len(enriched) == 1
    event = enriched[0]
    assert event["stability_eligibility"] == "eligible"
    assert event["footwork_eligibility"] == "eligible"
    assert event["mean_support_width_ratio"] == 0.35
    assert event["body_support_offset_rms"] == 0.1
    assert event["mean_knee_asymmetry_deg"] == 5.0
    assert event["candidate_left_elbow_angle_deg"] == 105.0
    assert event["candidate_right_shoulder_angle_deg"] == 95.0
    assert event["event_path_distance_m"] == 2.4
    assert event["recovery_frame"] == 8
    assert event["recovery_time_s"] == 0.3


def test_far_footwork_can_be_disabled_without_removing_stability() -> None:
    enriched = enrich_action_events(
        [_event("far")],
        _players("far"),
        _kinematics("far"),
        enable_far_player=False,
    )

    event = enriched[0]
    assert event["stability_eligibility"] == "eligible"
    assert event["footwork_eligibility"] == "not_eligible"
    assert event["footwork_reject_reason"] == "far_player_analysis_disabled"
