from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from badminton_data_process.analysis.biomechanics.review import (
    REVIEW_DRAFT_FIELDS,
    export_action_event_review,
)
from badminton_data_process.core.io import read_csv_rows, write_csv_rows


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        25.0,
        (160, 90),
    )
    assert writer.isOpened()
    for frame_id in range(8):
        frame = np.full((90, 160, 3), 20 + frame_id * 12, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_review_export_writes_three_frame_montage_and_pending_draft(tmp_path: Path) -> None:
    video_path = tmp_path / "rally.mp4"
    _write_video(video_path)
    events_csv = tmp_path / "action_events.csv"
    tracks_csv = tmp_path / "player_tracks.csv"
    write_csv_rows(
        events_csv,
        [
            "video_path",
            "video_stem",
            "rally_id",
            "event_id",
            "candidate_frame",
            "player_id",
            "event_eligibility",
            "candidate_score",
            "stroke_class",
            "classification_confidence",
        ],
        [
            {
                "video_path": str(video_path),
                "video_stem": "rally",
                "rally_id": "001",
                "event_id": "001_E001",
                "candidate_frame": 4,
                "player_id": "near",
                "event_eligibility": "eligible",
                "candidate_score": 0.8,
                "stroke_class": "smash",
                "classification_confidence": 0.7,
            }
        ],
    )
    write_csv_rows(
        tracks_csv,
        [
            "video_stem", "rally_id", "frame_id", "player_id",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "pose_keypoints_json",
        ],
        [
            {
                "video_stem": "rally",
                "rally_id": "001",
                "frame_id": frame_id,
                "player_id": "near",
                "bbox_x1": 40,
                "bbox_y1": 15,
                "bbox_x2": 100,
                "bbox_y2": 85,
                "pose_keypoints_json": "",
            }
            for frame_id in (2, 4, 6)
        ],
    )

    result = export_action_event_review(
        events_csv,
        tmp_path / "review",
        player_tracks_csv=tracks_csv,
        frame_radius=2,
        panel_width=160,
    )

    assert result["status"] == "success"
    assert result["selected_events"] == 1
    assert result["rendered_montages"] == 1
    rows = read_csv_rows(tmp_path / "review" / "biomechanics_ground_truth_draft.csv")
    assert list(rows[0]) == REVIEW_DRAFT_FIELDS
    assert rows[0]["annotation_scope"] == "prediction_seeded"
    assert rows[0]["review_status"] == "pending"
    assert rows[0]["reference_frame"] == "4"
    assert rows[0]["predicted_stroke_class"] == "smash"
    image_path = tmp_path / "review" / rows[0]["review_image"]
    assert image_path.is_file()
    montage = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert montage.shape[:2] == (90, 480)


def test_review_sampling_round_robins_across_rallies(tmp_path: Path) -> None:
    from badminton_data_process.analysis.biomechanics.review import _select_events

    rows = [
        {
            "video_stem": "video",
            "rally_id": rally,
            "event_id": f"{rally}_{index}",
            "candidate_frame": index,
            "event_eligibility": "eligible",
        }
        for rally in ("001", "002", "003")
        for index in range(3)
    ]

    selected = _select_events(rows, 4)

    assert [row["rally_id"] for row in selected] == ["001", "002", "003", "001"]


def test_review_sampling_balances_player_and_classification_outcome() -> None:
    from badminton_data_process.analysis.biomechanics.review import _select_events

    rows = []
    for index in range(12):
        rows.append(
            {
                "video_stem": "video",
                "rally_id": "001",
                "event_id": f"dominant_{index}",
                "candidate_frame": index,
                "event_eligibility": "eligible",
                "player_id": "near",
                "classification_eligibility": "not_eligible",
                "classification_reject_reason": "insufficient_opponent_pose_coverage",
            }
        )
    rows.extend(
        [
            {
                "video_stem": "video",
                "rally_id": "002",
                "event_id": "near_eligible",
                "candidate_frame": 20,
                "event_eligibility": "eligible",
                "player_id": "near",
                "classification_eligibility": "eligible",
            },
            {
                "video_stem": "video",
                "rally_id": "003",
                "event_id": "far_candidate_gap",
                "candidate_frame": 30,
                "event_eligibility": "eligible",
                "player_id": "far",
                "classification_eligibility": "not_eligible",
                "classification_reject_reason": "insufficient_candidate_pose_coverage",
            },
        ]
    )

    selected = _select_events(rows, 3)

    assert {row["event_id"] for row in selected} == {
        "dominant_0",
        "near_eligible",
        "far_candidate_gap",
    }


def test_review_manifest_reports_classifier_readiness_diagnostics(tmp_path: Path) -> None:
    events_csv = tmp_path / "action_events.csv"
    write_csv_rows(
        events_csv,
        [
            "video_path", "video_stem", "rally_id", "event_id", "candidate_frame",
            "player_id", "event_eligibility", "classification_eligibility",
            "classification_reject_reason",
        ],
        [
            {
                "video_stem": "video",
                "rally_id": "001",
                "event_id": "E1",
                "candidate_frame": 10,
                "player_id": "near",
                "event_eligibility": "eligible",
                "classification_eligibility": "eligible",
            },
            {
                "video_stem": "video",
                "rally_id": "001",
                "event_id": "E2",
                "candidate_frame": 20,
                "player_id": "far",
                "event_eligibility": "eligible",
                "classification_eligibility": "not_eligible",
                "classification_reject_reason": "insufficient_candidate_pose_coverage",
            },
        ],
    )

    result = export_action_event_review(events_csv, tmp_path / "review")

    diagnostics = result["diagnostics"]
    assert diagnostics["candidate_events"] == 2
    assert diagnostics["classification_eligible_events"] == 1
    assert diagnostics["classification_eligible_ratio"] == 0.5
    assert diagnostics["classification_outcomes"] == {
        "eligible": 1,
        "insufficient_candidate_pose_coverage": 1,
    }
    assert diagnostics["players"]["near"]["classification_eligible_ratio"] == 1.0
    assert diagnostics["players"]["far"]["classification_eligible_ratio"] == 0.0
