from __future__ import annotations

import json

import pytest

from badminton_data_process.analysis.biomechanics.evaluation import evaluate_action_events


def test_evaluation_reports_detection_timing_player_and_classification_metrics() -> None:
    predictions = [
        {
            "video_stem": "rally",
            "rally_id": "001",
            "candidate_frame": 11,
            "player_id": "near",
            "classification_eligibility": "eligible",
            "stroke_class": "smash",
            "top2_json": json.dumps(
                [
                    {"stroke_class": "smash", "confidence": 0.7},
                    {"stroke_class": "clear", "confidence": 0.2},
                ]
            ),
        },
        {
            "video_stem": "rally",
            "rally_id": "001",
            "candidate_frame": 31,
            "player_id": "near",
            "classification_eligibility": "eligible",
            "stroke_class": "drive",
            "top2_json": json.dumps(
                [
                    {"stroke_class": "drive", "confidence": 0.6},
                    {"stroke_class": "clear", "confidence": 0.3},
                ]
            ),
        },
        {
            "video_stem": "rally",
            "rally_id": "001",
            "candidate_frame": 80,
            "player_id": "far",
            "classification_eligibility": "not_eligible",
            "classification_reject_reason": "low_confidence_or_unknown",
        },
    ]
    references = [
        {
            "video_stem": "rally",
            "rally_id": "001",
            "reference_frame": 10,
            "player_id": "near",
            "stroke_class": "smash",
            "review_status": "accepted",
        },
        {
            "video_stem": "rally",
            "rally_id": "001",
            "reference_frame": 30,
            "player_id": "far",
            "stroke_class": "clear",
            "review_status": "accepted",
        },
    ]

    result = evaluate_action_events(predictions, references, tolerance_frames=3)

    assert result["matched_events"] == 2
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(1.0)
    assert result["mean_absolute_frame_error"] == pytest.approx(1.0)
    assert result["player_attribution_accuracy"] == pytest.approx(0.5)
    assert result["classification_top1_accuracy"] == pytest.approx(0.5)
    assert result["classification_top2_accuracy"] == pytest.approx(1.0)
    assert result["classification_macro_f1"] == pytest.approx(0.5)
    assert result["classification_reject_ratio"] == pytest.approx(1 / 3)


def test_prediction_seeded_review_excludes_pending_rows_and_does_not_claim_recall() -> None:
    predictions = [
        {
            "video_stem": "rally",
            "rally_id": "001",
            "event_id": event_id,
            "candidate_frame": frame,
            "player_id": "near",
            "classification_eligibility": "not_eligible",
        }
        for event_id, frame in (("E1", 10), ("E2", 20), ("E3", 30))
    ]
    review_rows = [
        {
            "video_stem": "rally",
            "rally_id": "001",
            "event_id": "E1",
            "reference_frame": 10,
            "player_id": "near",
            "annotation_scope": "prediction_seeded",
            "review_status": "accepted",
        },
        {
            "video_stem": "rally",
            "rally_id": "001",
            "event_id": "E2",
            "reference_frame": 20,
            "player_id": "near",
            "annotation_scope": "prediction_seeded",
            "review_status": "rejected",
        },
        {
            "video_stem": "rally",
            "rally_id": "001",
            "event_id": "E3",
            "reference_frame": 30,
            "player_id": "near",
            "annotation_scope": "prediction_seeded",
            "review_status": "pending",
        },
    ]

    result = evaluate_action_events(predictions, review_rows)

    assert result["annotation_scope"] == "prediction_seeded"
    assert result["reviewed_rows"] == 2
    assert result["pending_rows"] == 1
    assert result["evaluated_predictions"] == 2
    assert result["matched_events"] == 1
    assert result["false_positive_events"] == 1
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] is None
    assert result["false_negative_events"] is None
