from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

from badminton_data_process.core.io import read_csv_rows, write_json


GROUND_TRUTH_FIELDS = [
    "video_stem",
    "rally_id",
    "event_id",
    "reference_frame",
    "player_id",
    "stroke_class",
    "annotation_scope",
    "review_status",
    "review_image",
    "notes",
]


def _frame(row: Mapping[str, object], field: str) -> int | None:
    try:
        return int(float(str(row.get(field, ""))))
    except (TypeError, ValueError):
        return None


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _top2_classes(row: Mapping[str, object]) -> set[str]:
    try:
        payload = json.loads(str(row.get("top2_json") or "[]"))
    except (TypeError, json.JSONDecodeError):
        return set()
    return {
        str(item.get("stroke_class", ""))
        for item in payload
        if isinstance(item, dict) and item.get("stroke_class")
    }


def evaluate_action_events(
    predicted_rows: Iterable[Mapping[str, object]],
    ground_truth_rows: Iterable[Mapping[str, object]],
    *,
    tolerance_frames: int = 3,
) -> dict[str, object]:
    all_predictions = [dict(row) for row in predicted_rows]
    truth_rows = [dict(row) for row in ground_truth_rows]
    reviewed_rows = [
        row
        for row in truth_rows
        if str(row.get("review_status", "accepted")).lower()
        in {"accepted", "rejected"}
    ]
    references = [
        row
        for row in reviewed_rows
        if str(row.get("review_status", "accepted")).lower() == "accepted"
    ]
    scopes = {
        str(row.get("annotation_scope") or "exhaustive").strip().lower()
        for row in reviewed_rows
    }
    annotation_scope = scopes.pop() if len(scopes) == 1 else (
        "mixed" if scopes else "unreviewed"
    )

    # A prediction-seeded review sheet only evaluates rows the reviewer has
    # accepted or rejected. Pending rows are deliberately excluded instead of
    # becoming false positives. An exhaustive sheet covers complete rally
    # groups and can therefore support recall.
    reviewed_event_ids = {
        str(row.get("event_id")) for row in reviewed_rows if row.get("event_id")
    }
    reviewed_groups = {
        (str(row.get("video_stem", "")), str(row.get("rally_id", "")))
        for row in reviewed_rows
    }
    if annotation_scope == "prediction_seeded":
        predictions = [
            row
            for row in all_predictions
            if str(row.get("event_id", "")) in reviewed_event_ids
        ]
    elif reviewed_groups:
        predictions = [
            row
            for row in all_predictions
            if (str(row.get("video_stem", "")), str(row.get("rally_id", "")))
            in reviewed_groups
        ]
    else:
        predictions = []
    unmatched_predictions = set(range(len(predictions)))
    matches: list[tuple[dict[str, object], dict[str, object], int]] = []
    for reference in references:
        reference_frame = _frame(reference, "reference_frame")
        if reference_frame is None:
            continue
        group = (str(reference.get("video_stem", "")), str(reference.get("rally_id", "")))
        candidates = []
        for index in unmatched_predictions:
            prediction = predictions[index]
            if (
                str(prediction.get("video_stem", "")),
                str(prediction.get("rally_id", "")),
            ) != group:
                continue
            predicted_frame = _frame(prediction, "candidate_frame")
            if predicted_frame is None:
                continue
            error = abs(predicted_frame - reference_frame)
            if error <= tolerance_frames:
                candidates.append((error, index))
        if not candidates:
            continue
        error, selected_index = min(candidates)
        unmatched_predictions.remove(selected_index)
        matches.append((predictions[selected_index], reference, error))

    true_positive = len(matches)
    false_positive = len(unmatched_predictions)
    false_negative = len(references) - true_positive
    player_correct = sum(
        prediction.get("player_id") == reference.get("player_id")
        for prediction, reference, _ in matches
    )
    class_pairs = [
        (str(prediction.get("stroke_class") or "unknown"), str(reference.get("stroke_class")))
        for prediction, reference, _ in matches
        if reference.get("stroke_class")
    ]
    top1_correct = sum(predicted == expected for predicted, expected in class_pairs)
    top2_correct = sum(
        str(reference.get("stroke_class")) in _top2_classes(prediction)
        for prediction, reference, _ in matches
        if reference.get("stroke_class")
    )
    classes = sorted({expected for _, expected in class_pairs})
    f1_values = []
    for class_name in classes:
        tp = sum(predicted == expected == class_name for predicted, expected in class_pairs)
        fp = sum(predicted == class_name and expected != class_name for predicted, expected in class_pairs)
        fn = sum(predicted != class_name and expected == class_name for predicted, expected in class_pairs)
        precision = _ratio(tp, tp + fp) or 0.0
        recall = _ratio(tp, tp + fn) or 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    reject_reasons = Counter(
        str(row.get("classification_reject_reason"))
        for row in predictions
        if row.get("classification_eligibility") != "eligible"
        and row.get("classification_reject_reason")
    )
    errors = [error for _, _, error in matches]
    return {
        "schema_version": "bba_biomechanics_evaluation_v2",
        "tolerance_frames": tolerance_frames,
        "annotation_scope": annotation_scope,
        "reviewed_rows": len(reviewed_rows),
        "pending_rows": sum(
            str(row.get("review_status", "")).lower() == "pending"
            for row in truth_rows
        ),
        "ground_truth_events": len(references),
        "predicted_events": len(all_predictions),
        "evaluated_predictions": len(predictions),
        "matched_events": true_positive,
        "false_positive_events": false_positive,
        "false_negative_events": (
            false_negative if annotation_scope == "exhaustive" else None
        ),
        "precision": _ratio(true_positive, true_positive + false_positive),
        "recall": (
            _ratio(true_positive, true_positive + false_negative)
            if annotation_scope == "exhaustive"
            else None
        ),
        "mean_absolute_frame_error": statistics.fmean(errors) if errors else None,
        "median_absolute_frame_error": statistics.median(errors) if errors else None,
        "player_attribution_accuracy": _ratio(player_correct, true_positive),
        "classification_samples": len(class_pairs),
        "classification_top1_accuracy": _ratio(top1_correct, len(class_pairs)),
        "classification_top2_accuracy": _ratio(top2_correct, len(class_pairs)),
        "classification_macro_f1": statistics.fmean(f1_values) if f1_values else None,
        "classification_reject_ratio": _ratio(
            sum(row.get("classification_eligibility") != "eligible" for row in predictions),
            len(predictions),
        ),
        "classification_reject_reasons": dict(sorted(reject_reasons.items())),
    }


def evaluate_action_event_csv(
    predictions_csv: Path,
    ground_truth_csv: Path,
    output_json: Path,
    *,
    tolerance_frames: int = 3,
) -> dict[str, object]:
    result = evaluate_action_events(
        read_csv_rows(predictions_csv),
        read_csv_rows(ground_truth_csv),
        tolerance_frames=tolerance_frames,
    )
    write_json(output_json, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate BBA biomechanics action events.")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tolerance-frames", type=int, default=3)
    args = parser.parse_args(argv)
    if args.tolerance_frames < 0:
        parser.error("--tolerance-frames must be non-negative")
    result = evaluate_action_event_csv(
        args.predictions,
        args.ground_truth,
        args.output,
        tolerance_frames=args.tolerance_frames,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
