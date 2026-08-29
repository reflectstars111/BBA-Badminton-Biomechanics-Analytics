from __future__ import annotations

import argparse
import math
from pathlib import Path

from badminton_data_process.core.io import ensure_dir, read_json, write_csv_rows
from badminton_data_process.core.schemas import (
    RALLY_DECISION_FIELDS,
    RALLY_FIELDS,
    MainViewLabel,
    RallyEligibility,
    parse_main_view_label,
)
from badminton_data_process.rally.activity import (
    analyze_video,
    frame_metrics,
    frame_motion_scores,
)
from badminton_data_process.legacy import load_legacy_module, run_legacy_main


def _module():
    return load_legacy_module("rally_segmentation.py")


def merge_candidate_segments(*args, **kwargs):
    return _module().merge_candidate_segments(*args, **kwargs)


def select_live_rallies(*args, **kwargs):
    return _module().select_live_rallies(*args, **kwargs)


def segment_rallies(*args, **kwargs):
    return _module().segment_rallies(*args, **kwargs)


def _validate_args(args: argparse.Namespace) -> None:
    if args.sample_every <= 0:
        raise SystemExit("--sample-every must be > 0")
    if args.min_rally_seconds <= 0:
        raise SystemExit("--min-rally-seconds must be > 0")
    if args.max_rally_seconds <= args.min_rally_seconds:
        raise SystemExit("--max-rally-seconds must be > --min-rally-seconds")
    if args.max_gap_seconds < 0:
        raise SystemExit("--max-gap-seconds must be >= 0")
    if args.pre_context_seconds < 0 or args.post_context_seconds < 0:
        raise SystemExit("rally context seconds must be >= 0")
    if args.min_active_samples <= 0:
        raise SystemExit("--min-active-samples must be > 0")
    if not math.isfinite(args.min_motion_score) or args.min_motion_score < 0:
        raise SystemExit("--min-motion-score must be a finite value >= 0")
    if not math.isfinite(args.max_motion_score) or args.max_motion_score <= args.min_motion_score:
        raise SystemExit("--max-motion-score must be > --min-motion-score")


def _timeline_intervals(timeline_path: Path) -> list[tuple[int, int, str]]:
    payload = read_json(timeline_path)
    intervals: list[tuple[int, int, str]] = []
    for index, item in enumerate(payload, start=1):
        if parse_main_view_label(item.get("label")) != MainViewLabel.MAIN_VIEW:
            continue
        start = int(item["start_frame"])
        end = int(item["end_frame"])
        if end <= start:
            continue
        intervals.append((start, end, str(item.get("segment_id", f"{index:03d}"))))
    return intervals


def _rally_decision(
    source_segment_id: str,
    start_frame: int,
    end_frame: int,
    status: RallyEligibility,
    reason: str,
    evidence_sample_count: int,
) -> dict[str, object]:
    return {
        "candidate_id": "",
        "source_main_view_segment_id": source_segment_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frame_interval": "[start_frame,end_frame)",
        "status": status.value,
        "reason": reason,
        "evidence_sample_count": evidence_sample_count,
        "rally_id": "",
        "output_path": "",
    }


def classify_usable_rallies(
    analysis_rows: list[dict[str, float]],
    *,
    source_segment_id: str,
    interval_start: int,
    interval_end: int,
    sample_every: int,
    fps: float,
    min_rally_seconds: float,
    max_rally_seconds: float,
    max_gap_seconds: float,
    pre_context_seconds: float,
    post_context_seconds: float,
    min_active_samples: int,
) -> list[dict[str, object]]:
    """Classify active-play groups inside one half-open Main View interval."""
    rows = sorted(
        (
            row
            for row in analysis_rows
            if interval_start <= int(row["sample_frame"]) < interval_end
        ),
        key=lambda row: int(row["sample_frame"]),
    )
    if not rows:
        return [
            _rally_decision(
                source_segment_id,
                interval_start,
                interval_end,
                RallyEligibility.REJECTED,
                "no_sampled_frames_in_main_view",
                0,
            )
        ]

    active_frames = sorted(
        {int(row["sample_frame"]) for row in rows if bool(row.get("is_candidate"))}
    )
    if not active_frames:
        return [
            _rally_decision(
                source_segment_id,
                interval_start,
                interval_end,
                RallyEligibility.REJECTED,
                "no_active_play_evidence",
                0,
            )
        ]

    max_missing_gap_frames = int(round(max_gap_seconds * fps))
    groups: list[list[int]] = [[active_frames[0]]]
    for frame_id in active_frames[1:]:
        missing_gap = frame_id - groups[-1][-1] - sample_every
        if missing_gap <= max_missing_gap_frames:
            groups[-1].append(frame_id)
        else:
            groups.append([frame_id])

    pre_frames = int(round(pre_context_seconds * fps))
    post_frames = int(round(post_context_seconds * fps))
    extended: list[list[int]] = []
    previous_end = interval_start
    for index, group in enumerate(groups):
        start = max(interval_start, group[0] - pre_frames)
        end = min(interval_end, group[-1] + sample_every + post_frames)
        if index + 1 < len(groups):
            end = min(end, groups[index + 1][0])
        start = max(start, previous_end)
        if end > start:
            extended.append([start, end, len(group)])
            previous_end = end

    decisions: list[dict[str, object]] = []
    min_rally_frames = max(1, int(round(min_rally_seconds * fps)))
    max_rally_frames = max(min_rally_frames, int(round(max_rally_seconds * fps)))
    for start, end, evidence_count in extended:
        frame_count = end - start
        if evidence_count < min_active_samples:
            status = RallyEligibility.REJECTED
            reason = "insufficient_active_samples"
        elif frame_count < min_rally_frames:
            status = RallyEligibility.REJECTED
            reason = "below_min_rally_duration"
        elif frame_count > max_rally_frames:
            status = RallyEligibility.REJECTED
            reason = "above_max_rally_duration"
        else:
            status = RallyEligibility.ACCEPTED
            reason = "active_play_evidence"
        decisions.append(
            _rally_decision(
                source_segment_id,
                start,
                end,
                status,
                reason,
                evidence_count,
            )
        )
    return decisions


def segment_rallies_with_timeline(
    input_path: Path,
    timeline_path: Path,
    output_dir: Path,
    metadata_csv: Path,
    decisions_csv: Path,
    sample_every: int,
    min_rally_seconds: float,
    max_rally_seconds: float,
    max_gap_seconds: float,
    min_motion_score: float,
    max_motion_score: float,
    min_center_green_ratio: float,
    min_bottom_green_ratio: float,
    min_line_ratio: float,
    min_top_green_ratio: float,
    min_middle_green_ratio: float,
    max_left_right_green_diff: float,
    min_top_dark_ratio: float,
    min_middle_edge_ratio: float,
    pre_context_seconds: float,
    post_context_seconds: float,
    min_active_samples: int,
    overwrite: bool,
    scoreboard_score_roi: tuple[float, float, float, float] | None = None,
    scoreboard_context_roi: tuple[float, float, float, float] | None = None,
    scoreboard_max_lag_seconds: float = 8.0,
) -> int:
    module = _module()
    intervals = _timeline_intervals(timeline_path)
    if not intervals:
        raise RuntimeError(f"Timeline contains no valid intervals: {timeline_path}")

    analysis_rows, fps, source_frame_count, _ = analyze_video(
        input_path,
        sample_every,
        min_motion_score,
        max_motion_score,
        min_center_green_ratio,
        min_bottom_green_ratio,
        min_line_ratio,
        min_top_green_ratio,
        min_middle_green_ratio,
        max_left_right_green_diff,
        min_top_dark_ratio,
        min_middle_edge_ratio,
    )

    decisions: list[dict[str, object]] = []
    timeline_filtered_rows: list[dict[str, float]] = []
    for start_frame, end_frame, segment_id in intervals:
        end_frame = min(end_frame, source_frame_count)
        if end_frame <= start_frame:
            decisions.append(
                _rally_decision(
                    segment_id,
                    start_frame,
                    start_frame,
                    RallyEligibility.REJECTED,
                    "main_view_interval_outside_source_video",
                    0,
                )
            )
            continue
        rows = [
            row
            for row in analysis_rows
            if start_frame <= int(row["sample_frame"]) < end_frame
        ]
        timeline_filtered_rows.extend(rows)
        decisions.extend(
            classify_usable_rallies(
                rows,
                source_segment_id=segment_id,
                interval_start=start_frame,
                interval_end=end_frame,
                sample_every=sample_every,
                fps=fps,
                min_rally_seconds=min_rally_seconds,
                max_rally_seconds=max_rally_seconds,
                max_gap_seconds=max_gap_seconds,
                pre_context_seconds=pre_context_seconds,
                post_context_seconds=post_context_seconds,
                min_active_samples=min_active_samples,
            )
        )

    for index, decision in enumerate(decisions, start=1):
        decision["candidate_id"] = f"C{index:03d}"
        decision["start_time"] = round(int(decision["start_frame"]) / fps, 3)
        decision["end_time"] = round(int(decision["end_frame"]) / fps, 3)
        decision["duration_seconds"] = round(
            (int(decision["end_frame"]) - int(decision["start_frame"])) / fps,
            3,
        )

    accepted_decisions = [
        decision
        for decision in decisions
        if decision["status"] == RallyEligibility.ACCEPTED.value
    ]
    if scoreboard_score_roi is not None and scoreboard_context_roi is not None:
        score_changes = module.detect_score_changes(
            input_path,
            scoreboard_score_roi,
            scoreboard_context_roi,
        )
        selected_segments = set(
            module.select_live_rallies(
                [
                    (int(decision["start_frame"]), int(decision["end_frame"]))
                    for decision in accepted_decisions
                ],
                score_changes,
                fps,
                max_score_lag_seconds=scoreboard_max_lag_seconds,
            )
        )
        for decision in accepted_decisions:
            segment = (int(decision["start_frame"]), int(decision["end_frame"]))
            if segment in selected_segments:
                decision["reason"] = "scoreboard_confirmed"
            else:
                decision["status"] = RallyEligibility.REJECTED.value
                decision["reason"] = "no_scoreboard_confirmation"
        accepted_decisions = [
            decision
            for decision in decisions
            if decision["status"] == RallyEligibility.ACCEPTED.value
        ]

    accepted_segments = [
        (int(decision["start_frame"]), int(decision["end_frame"]))
        for decision in accepted_decisions
    ]
    match_id = module.extract_match_id(input_path)
    output_dir = ensure_dir(output_dir)
    if overwrite:
        removed = module.remove_previous_outputs(output_dir, match_id)
        if removed:
            print(f"Removed previous clips: {removed}")
    rally_rows = module.write_rally_clips(
        input_path,
        output_dir,
        accepted_segments,
        fps,
        match_id,
    )
    for decision, row in zip(accepted_decisions, rally_rows, strict=True):
        row["frame_interval"] = "[start_frame,end_frame)"
        row["source_main_view_segment_id"] = decision["source_main_view_segment_id"]
        row["eligibility"] = RallyEligibility.ACCEPTED.value
        row["eligibility_reason"] = decision["reason"]
        row["evidence_sample_count"] = decision["evidence_sample_count"]
        row["notes"] = "usable rally accepted by active-play quality gate"
        decision["rally_id"] = row["rally_id"]
        decision["output_path"] = row["output_path"]
    module.write_analysis_csv(metadata_csv, timeline_filtered_rows)
    write_csv_rows(metadata_csv, RALLY_FIELDS, rally_rows)
    write_csv_rows(decisions_csv, RALLY_DECISION_FIELDS, decisions)

    print(f"Video: {input_path}")
    print(f"Timeline: {timeline_path}")
    print(f"FPS: {fps:.3f}")
    print(f"Timeline intervals: {len(intervals)}")
    print(f"Sampled frames in timeline: {len(timeline_filtered_rows)}")
    print(f"Accepted Usable Rallies: {len(rally_rows)}")
    print(f"Rejected rally candidates: {len(decisions) - len(rally_rows)}")
    print(f"Rally metadata saved to: {metadata_csv}")
    print(f"Rally decisions saved to: {decisions_csv}")
    return 0


def build_timeline_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Segment rally clips, optionally constrained by main-view timeline.")
    parser.add_argument("input", type=Path, help="Path to the full match video.")
    parser.add_argument("--timeline", type=Path, default=None, help="Main-view timeline JSON.")
    parser.add_argument("--output-dir", type=Path, default=Path("rallies"))
    parser.add_argument("--metadata-csv", type=Path, default=Path("metadata/rallies.csv"))
    parser.add_argument("--decisions-csv", type=Path, default=Path("metadata/rally_decisions.csv"))
    parser.add_argument("--sample-every", type=int, default=15)
    parser.add_argument("--min-rally-seconds", type=float, default=4.0)
    parser.add_argument("--max-rally-seconds", type=float, default=45.0)
    parser.add_argument("--max-gap-seconds", type=float, default=3.0)
    parser.add_argument("--min-motion-score", type=float, default=0.01)
    parser.add_argument("--max-motion-score", type=float, default=0.16)
    parser.add_argument("--min-center-green-ratio", type=float, default=0.22)
    parser.add_argument("--min-bottom-green-ratio", type=float, default=0.36)
    parser.add_argument("--min-line-ratio", type=float, default=0.09)
    parser.add_argument("--min-top-green-ratio", type=float, default=0.05)
    parser.add_argument("--min-middle-green-ratio", type=float, default=0.20)
    parser.add_argument("--max-left-right-green-diff", type=float, default=0.16)
    parser.add_argument("--min-top-dark-ratio", type=float, default=0.15)
    parser.add_argument("--min-middle-edge-ratio", type=float, default=0.16)
    parser.add_argument("--pre-context-seconds", type=float, default=2.2)
    parser.add_argument("--post-context-seconds", type=float, default=1.4)
    parser.add_argument("--min-active-samples", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    if argv and "--timeline" in argv:
        args = build_timeline_parser().parse_args(argv)
        _validate_args(args)
        return segment_rallies_with_timeline(
            input_path=args.input,
            timeline_path=args.timeline,
            output_dir=args.output_dir,
            metadata_csv=args.metadata_csv,
            decisions_csv=args.decisions_csv,
            sample_every=args.sample_every,
            min_rally_seconds=args.min_rally_seconds,
            max_rally_seconds=args.max_rally_seconds,
            max_gap_seconds=args.max_gap_seconds,
            min_motion_score=args.min_motion_score,
            max_motion_score=args.max_motion_score,
            min_center_green_ratio=args.min_center_green_ratio,
            min_bottom_green_ratio=args.min_bottom_green_ratio,
            min_line_ratio=args.min_line_ratio,
            min_top_green_ratio=args.min_top_green_ratio,
            min_middle_green_ratio=args.min_middle_green_ratio,
            max_left_right_green_diff=args.max_left_right_green_diff,
            min_top_dark_ratio=args.min_top_dark_ratio,
            min_middle_edge_ratio=args.min_middle_edge_ratio,
            pre_context_seconds=args.pre_context_seconds,
            post_context_seconds=args.post_context_seconds,
            min_active_samples=args.min_active_samples,
            overwrite=args.overwrite,
        )
    return run_legacy_main("rally_segmentation.py", argv)
