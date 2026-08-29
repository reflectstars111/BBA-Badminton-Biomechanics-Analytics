from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from badminton_data_process.calibration.hough import generate_hough_candidates
from badminton_data_process.calibration.reference import STANDARD_COURT, score_court_line_support
from badminton_data_process.calibration.validation import (
    CalibrationCandidate,
    CalibrationCandidateSource,
    CalibrationThresholds,
    CalibrationValidationResult,
    select_stable_calibration,
    validate_calibration_candidate,
)
from badminton_data_process.core.io import ensure_dir, read_csv_rows, write_csv_rows


COURT_FIELDNAMES = [
    "video_path",
    "video_stem",
    "status",
    "frame_index",
    "json_path",
    "preview_path",
    "detector",
    "candidate_count",
    "accepted_candidate_count",
    "stable_candidate_count",
    "quality_score",
    "message",
]
COURT_POINTS = STANDARD_COURT.corners_array()

__all__ = [
    "COURT_FIELDNAMES",
    "COURT_POINTS",
    "iter_video_paths",
    "representative_frame_indices",
    "read_frame_at",
    "representative_frame",
    "court_mask",
    "detect_court_corners",
    "draw_preview",
    "normalized_corners",
    "court_line_support",
    "calibrate_video",
    "calibrate_courts",
    "build_parser",
    "parse_reference_points",
    "main",
]


def iter_video_paths(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".csv":
        return [Path(row["output_path"]) for row in read_csv_rows(input_path) if row.get("output_path")]
    if input_path.is_dir():
        return sorted(input_path.glob("*.mp4"))
    return [input_path]


def representative_frame_indices(frame_count: int) -> list[int]:
    if frame_count <= 0:
        return [0]
    indices: list[int] = []
    for ratio in (0.50, 0.35, 0.65, 0.20, 0.80):
        index = max(0, min(frame_count - 1, int(frame_count * ratio)))
        if index not in indices:
            indices.append(index)
    return indices


def read_frame_at(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"failed to read frame {frame_index}")
    return frame


def representative_frame(capture: cv2.VideoCapture) -> tuple[np.ndarray, int]:
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    best_frame: np.ndarray | None = None
    best_index = 0
    best_score = -1.0
    for frame_index in representative_frame_indices(frame_count):
        frame = read_frame_at(capture, frame_index)
        score = float(np.mean(court_mask(frame) > 0))
        if score > best_score:
            best_frame = frame
            best_index = frame_index
            best_score = score
    if best_frame is None:
        raise RuntimeError("failed to read representative frame")
    return best_frame, best_index


def court_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    mask = (
        (hue >= 35)
        & (hue <= 95)
        & (saturation >= 35)
        & (value >= 30)
    ).astype(np.uint8) * 255
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)


def _band_points(points: np.ndarray, y_min: float, y_max: float) -> np.ndarray:
    selected = points[(points[:, 1] >= y_min) & (points[:, 1] <= y_max)]
    return selected if len(selected) >= 4 else points


def contour_candidate(frame: np.ndarray, frame_index: int = 0) -> CalibrationCandidate:
    mask = court_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("no court contour detected")
    contour = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(contour) / max(1.0, float(frame.shape[0] * frame.shape[1]))
    if area_ratio < 0.08:
        raise RuntimeError("detected court area is too small")
    points = contour.reshape(-1, 2)
    _, y, _, height = cv2.boundingRect(contour)
    top_band = _band_points(points, y, y + height * 0.38)
    bottom_band = _band_points(points, y + height * 0.58, y + height)
    corners = np.asarray(
        [
            top_band[np.argmin(top_band[:, 0])],
            top_band[np.argmax(top_band[:, 0])],
            bottom_band[np.argmax(bottom_band[:, 0])],
            bottom_band[np.argmin(bottom_band[:, 0])],
        ],
        dtype=np.float32,
    )
    return CalibrationCandidate(
        corners=corners,
        source=CalibrationCandidateSource.GREEN_CONTOUR,
        frame_index=frame_index,
        diagnostics={"green_contour_area_ratio": round(float(area_ratio), 6)},
    )


def detect_court_corners(frame: np.ndarray) -> np.ndarray:
    """Compatibility Interface returning green-contour TL, TR, BR, BL corners."""
    return contour_candidate(frame).corners


def normalized_corners(
    frame_shape: tuple[int, int, int],
    points: list[float] | list[list[float]] | tuple[tuple[float, float], ...],
) -> np.ndarray:
    height, width = frame_shape[:2]
    corners = np.asarray(points, dtype=np.float32)
    if corners.size == 8:
        corners = corners.reshape(4, 2)
    if corners.shape != (4, 2):
        raise ValueError("reference court points must contain four x,y pairs")
    if np.any(corners < 0.0) or np.any(corners > 1.0):
        raise ValueError("reference court points must be normalized from 0 to 1")
    corners[:, 0] *= width
    corners[:, 1] *= height
    return corners


def court_line_support(frame: np.ndarray, corners: np.ndarray, line_width: int = 9) -> float:
    # line_width remains on the compatibility Interface. The Implementation
    # uses a resolution-aware tolerance and every regulation line.
    _ = line_width
    return score_court_line_support(frame, corners).score


def _frame_candidates(frame: np.ndarray, frame_index: int, detector: str) -> list[CalibrationCandidate]:
    candidates: list[CalibrationCandidate] = []
    if detector == "contour":
        try:
            candidates.append(contour_candidate(frame, frame_index))
        except RuntimeError:
            pass
    # In hybrid mode the colour mask may remain useful as rough scene/ROI
    # evidence, but its perimeter is not a badminton regulation line and must
    # never become the four formal Homography correspondences.
    if detector in {"hough", "hybrid"}:
        candidates.extend(generate_hough_candidates(frame, frame_index=frame_index))
    if detector not in {"contour", "hough", "hybrid"}:
        raise ValueError(f"unsupported court calibration detector: {detector}")
    return candidates


def draw_preview(
    frame: np.ndarray,
    result: CalibrationValidationResult,
    *,
    stable_frames: list[int] | None = None,
) -> np.ndarray:
    preview = frame.copy()
    corners = result.candidate.corners
    color = (0, 220, 0) if result.accepted else (0, 0, 255)
    cv2.polylines(preview, [corners.astype(np.int32)], True, color, 3)
    for name, point in zip(("TL", "TR", "BR", "BL"), corners, strict=True):
        cv2.circle(preview, tuple(np.rint(point).astype(int)), 6, (0, 100, 255), -1)
        cv2.putText(
            preview,
            name,
            tuple(np.rint(point + np.asarray([6, -6])).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    for line, start, end in STANDARD_COURT.project_lines(corners):
        line_color = (245, 245, 245) if line.name != "net" else (40, 220, 255)
        cv2.line(
            preview,
            tuple(np.rint(start).astype(int)),
            tuple(np.rint(end).astype(int)),
            line_color,
            1,
            cv2.LINE_AA,
        )
    text = (
        f"{result.candidate.source.value} quality={result.quality.quality_score:.3f} "
        f"stable={len(stable_frames or [])}"
    )
    cv2.putText(preview, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    if result.reasons:
        cv2.putText(
            preview,
            ",".join(result.reasons)[:100],
            (12, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return preview


def _save_calibration(
    video_path: Path,
    frame_shape: tuple[int, int, int],
    selected: CalibrationValidationResult,
    all_results: list[CalibrationValidationResult],
    stable_frames: list[int],
    output_json: Path,
    *,
    detector: str,
    stability_corner_rmse_ratio: float,
    min_stable_candidates: int,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_version": "2.0",
        "validated": True,
        "court_type": STANDARD_COURT.court_type,
        "coordinate_unit": "metre",
        "video_path": str(video_path),
        "frame_index": selected.candidate.frame_index,
        "image_size": {"width": int(frame_shape[1]), "height": int(frame_shape[0])},
        "image_points_tl_tr_br_bl": selected.candidate.corners.astype(float).tolist(),
        "court_points_tl_tr_br_bl": STANDARD_COURT.corners_array().astype(float).tolist(),
        "homography_image_to_court": selected.homography_image_to_court.astype(float).tolist(),
        "candidate_source": selected.candidate.source.value,
        "quality": selected.quality.as_dict(),
        "temporal_validation": {
            "detector": detector,
            "sampled_frame_indices": sorted({item.candidate.frame_index for item in all_results}),
            "candidate_count": len(all_results),
            "accepted_candidate_count": sum(item.accepted for item in all_results),
            "stable_frame_indices": stable_frames,
            "max_corner_rmse_ratio": stability_corner_rmse_ratio,
            "min_stable_candidates": min_stable_candidates,
        },
        "candidate_audit": [item.as_dict() for item in all_results],
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _failure(video_path: Path, detector: str, message: str) -> dict[str, object]:
    return {
        "video_path": str(video_path),
        "video_stem": video_path.stem,
        "status": "failed",
        "frame_index": -1,
        "json_path": "",
        "preview_path": "",
        "detector": detector,
        "candidate_count": 0,
        "accepted_candidate_count": 0,
        "stable_candidate_count": 0,
        "quality_score": "",
        "message": message,
    }


def calibrate_video(
    video_path: Path,
    output_dir: Path,
    preview_dir: Path,
    reference_points: list[float] | list[list[float]] | None = None,
    min_line_support: float = 0.15,
    *,
    detector: str = "contour",
    min_area_ratio: float = 0.08,
    max_condition_number: float = 1.0e10,
    max_reprojection_error_px: float = 1.0,
    stability_corner_rmse_ratio: float = 0.04,
    min_stable_candidates: int = 2,
) -> dict[str, object]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return _failure(video_path, detector, "cannot open video")
    thresholds = CalibrationThresholds(
        min_area_ratio=min_area_ratio,
        min_line_support=min_line_support,
        max_condition_number=max_condition_number,
        max_reprojection_error_px=max_reprojection_error_px,
    )
    try:
        frame_by_index: dict[int, np.ndarray] = {}
        results: list[CalibrationValidationResult] = []
        if reference_points is not None:
            frame, frame_index = representative_frame(capture)
            frame_by_index[frame_index] = frame
            candidate = CalibrationCandidate(
                corners=normalized_corners(frame.shape, reference_points),
                source=CalibrationCandidateSource.MANUAL,
                frame_index=frame_index,
            )
            results.append(validate_calibration_candidate(frame, candidate, thresholds=thresholds))
            selected = results[0] if results[0].accepted else None
            stable_frames = [frame_index] if selected else []
        else:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            for frame_index in representative_frame_indices(frame_count):
                frame = read_frame_at(capture, frame_index)
                frame_by_index[frame_index] = frame
                for candidate in _frame_candidates(frame, frame_index, detector):
                    results.append(validate_calibration_candidate(frame, candidate, thresholds=thresholds))
            selected, stable_frames = select_stable_calibration(
                results,
                next(iter(frame_by_index.values())).shape if frame_by_index else (1, 1, 3),
                max_corner_rmse_ratio=stability_corner_rmse_ratio,
                min_stable_candidates=min_stable_candidates,
            )
        if selected is None:
            reasons = sorted({reason for result in results for reason in result.reasons})
            failure = _failure(
                video_path,
                detector,
                "calibration rejected: " + (", ".join(reasons) if reasons else "no stable validated candidate"),
            )
            failure["candidate_count"] = len(results)
            failure["accepted_candidate_count"] = sum(result.accepted for result in results)
            failure["stable_candidate_count"] = len(stable_frames)
            return failure

        frame = frame_by_index[selected.candidate.frame_index]
        output_json = output_dir / f"{video_path.stem}.json"
        output_preview = preview_dir / f"{video_path.stem}.png"
        output_preview.parent.mkdir(parents=True, exist_ok=True)
        _save_calibration(
            video_path,
            frame.shape,
            selected,
            results,
            stable_frames,
            output_json,
            detector=detector,
            stability_corner_rmse_ratio=stability_corner_rmse_ratio,
            min_stable_candidates=min_stable_candidates,
        )
        cv2.imwrite(str(output_preview), draw_preview(frame, selected, stable_frames=stable_frames))
        return {
            "video_path": str(video_path),
            "video_stem": video_path.stem,
            "status": "success",
            "frame_index": selected.candidate.frame_index,
            "json_path": str(output_json),
            "preview_path": str(output_preview),
            "detector": detector,
            "candidate_count": len(results),
            "accepted_candidate_count": sum(result.accepted for result in results),
            "stable_candidate_count": len(stable_frames),
            "quality_score": round(selected.quality.quality_score, 6),
            "message": (
                f"validated {selected.candidate.source.value} calibration; "
                f"quality={selected.quality.quality_score:.3f}; stable_frames={stable_frames}"
            ),
        }
    except Exception as exc:  # pragma: no cover - runtime data dependent
        return _failure(video_path, detector, str(exc))
    finally:
        capture.release()


def calibrate_courts(
    input_path: Path,
    output_dir: Path,
    preview_dir: Path,
    summary_csv: Path,
    reference_points: list[float] | list[list[float]] | None = None,
    min_line_support: float = 0.15,
    **options: Any,
) -> int:
    videos = iter_video_paths(input_path)
    output_dir = ensure_dir(output_dir)
    preview_dir = ensure_dir(preview_dir)
    rows = [
        calibrate_video(
            video_path,
            output_dir,
            preview_dir,
            reference_points=reference_points,
            min_line_support=min_line_support,
            **options,
        )
        for video_path in videos
    ]
    write_csv_rows(summary_csv, COURT_FIELDNAMES, rows)
    success_count = sum(row["status"] == "success" for row in rows)
    print(f"Videos processed: {len(rows)}")
    print(f"Successful calibrations: {success_count}")
    print(f"Summary CSV: {summary_csv}")
    return 0 if success_count == len(rows) else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate badminton court homography.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("annotations/court_calibration"))
    parser.add_argument("--preview-dir", type=Path, default=Path("outputs/court_calibration_debug"))
    parser.add_argument("--summary-csv", type=Path, default=Path("annotations/court_calibration_summary.csv"))
    parser.add_argument("--reference-points", default=None)
    parser.add_argument("--detector", choices=("contour", "hough", "hybrid"), default="contour")
    parser.add_argument("--min-line-support", type=float, default=0.15)
    parser.add_argument("--min-area-ratio", type=float, default=0.08)
    parser.add_argument("--max-condition-number", type=float, default=1.0e10)
    parser.add_argument("--max-reprojection-error-px", type=float, default=1.0)
    parser.add_argument("--stability-corner-rmse-ratio", type=float, default=0.04)
    parser.add_argument("--min-stable-candidates", type=int, default=2)
    return parser


def parse_reference_points(value: str | None) -> list[list[float]] | None:
    if value is None:
        return None
    points = [[float(item) for item in pair.split(",")] for pair in value.split(";")]
    if len(points) != 4 or any(len(point) != 2 for point in points):
        raise SystemExit("--reference-points must be x,y;x,y;x,y;x,y")
    return points


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return calibrate_courts(
        input_path=args.input,
        output_dir=args.output_dir,
        preview_dir=args.preview_dir,
        summary_csv=args.summary_csv,
        reference_points=parse_reference_points(args.reference_points),
        min_line_support=args.min_line_support,
        detector=args.detector,
        min_area_ratio=args.min_area_ratio,
        max_condition_number=args.max_condition_number,
        max_reprojection_error_px=args.max_reprojection_error_px,
        stability_corner_rmse_ratio=args.stability_corner_rmse_ratio,
        min_stable_candidates=args.min_stable_candidates,
    )
