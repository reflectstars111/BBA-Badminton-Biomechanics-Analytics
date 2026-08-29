from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .schemas import ArtifactKind, ArtifactReport, ArtifactStatus


def _report(
    name: str,
    path: Path,
    kind: ArtifactKind,
    status: ArtifactStatus,
    message: str = "",
    **details: object,
) -> ArtifactReport:
    return ArtifactReport(
        name=name,
        path=str(path),
        kind=kind,
        status=status,
        message=message,
        details=details,
    )


def inspect_file(
    path: str | Path,
    *,
    name: str | None = None,
    min_bytes: int = 1,
) -> ArtifactReport:
    artifact_path = Path(path)
    label = name or artifact_path.name
    if not artifact_path.exists():
        return _report(
            label,
            artifact_path,
            ArtifactKind.FILE,
            ArtifactStatus.MISSING,
            "required file does not exist",
        )
    if not artifact_path.is_file():
        return _report(
            label,
            artifact_path,
            ArtifactKind.FILE,
            ArtifactStatus.INVALID,
            "artifact path is not a file",
        )
    size_bytes = artifact_path.stat().st_size
    if size_bytes < min_bytes:
        return _report(
            label,
            artifact_path,
            ArtifactKind.FILE,
            ArtifactStatus.EMPTY,
            f"file has {size_bytes} bytes; expected at least {min_bytes}",
            size_bytes=size_bytes,
        )
    return _report(
        label,
        artifact_path,
        ArtifactKind.FILE,
        ArtifactStatus.VALID,
        size_bytes=size_bytes,
    )


def inspect_csv(
    path: str | Path,
    *,
    name: str | None = None,
    min_rows: int = 0,
    required_fields: Iterable[str] = (),
) -> ArtifactReport:
    artifact_path = Path(path)
    label = name or artifact_path.name
    file_report = inspect_file(artifact_path, name=label)
    if file_report.status != ArtifactStatus.VALID:
        file_report.kind = ArtifactKind.CSV
        return file_report

    try:
        with artifact_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = list(reader.fieldnames or [])
            missing_fields = sorted(set(required_fields) - set(fieldnames))
            row_count = sum(1 for _row in reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        return _report(
            label,
            artifact_path,
            ArtifactKind.CSV,
            ArtifactStatus.INVALID,
            f"cannot read CSV: {exc}",
        )

    details = {
        "size_bytes": artifact_path.stat().st_size,
        "row_count": row_count,
        "fieldnames": fieldnames,
    }
    if not fieldnames:
        return _report(
            label,
            artifact_path,
            ArtifactKind.CSV,
            ArtifactStatus.INVALID,
            "CSV has no header",
            **details,
        )
    if missing_fields:
        return _report(
            label,
            artifact_path,
            ArtifactKind.CSV,
            ArtifactStatus.INVALID,
            f"CSV is missing required fields: {missing_fields}",
            missing_fields=missing_fields,
            **details,
        )
    if row_count < min_rows:
        return _report(
            label,
            artifact_path,
            ArtifactKind.CSV,
            ArtifactStatus.EMPTY if row_count == 0 else ArtifactStatus.INVALID,
            f"CSV has {row_count} rows; expected at least {min_rows}",
            min_rows=min_rows,
            **details,
        )
    return _report(
        label,
        artifact_path,
        ArtifactKind.CSV,
        ArtifactStatus.VALID,
        min_rows=min_rows,
        **details,
    )


def inspect_calibration_json(
    path: str | Path,
    *,
    name: str | None = None,
) -> ArtifactReport:
    artifact_path = Path(path)
    label = name or artifact_path.name
    file_report = inspect_file(artifact_path, name=label)
    if file_report.status != ArtifactStatus.VALID:
        file_report.kind = ArtifactKind.JSON
        return file_report
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return _report(
            label,
            artifact_path,
            ArtifactKind.JSON,
            ArtifactStatus.INVALID,
            f"cannot read calibration JSON: {exc}",
        )
    required = {
        "artifact_version",
        "validated",
        "court_type",
        "coordinate_unit",
        "image_points_tl_tr_br_bl",
        "court_points_tl_tr_br_bl",
        "homography_image_to_court",
        "quality",
        "temporal_validation",
    }
    missing = sorted(required - set(payload)) if isinstance(payload, dict) else sorted(required)
    valid_matrix = (
        isinstance(payload, dict)
        and isinstance(payload.get("homography_image_to_court"), list)
        and len(payload["homography_image_to_court"]) == 3
        and all(isinstance(row, list) and len(row) == 3 for row in payload["homography_image_to_court"])
    )
    details = {
        "size_bytes": artifact_path.stat().st_size,
        "missing_fields": missing,
        "validated": payload.get("validated") if isinstance(payload, dict) else None,
        "court_type": payload.get("court_type") if isinstance(payload, dict) else None,
        "artifact_version": payload.get("artifact_version") if isinstance(payload, dict) else None,
    }
    if missing:
        return _report(
            label,
            artifact_path,
            ArtifactKind.JSON,
            ArtifactStatus.INVALID,
            f"calibration JSON is missing required fields: {missing}",
            **details,
        )
    if payload.get("validated") is not True:
        return _report(
            label,
            artifact_path,
            ArtifactKind.JSON,
            ArtifactStatus.INVALID,
            "calibration JSON is not a Validated Calibration",
            **details,
        )
    if payload.get("coordinate_unit") != "metre" or not valid_matrix:
        return _report(
            label,
            artifact_path,
            ArtifactKind.JSON,
            ArtifactStatus.INVALID,
            "calibration JSON has invalid units or homography shape",
            **details,
        )
    return _report(
        label,
        artifact_path,
        ArtifactKind.JSON,
        ArtifactStatus.VALID,
        **details,
    )


def inspect_directory(
    path: str | Path,
    *,
    name: str | None = None,
    pattern: str = "*",
    min_files: int = 1,
) -> ArtifactReport:
    artifact_path = Path(path)
    label = name or artifact_path.name
    if not artifact_path.exists():
        return _report(
            label,
            artifact_path,
            ArtifactKind.DIRECTORY,
            ArtifactStatus.MISSING,
            "required directory does not exist",
        )
    if not artifact_path.is_dir():
        return _report(
            label,
            artifact_path,
            ArtifactKind.DIRECTORY,
            ArtifactStatus.INVALID,
            "artifact path is not a directory",
        )
    files = [candidate for candidate in artifact_path.glob(pattern) if candidate.is_file()]
    if len(files) < min_files:
        return _report(
            label,
            artifact_path,
            ArtifactKind.DIRECTORY,
            ArtifactStatus.EMPTY,
            f"directory has {len(files)} matching files; expected at least {min_files}",
            file_count=len(files),
            pattern=pattern,
            min_files=min_files,
        )
    return _report(
        label,
        artifact_path,
        ArtifactKind.DIRECTORY,
        ArtifactStatus.VALID,
        file_count=len(files),
        pattern=pattern,
        min_files=min_files,
    )


def inspect_file_set(
    paths: Iterable[str | Path],
    *,
    name: str,
    min_bytes: int = 1,
) -> ArtifactReport:
    artifact_paths = [Path(path) for path in paths]
    if not artifact_paths:
        return _report(
            name,
            Path(name),
            ArtifactKind.FILE_SET,
            ArtifactStatus.EMPTY,
            "file set contains no paths",
            file_count=0,
        )

    missing: list[str] = []
    empty: list[str] = []
    total_bytes = 0
    for artifact_path in artifact_paths:
        if not artifact_path.exists() or not artifact_path.is_file():
            missing.append(str(artifact_path))
            continue
        size_bytes = artifact_path.stat().st_size
        total_bytes += size_bytes
        if size_bytes < min_bytes:
            empty.append(str(artifact_path))

    details = {
        "file_count": len(artifact_paths),
        "total_bytes": total_bytes,
        "missing": missing[:10],
        "empty": empty[:10],
    }
    if missing:
        return _report(
            name,
            Path(name),
            ArtifactKind.FILE_SET,
            ArtifactStatus.MISSING,
            f"file set has {len(missing)} missing or invalid paths",
            **details,
        )
    if empty:
        return _report(
            name,
            Path(name),
            ArtifactKind.FILE_SET,
            ArtifactStatus.EMPTY,
            f"file set has {len(empty)} files smaller than {min_bytes} bytes",
            **details,
        )
    return _report(
        name,
        Path(name),
        ArtifactKind.FILE_SET,
        ArtifactStatus.VALID,
        **details,
    )


def inspect_video(
    path: str | Path,
    *,
    name: str | None = None,
) -> ArtifactReport:
    artifact_path = Path(path)
    label = name or artifact_path.name
    file_report = inspect_file(artifact_path, name=label)
    if file_report.status != ArtifactStatus.VALID:
        file_report.kind = ArtifactKind.VIDEO
        return file_report

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - declared runtime dependency
        return _report(
            label,
            artifact_path,
            ArtifactKind.VIDEO,
            ArtifactStatus.INVALID,
            f"cannot validate video without OpenCV: {exc}",
        )

    capture = cv2.VideoCapture(str(artifact_path))
    try:
        if not capture.isOpened():
            return _report(
                label,
                artifact_path,
                ArtifactKind.VIDEO,
                ArtifactStatus.INVALID,
                "video cannot be opened",
                size_bytes=artifact_path.stat().st_size,
            )
        ok, frame = capture.read()
        if not ok or frame is None:
            return _report(
                label,
                artifact_path,
                ArtifactKind.VIDEO,
                ArtifactStatus.INVALID,
                "video contains no decodable frame",
                size_bytes=artifact_path.stat().st_size,
            )
        return _report(
            label,
            artifact_path,
            ArtifactKind.VIDEO,
            ArtifactStatus.VALID,
            size_bytes=artifact_path.stat().st_size,
            width=int(frame.shape[1]),
            height=int(frame.shape[0]),
            fps=float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        )
    finally:
        capture.release()
