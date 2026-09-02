from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from badminton_data_process.analysis.biomechanics.events import ACTION_EVENT_FIELDS
from badminton_data_process.core.io import read_csv_rows, write_csv_rows
from badminton_data_process.tracking.player.pose import (
    COCO_KEYPOINT_NAMES,
    pose_keypoints_from_json,
)


BST_ADAPTER_VERSION = "bba_bst_adapter_v1"
BST_OFFICIAL_PROFILES = {
    "shuttleset_merged_seq30_balanced": {
        "model_name": "BST_AP",
        "pose_style": "JnB_bone",
        "seq_len": 30,
        "num_classes": 25,
        "weight_filename": "bst_AP_JnB_bone_merged_3.pt",
        "official_top1": 0.830,
        "official_macro_f1": 0.814,
        "official_top2": 0.952,
    },
    "shuttleset_merged_seq30_top1": {
        "model_name": "BST_AP",
        "pose_style": "JnB_bone",
        "seq_len": 30,
        "num_classes": 25,
        "weight_filename": "bst_AP_JnB_bone_merged_9.pt",
        "official_top1": 0.831,
        "official_macro_f1": 0.809,
        "official_top2": 0.952,
    },
    "shuttleset_merged_seq30_cg_ap": {
        "model_name": "BST_CG_AP",
        "pose_style": "JnB_bone",
        "seq_len": 30,
        "num_classes": 25,
        "weight_filename": "bst_CG_AP_JnB_bone_merged_2.pt",
        "official_top1": 0.831,
        "official_macro_f1": 0.800,
        "official_top2": 0.951,
    },
}
BST_BONE_PAIRS = (
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),
    (3, 5), (4, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)

_MERGED_STROKES = (
    "放小球", "擋小球", "殺球", "挑球", "長球", "平球",
    "切球", "推球", "撲球", "勾球", "發短球", "發長球",
)
_FINE_STROKES = (
    "放小球", "擋小球", "殺球", "點扣", "挑球", "防守回挑",
    "長球", "平球", "後場抽平球", "切球", "過渡切球", "推球",
    "撲球", "防守回抽", "勾球", "發短球", "發長球",
)
_ENGLISH = {
    "放小球": "net_drop",
    "擋小球": "block",
    "殺球": "smash",
    "點扣": "half_smash",
    "挑球": "lift",
    "防守回挑": "defensive_lift",
    "長球": "clear",
    "平球": "drive",
    "後場抽平球": "rear_court_drive",
    "切球": "slice",
    "過渡切球": "transitional_slice",
    "推球": "push",
    "撲球": "net_kill",
    "防守回抽": "defensive_drive",
    "勾球": "cross_court_net",
    "發短球": "short_serve",
    "發長球": "long_serve",
    "未知球種": "unknown",
}


@dataclass(frozen=True, slots=True)
class BSTInput:
    human_pose: np.ndarray
    position: np.ndarray
    shuttle: np.ndarray
    valid_length: int
    candidate_pose_coverage: float
    opponent_pose_coverage: float
    shuttle_coverage: float
    eligibility: str
    reject_reason: str


def _float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int(value: object) -> int | None:
    value_float = _float(value)
    return int(value_float) if value_float is not None else None


def _true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def bst_class_labels(num_classes: int) -> tuple[str, ...]:
    if num_classes == 25:
        return ("未知球種",) + tuple(f"Top_{name}" for name in _MERGED_STROKES) + tuple(
            f"Bottom_{name}" for name in _MERGED_STROKES
        )
    if num_classes == 35:
        return tuple(f"Top_{name}" for name in _FINE_STROKES) + tuple(
            f"Bottom_{name}" for name in _FINE_STROKES
        ) + ("未知球種",)
    raise ValueError(f"unsupported BST class count: {num_classes}")


def _same_length(
    target: int, pose: np.ndarray, position: np.ndarray, shuttle: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    source_length = len(position)
    if source_length > target:
        needs_padding = source_length % target > target // 2
        stride = source_length // target + int(needs_padding)
        pose = pose[::stride][:target]
        position = position[::stride][:target]
        shuttle = shuttle[::stride][:target]
    valid_length = len(position)
    if valid_length < target:
        padding = target - valid_length
        pose = np.pad(pose, ((0, padding), (0, 0), (0, 0), (0, 0)))
        position = np.pad(position, ((0, padding), (0, 0), (0, 0)))
        shuttle = np.pad(shuttle, ((0, padding), (0, 0)))
    return pose, position, shuttle, valid_length


def _add_bones(joints: np.ndarray) -> np.ndarray:
    bones = []
    for start, end in BST_BONE_PAIRS:
        start_joint = joints[:, :, start, :]
        end_joint = joints[:, :, end, :]
        valid = np.all(start_joint != 0.0, axis=-1) & np.all(end_joint != 0.0, axis=-1)
        bones.append(np.where(valid[..., None], end_joint - start_joint, 0.0))
    return np.concatenate((joints, np.stack(bones, axis=-2)), axis=-2)


def build_bst_input(
    event: Mapping[str, object],
    player_rows: Iterable[Mapping[str, object]],
    shuttle_rows: Iterable[Mapping[str, object]],
    *,
    frame_width: int,
    frame_height: int,
    seq_len: int = 30,
    pose_style: str = "JnB_bone",
    keypoint_threshold: float = 0.35,
) -> BSTInput:
    start = _int(event.get("window_start_frame"))
    end = _int(event.get("window_end_frame"))
    player_id = str(event.get("player_id", ""))
    if start is None or end is None or end <= start:
        start, end = 0, 0
    frame_count = end - start
    joints = np.zeros((frame_count, 2, len(COCO_KEYPOINT_NAMES), 2), dtype=np.float32)
    positions = np.zeros((frame_count, 2, 2), dtype=np.float32)
    shuttle = np.zeros((frame_count, 2), dtype=np.float32)
    role_index = {"far": 0, "near": 1}
    pose_frames = {"far": set(), "near": set()}
    for row in player_rows:
        frame = _int(row.get("frame_id"))
        role = str(row.get("player_id", ""))
        if frame is None or role not in role_index or not start <= frame < end:
            continue
        top = _float(row.get("bbox_y1"))
        bottom = _float(row.get("bbox_y2"))
        left = _float(row.get("bbox_x1"))
        right = _float(row.get("bbox_x2"))
        if None in (top, bottom, left, right) or right <= left or bottom <= top:
            continue
        assert top is not None and bottom is not None and left is not None and right is not None
        diagonal = math.hypot(right - left, bottom - top)
        center_x, center_y = (left + right) / 2.0, (top + bottom) / 2.0
        valid_count = 0
        joint_indices = {name: index for index, name in enumerate(COCO_KEYPOINT_NAMES)}
        for point in (
            pose_keypoints_from_json(str(row.get("pose_keypoints_json") or ""))
        ):
            joint_index = joint_indices.get(point.name)
            if point.confidence < keypoint_threshold or joint_index is None:
                continue
            joints[frame - start, role_index[role], joint_index] = (
                (point.x - center_x) / diagonal,
                (point.y - center_y) / diagonal,
            )
            valid_count += 1
        if valid_count >= 5:
            pose_frames[role].add(frame)
        court_x = _float(row.get("court_x"))
        court_y = _float(row.get("court_y"))
        if court_x is not None and court_y is not None and 0 <= court_x <= 6.10 and 0 <= court_y <= 13.40:
            positions[frame - start, role_index[role]] = (court_x / 6.10, court_y / 13.40)

    shuttle_frames = set()
    if frame_width > 0 and frame_height > 0:
        for row in shuttle_rows:
            frame = _int(row.get("frame_id"))
            x, y = _float(row.get("x")), _float(row.get("y"))
            if (
                frame is None
                or not start <= frame < end
                or x is None
                or y is None
                or _true(row.get("is_interpolated"))
                or not _true(row.get("visibility"))
            ):
                continue
            shuttle[frame - start] = (x / frame_width, y / frame_height)
            shuttle_frames.add(frame)

    denominator = max(1, frame_count)
    candidate_coverage = len(pose_frames.get(player_id, set())) / denominator
    opponent = "near" if player_id == "far" else "far"
    opponent_coverage = len(pose_frames.get(opponent, set())) / denominator
    shuttle_coverage = len(shuttle_frames) / denominator
    if player_id not in role_index:
        eligibility, reason = "not_eligible", "unsupported_player_role"
    elif candidate_coverage < 0.4:
        eligibility, reason = "not_eligible", "insufficient_candidate_pose_coverage"
    elif opponent_coverage < 0.2:
        eligibility, reason = "not_eligible", "insufficient_opponent_pose_coverage"
    elif shuttle_coverage < 0.15:
        eligibility, reason = "not_eligible", "insufficient_shuttle_coverage"
    else:
        eligibility, reason = "eligible", ""

    pose = joints if pose_style == "J_only" else _add_bones(joints)
    pose, positions, shuttle, valid_length = _same_length(
        seq_len, pose, positions, shuttle
    )
    return BSTInput(
        human_pose=pose,
        position=positions,
        shuttle=shuttle,
        valid_length=valid_length,
        candidate_pose_coverage=round(candidate_coverage, 4),
        opponent_pose_coverage=round(opponent_coverage, 4),
        shuttle_coverage=round(shuttle_coverage, 4),
        eligibility=eligibility,
        reject_reason=reason,
    )


class BSTRuntime:
    def __init__(
        self,
        repository: Path,
        weights: Path,
        *,
        device: str,
        model_name: str,
        pose_style: str,
        seq_len: int,
        num_classes: int,
    ) -> None:
        import torch

        source = repository / "stroke_classification"
        if not source.is_dir():
            raise FileNotFoundError("bst_repository_missing")
        if not weights.is_file():
            raise FileNotFoundError("bst_weights_missing")
        selected_device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else
            "cpu" if device == "auto" else device
        )
        if selected_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("bst_cuda_unavailable")
        sys.path.insert(0, str(source))
        try:
            module = importlib.import_module("model.bst")
        finally:
            sys.path.pop(0)
        architecture = getattr(module, model_name)
        pose_points = len(COCO_KEYPOINT_NAMES) + (
            len(BST_BONE_PAIRS) if pose_style == "JnB_bone" else 0
        )
        self.model = architecture(
            in_dim=pose_points * 2,
            n_class=num_classes,
            seq_len=seq_len,
            depth_tem=2,
            depth_inter=1,
        ).to(selected_device)
        state = torch.load(str(weights), map_location=selected_device, weights_only=True)
        try:
            self.model.load_state_dict(state)
        except RuntimeError as exc:
            raise RuntimeError(
                "bst_checkpoint_incompatible: "
                f"model={model_name}, pose_style={pose_style}, seq_len={seq_len}, "
                f"num_classes={num_classes}; {exc}"
            ) from exc
        self.model.eval()
        self.torch = torch
        self.device = selected_device
        self.labels = bst_class_labels(num_classes)
        self.model_id = (
            f"BST/{model_name}/{pose_style}/seq{seq_len}/{num_classes}/"
            f"{hashlib.sha256(weights.read_bytes()).hexdigest()[:12]}"
        )

    def predict(self, sample: BSTInput) -> tuple[list[tuple[int, float]], str]:
        torch = self.torch
        with torch.no_grad():
            pose = torch.from_numpy(sample.human_pose).unsqueeze(0).to(self.device)
            pose = pose.view(*pose.shape[:-2], -1)
            position = torch.from_numpy(sample.position).unsqueeze(0).to(self.device)
            shuttle = torch.from_numpy(sample.shuttle).unsqueeze(0).to(self.device)
            lengths = torch.tensor([sample.valid_length], device=self.device)
            probabilities = torch.softmax(
                self.model(pose, shuttle, position, lengths), dim=1
            )[0]
            values, indices = torch.topk(probabilities, k=min(2, len(self.labels)))
        return [
            (int(index), float(value))
            for index, value in zip(indices.cpu(), values.cpu(), strict=True)
        ], self.model_id


def _decode_label(label: str) -> tuple[str | None, str, str]:
    if label == "未知球種":
        return None, "unknown", "未知球种"
    side, chinese = label.split("_", 1)
    role = "far" if side == "Top" else "near"
    return role, _ENGLISH.get(chinese, chinese), chinese


def classify_events_with_bst(
    events: list[dict[str, str]],
    player_rows: list[dict[str, str]],
    shuttle_rows: list[dict[str, str]],
    runtime: BSTRuntime,
    *,
    seq_len: int,
    pose_style: str,
    keypoint_threshold: float,
    min_confidence: float,
) -> list[dict[str, object]]:
    players_by_group: dict[tuple[str, str], list[dict[str, str]]] = {}
    shuttles_by_group: dict[tuple[str, str], list[dict[str, str]]] = {}
    for rows, destination in ((player_rows, players_by_group), (shuttle_rows, shuttles_by_group)):
        for row in rows:
            destination.setdefault((row.get("video_stem", ""), row.get("rally_id", "")), []).append(row)
    dimensions: dict[str, tuple[int, int]] = {}
    output: list[dict[str, object]] = []
    for source_event in events:
        event: dict[str, object] = dict(source_event)
        video_path = str(event.get("video_path", ""))
        if video_path not in dimensions:
            try:
                import cv2

                capture = cv2.VideoCapture(video_path)
                dimensions[video_path] = (
                    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                )
                capture.release()
            except Exception:
                dimensions[video_path] = (0, 0)
        width, height = dimensions[video_path]
        group = (str(event.get("video_stem", "")), str(event.get("rally_id", "")))
        sample = build_bst_input(
            event,
            players_by_group.get(group, []),
            shuttles_by_group.get(group, []),
            frame_width=width,
            frame_height=height,
            seq_len=seq_len,
            pose_style=pose_style,
            keypoint_threshold=keypoint_threshold,
        )
        if sample.eligibility != "eligible":
            event["classification_eligibility"] = "not_eligible"
            event["classification_reject_reason"] = sample.reject_reason
            output.append(event)
            continue
        predictions, model_id = runtime.predict(sample)
        top_index, top_confidence = predictions[0]
        predicted_role, stroke_class, stroke_class_zh = _decode_label(runtime.labels[top_index])
        if predicted_role is not None and predicted_role != event.get("player_id"):
            eligibility, reason = "not_eligible", "predicted_player_mismatch"
        elif top_confidence < min_confidence or stroke_class == "unknown":
            eligibility, reason = "not_eligible", "low_confidence_or_unknown"
        else:
            eligibility, reason = "eligible", ""
        event.update(
            {
                "classification_eligibility": eligibility,
                "classification_reject_reason": reason,
                "stroke_class": stroke_class if eligibility == "eligible" else "",
                "stroke_class_zh": stroke_class_zh if eligibility == "eligible" else "",
                "top2_json": json.dumps(
                    [
                        {
                            "label": runtime.labels[index],
                            "player_id": _decode_label(runtime.labels[index])[0],
                            "stroke_class": _decode_label(runtime.labels[index])[1],
                            "stroke_class_zh": _decode_label(runtime.labels[index])[2],
                            "confidence": round(confidence, 6),
                        }
                        for index, confidence in predictions
                    ],
                    ensure_ascii=False,
                ),
                "classification_confidence": round(top_confidence, 6),
                "model_id": model_id,
            }
        )
        output.append(event)
    return output


def classify_action_events_csv(
    action_events_csv: Path,
    player_tracks_csv: Path,
    shuttle_tracks_csv: Path,
    *,
    repository: Path,
    weights: Path,
    device: str,
    model_name: str,
    pose_style: str,
    seq_len: int,
    num_classes: int,
    keypoint_threshold: float,
    min_confidence: float,
) -> dict[str, object]:
    events = read_csv_rows(action_events_csv)
    if not events:
        return {"status": "empty", "classified_events": 0}
    try:
        runtime = BSTRuntime(
            repository,
            weights,
            device=device,
            model_name=model_name,
            pose_style=pose_style,
            seq_len=seq_len,
            num_classes=num_classes,
        )
    except (ImportError, FileNotFoundError, RuntimeError, AttributeError) as exc:
        reason = str(exc) or type(exc).__name__
        for event in events:
            event["classification_eligibility"] = "not_eligible"
            event["classification_reject_reason"] = reason
        write_csv_rows(action_events_csv, ACTION_EVENT_FIELDS, events)
        return {"status": "degraded", "classified_events": 0, "reason": reason}
    classified = classify_events_with_bst(
        events,
        read_csv_rows(player_tracks_csv),
        read_csv_rows(shuttle_tracks_csv),
        runtime,
        seq_len=seq_len,
        pose_style=pose_style,
        keypoint_threshold=keypoint_threshold,
        min_confidence=min_confidence,
    )
    write_csv_rows(action_events_csv, ACTION_EVENT_FIELDS, classified)
    return {
        "status": "success",
        "classified_events": sum(
            event["classification_eligibility"] == "eligible" for event in classified
        ),
        "model_id": runtime.model_id,
    }
