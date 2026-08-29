from __future__ import annotations

import sys
import types

import pytest
import numpy as np

from badminton_data_process.tracking.player.pose import (
    COCO_KEYPOINT_NAMES,
    RtmposeImplementation,
    build_pose_observation,
    pose_keypoints_from_json,
    rtmpose_outputs_to_candidates,
    skeleton_segments,
)


def _coordinates() -> list[tuple[float, float]]:
    return [(float(index * 10), float(index * 5)) for index in range(17)]


def test_pose_observation_round_trips_named_coco_keypoints() -> None:
    pose = build_pose_observation(
        _coordinates(),
        [0.9] * 17,
        model_name="yolo11n-pose.pt",
        detection_confidence=0.8,
    )

    restored = pose_keypoints_from_json(pose.to_json())
    assert tuple(point.name for point in restored) == COCO_KEYPOINT_NAMES
    assert restored[15].x == 150.0
    assert restored[15].confidence == pytest.approx(0.9)


def test_skeleton_omits_edges_with_low_confidence_endpoint() -> None:
    scores = [0.9] * 17
    scores[7] = 0.1  # left elbow
    pose = build_pose_observation(
        _coordinates(), scores, model_name="pose.pt", detection_confidence=0.8
    )

    segments = skeleton_segments(pose.keypoints, threshold=0.35)
    names = {(start.name, end.name) for start, end in segments}
    assert ("left_shoulder", "left_elbow") not in names
    assert ("left_elbow", "left_wrist") not in names
    assert ("right_shoulder", "right_elbow") in names


def test_pose_requires_exact_coco_17_shape() -> None:
    with pytest.raises(ValueError, match="requires 17"):
        build_pose_observation(
            [(0.0, 0.0)] * 16,
            [0.9] * 16,
            model_name="pose.pt",
            detection_confidence=0.8,
        )


def test_rtmpose_output_becomes_role_candidate_with_pose() -> None:
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    court_mask = np.full((300, 400), 255, dtype=np.uint8)
    coordinates = np.asarray([_coordinates()], dtype=np.float32)
    coordinates[0, :, 0] += 100
    coordinates[0, :, 1] += 100
    scores = np.full((1, 17), 0.9, dtype=np.float32)

    candidates = rtmpose_outputs_to_candidates(
        frame,
        court_mask,
        coordinates,
        scores,
        model_name="rtmlib:rtmpose:balanced",
        keypoint_threshold=0.35,
        min_valid_keypoints=5,
    )

    assert len(candidates) == 1
    assert candidates[0]["pose"].model_name == "rtmlib:rtmpose:balanced"
    assert candidates[0]["score"] == pytest.approx(0.9)


def test_rtmpose_rejects_insufficient_valid_keypoints() -> None:
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    court_mask = np.full((300, 400), 255, dtype=np.uint8)
    candidates = rtmpose_outputs_to_candidates(
        frame,
        court_mask,
        np.asarray([_coordinates()], dtype=np.float32),
        np.full((1, 17), 0.1, dtype=np.float32),
        model_name="rtmpose",
        keypoint_threshold=0.35,
        min_valid_keypoints=5,
    )
    assert candidates == []


def test_rtmpose_explicit_cuda_rejects_cpu_only_onnxruntime(monkeypatch) -> None:
    fake_rtmlib = types.ModuleType("rtmlib")
    fake_rtmlib.Body = lambda **_kwargs: object()
    fake_onnxruntime = types.ModuleType("onnxruntime")
    fake_onnxruntime.get_available_providers = lambda: ["CPUExecutionProvider"]
    monkeypatch.setitem(sys.modules, "rtmlib", fake_rtmlib)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime)

    implementation = RtmposeImplementation(backend="onnxruntime", device="cuda")

    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        implementation._load()
