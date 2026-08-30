from __future__ import annotations

import math
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - runtime dependency
    cv2 = None
    np = None
    torch = None
    nn = None


WIDTH = 512
HEIGHT = 288
VIS_THRESHOLD = 0.15


class _Conv2DBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding="same", bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class _Double2DConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv_1 = _Conv2DBlock(in_dim, out_dim)
        self.conv_2 = _Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        return self.conv_2(self.conv_1(x))


class _Triple2DConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv_1 = _Conv2DBlock(in_dim, out_dim)
        self.conv_2 = _Conv2DBlock(out_dim, out_dim)
        self.conv_3 = _Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        return self.conv_3(self.conv_2(self.conv_1(x)))


class TrackNet(nn.Module):
    """TrackNetV3 shuttle detector network (U-Net style, multi-frame input).

    Input:  N x (seq_len+1)*3 x 288 x 512  (median frame concat when bg_mode='concat')
    Output: N x seq_len x 288 x 512 sigmoid heatmaps, one per input frame.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.down_block_1 = _Double2DConv(in_dim, 64)
        self.down_block_2 = _Double2DConv(64, 128)
        self.down_block_3 = _Triple2DConv(128, 256)
        self.bottleneck = _Triple2DConv(256, 512)
        self.up_block_1 = _Triple2DConv(768, 256)
        self.up_block_2 = _Double2DConv(384, 128)
        self.up_block_3 = _Double2DConv(192, 64)
        self.predictor = nn.Conv2d(64, out_dim, (1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.down_block_1(x)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x1)
        x2 = self.down_block_2(x)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x2)
        x3 = self.down_block_3(x)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x3)
        x = self.bottleneck(x)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x3], dim=1)
        x = self.up_block_1(x)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x2], dim=1)
        x = self.up_block_2(x)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x1], dim=1)
        x = self.up_block_3(x)
        x = self.predictor(x)
        return self.sigmoid(x)


def _build_model(seq_len: int, bg_mode: str) -> TrackNet:
    if bg_mode == "concat":
        in_dim = (seq_len + 1) * 3
    elif bg_mode == "subtract":
        in_dim = seq_len
    elif bg_mode == "subtract_concat":
        in_dim = seq_len * 4
    else:
        in_dim = seq_len * 3
    return TrackNet(in_dim=in_dim, out_dim=seq_len)


def _largest_blob_center(binary: np.ndarray) -> tuple[float, float] | None:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(best)
    return x + w / 2.0, y + h / 2.0


class TrackNetDetector:
    """ShuttleDetector backed by a trained TrackNetV3 checkpoint.

    Implements `detect_sequence(frames, context)` from the ShuttleDetector
    Protocol: takes a list of BGR frames, runs the multi-frame model over a
    sliding window, and returns one detection dict per frame.
    """

    name = "tracknet"

    def __init__(
        self,
        weights_path: str | Path,
        device: str | None = None,
        vis_threshold: float = VIS_THRESHOLD,
    ) -> None:
        if torch is None or cv2 is None or np is None:
            raise RuntimeError("torch/opencv/numpy required for TrackNetDetector")
        self.weights_path = Path(weights_path)
        self.vis_threshold = vis_threshold
        requested_device = device or "auto"
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "TrackNet CUDA was requested, but torch.cuda.is_available() is false. "
                "Install a CUDA-enabled PyTorch build or select device='cpu'."
            )
        selected_device = (
            "cuda" if requested_device == "auto" and torch.cuda.is_available()
            else "cpu" if requested_device == "auto"
            else requested_device
        )
        self.device = torch.device(selected_device)
        self._model: TrackNet | None = None
        self._seq_len = 3
        self._bg_mode = ""

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if not self.weights_path.exists():
            raise FileNotFoundError(f"TrackNet weights not found: {self.weights_path}")
        ckpt = torch.load(self.weights_path, map_location="cpu", weights_only=False)
        params = ckpt.get("param_dict", {})
        seq_len = int(params.get("seq_len", 3))
        bg_mode = str(params.get("bg_mode", ""))
        model = _build_model(seq_len, bg_mode)
        model.load_state_dict(ckpt["model"])
        model.eval().to(self.device)
        self._model = model
        self._seq_len = seq_len
        self._bg_mode = bg_mode

    def _build_input(self, frames: list[np.ndarray], median_chw: np.ndarray, start: int) -> np.ndarray:
        """Build one (seq_len+1)*3 x 288 x 512 input window ending at `start`."""
        channels: list[np.ndarray] = [median_chw]
        for i in range(self._seq_len):
            idx = max(0, start - (self._seq_len - 1) + i)
            rgb = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (WIDTH, HEIGHT))
            channels.append(np.moveaxis(resized, -1, 0).astype(np.float32))
        return np.concatenate(channels, axis=0) / 255.0

    def detect_sequence(self, frames: list, context: dict | None = None) -> list[dict]:
        """Return one detection dict per frame: {frame_id, x, y, confidence, visibility}."""
        self._ensure_model()
        context = context or {}
        if not frames:
            return []
        width = frames[0].shape[1]
        height = frames[0].shape[0]

        median = context.get("median")
        if median is None:
            sample = frames[:: max(1, len(frames) // 400)][:400]
            median = np.median(np.stack(sample, axis=0), axis=0).astype(np.uint8)
        median_rgb = median[..., ::-1]  # BGR -> RGB
        median_chw = np.moveaxis(
            cv2.resize(median_rgb, (WIDTH, HEIGHT)), -1, 0
        ).astype(np.float32)

        detections: list[dict] = []
        progress_callback = context.get("progress_callback")
        if callable(progress_callback):
            progress_callback(0, len(frames))
        with torch.no_grad():
            for start in range(len(frames)):
                x = self._build_input(frames, median_chw, start)
                tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)
                heatmap = self._model(tensor).squeeze(0)[self._seq_len - 1].cpu().numpy()
                peak = float(heatmap.max())
                if peak < self.vis_threshold:
                    detections.append(
                        {"frame_id": start, "x": None, "y": None, "confidence": 0.0, "visibility": 0}
                    )
                else:
                    binary = (heatmap > self.vis_threshold).astype(np.uint8)
                    center = _largest_blob_center(binary)
                    if center is None:
                        detections.append(
                            {"frame_id": start, "x": None, "y": None, "confidence": 0.0, "visibility": 0}
                        )
                    else:
                        cx, cy = center
                        detections.append(
                            {
                                "frame_id": start,
                                "x": round(cx * width / WIDTH, 2),
                                "y": round(cy * height / HEIGHT, 2),
                                "confidence": round(peak, 3),
                                "visibility": 1,
                            }
                        )
                if callable(progress_callback) and (
                    start + 1 == len(frames) or (start + 1) % 10 == 0
                ):
                    progress_callback(start + 1, len(frames))
        return detections
