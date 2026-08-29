from __future__ import annotations

import pytest

from badminton_data_process.tracking.shuttle import tracknet


def test_tracknet_explicit_cuda_does_not_silently_fallback(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(tracknet.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        tracknet.TrackNetDetector(tmp_path / "weights.pt", device="cuda")
