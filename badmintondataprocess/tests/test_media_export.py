from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import badminton_data_process.media.export as export_module


def test_browser_export_uses_explicit_codec_contract(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"raw-video")
    target = tmp_path / "browser.mp4"
    captured: list[str] = []

    def fake_run(command, **_kwargs):
        captured.extend(command)
        Path(command[-1]).write_bytes(b"encoded-video")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(export_module.subprocess, "run", fake_run)
    result = export_module.export_browser_video(
        source,
        target,
        preserve_audio=True,
        ffmpeg_executable="bundled-ffmpeg",
    )

    assert target.read_bytes() == b"encoded-video"
    assert ["-c:v", "libx264"] == captured[captured.index("-c:v") : captured.index("-c:v") + 2]
    assert "yuv420p" in captured
    assert "+faststart" in captured
    assert "0:a:0?" in captured
    assert result.audio_codec == "aac"


def test_browser_export_without_audio_is_explicit(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"raw-video")
    target = tmp_path / "browser.mp4"

    def fake_run(command, **_kwargs):
        Path(command[-1]).write_bytes(b"encoded-video")
        assert "-an" in command
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(export_module.subprocess, "run", fake_run)
    result = export_module.export_browser_video(source, target, ffmpeg_executable="ffmpeg")
    assert result.audio_codec == "none"
