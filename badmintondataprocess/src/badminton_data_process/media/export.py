from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from imageio_ffmpeg import get_ffmpeg_exe


@dataclass(frozen=True, slots=True)
class ExportResult:
    output_path: str
    video_codec: str
    pixel_format: str
    audio_codec: str
    audio_source: str
    faststart: bool
    size_bytes: int

    def artifact_details(self) -> dict[str, object]:
        return asdict(self)


def export_browser_video(
    input_video: Path,
    output_video: Path,
    *,
    preserve_audio: bool = False,
    audio_source: Path | None = None,
    ffmpeg_executable: str | Path | None = None,
) -> ExportResult:
    """Create an H.264/yuv420p/faststart MP4 as a derived Artifact.

    When ``audio_source`` is supplied its first audio stream is used. Otherwise
    ``preserve_audio`` optionally maps audio from ``input_video``. A temporary
    sibling is atomically promoted only after FFmpeg produced a non-empty file.
    """

    source = Path(input_video).resolve()
    target = Path(output_video).resolve()
    if not source.is_file() or source.stat().st_size <= 0:
        raise RuntimeError(f"media export input is missing or empty: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.{uuid4().hex}.tmp.mp4")
    executable = str(ffmpeg_executable or get_ffmpeg_exe())

    command = [executable, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source)]
    selected_audio = "none"
    if audio_source is not None:
        resolved_audio = Path(audio_source).resolve()
        if not resolved_audio.is_file():
            raise RuntimeError(f"audio source is missing: {resolved_audio}")
        command.extend(["-i", str(resolved_audio), "-map", "0:v:0", "-map", "1:a:0?"])
        selected_audio = str(resolved_audio)
    elif preserve_audio:
        command.extend(["-map", "0:v:0", "-map", "0:a:0?"])
        selected_audio = str(source)
    else:
        command.extend(["-map", "0:v:0", "-an"])

    command.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    if selected_audio != "none":
        command.extend(["-c:a", "aac", "-shortest"])
    command.append(str(temporary))

    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "unknown FFmpeg error").strip()
            raise RuntimeError(f"browser video export failed: {message}")
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RuntimeError("browser video export produced an empty file")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()

    return ExportResult(
        output_path=str(target),
        video_codec="h264",
        pixel_format="yuv420p",
        audio_codec="aac" if selected_audio != "none" else "none",
        audio_source=selected_audio,
        faststart=True,
        size_bytes=target.stat().st_size,
    )
