from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import cv2
except ImportError:  # pragma: no cover - runtime dependency
    cv2 = None


def probe_video(input_path: Path, sample_every: int = 30) -> dict[str, object]:
    if cv2 is None:
        raise RuntimeError('OpenCV is required. Install it with: pip install opencv-python')
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open video: {input_path}')

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Verify readability by sampling a few frames.
    sampled = 0
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % sample_every == 0:
            sampled += 1
        frame_index += 1
    capture.release()

    # Prefer a real frame count; fall back to CAP_PROP if unreliable.
    read_frames = frame_index
    frame_count = total_frames if total_frames and abs(total_frames - read_frames) <= max(1, read_frames * 0.05) else read_frames
    duration_seconds = frame_count / fps if fps else 0.0

    return {
        'video_path': str(input_path),
        'video_stem': input_path.stem,
        'width': width,
        'height': height,
        'fps': round(fps, 3),
        'frame_count': frame_count,
        'duration_seconds': round(duration_seconds, 3),
        'readable_frames': read_frames,
        'sampled_frames': sampled,
        'ok': bool(width and height and fps and frame_count),
    }


def preprocess_video(input_path: Path, output_dir: Path, sample_every: int = 30) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = probe_video(input_path, sample_every=sample_every)
    report_path = output_dir / f'{input_path.stem}_metadata.json'
    report_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    if not metadata['ok']:
        raise RuntimeError(f'Video failed readability probe: {input_path}')
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Preprocess a badminton broadcast video: probe and validate readability.')
    parser.add_argument('input', type=Path, help='Path to the input video file.')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/preprocess'))
    parser.add_argument('--sample-every', type=int, default=30, help='Frame interval used to sample readability.')
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metadata = preprocess_video(args.input, args.output_dir, sample_every=args.sample_every)
    print(f'Probed {metadata["video_path"]}')
    print(
        f'  {metadata["width"]}x{metadata["height"]} @ {metadata["fps"]}fps, '
        f'{metadata["frame_count"]} frames ({metadata["duration_seconds"]}s)'
    )
    print(f'  readable: {metadata["readable_frames"]} frames read')
    report_path = args.output_dir / (metadata['video_stem'] + '_metadata.json')
    print(f'  metadata saved to: {report_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
