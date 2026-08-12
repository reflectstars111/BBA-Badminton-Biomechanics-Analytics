from __future__ import annotations

import argparse
from pathlib import Path


def visualize(input_video: Path, output_video: Path) -> None:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    print(f'[TODO] render visualization for: {input_video}')
    print(f'[TODO] save visualization to: {output_video}')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Visualize tracking and tactical analysis outputs.')
    parser.add_argument('input', type=Path, help='Input rally video.')
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('outputs/trajectory_videos/visualization.mp4'),
        help='Output visualization video path.',
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    visualize(args.input, args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
