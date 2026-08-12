from __future__ import annotations

import argparse
from pathlib import Path


def preprocess_video(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'[TODO] preprocess video: {input_path}')
    print(f'[TODO] save intermediate outputs to: {output_dir}')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Preprocess a badminton broadcast video.')
    parser.add_argument('input', type=Path, help='Path to the input video file.')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/preprocess'))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    preprocess_video(args.input, args.output_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
