from __future__ import annotations

import argparse
import os
from pathlib import Path

from badminton_data_process.core.config import load_config
from badminton_data_process.core.config_schema import parse_config
from badminton_data_process.core.paths import ProjectPaths, RunLayout
from badminton_data_process.pipeline.run import run_pipeline
from common import read_csv_rows, write_csv_rows

# This script lives in scripts/, so the project root is one level up. A
# cwd-based lookup is fragile: 'pipeline batch' may be launched from
# anywhere, and the pipeline stages must resolve configs/ and runs/ under
# the project root regardless of the caller's working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BATCH_FIELDNAMES = ['match_id', 'video', 'run_dir', 'status', 'message']


def batch_run_matches(
    matches_csv: Path,
    runs_dir: Path,
    stop_after: str | None = None,
    config_path: Path | None = None,
    skip_visualize: bool = False,
    skip_demo: bool = False,
    force: bool = False,
    max_matches: int | None = None,
) -> list[dict[str, str]]:
    """Run the pipeline over every match in matches.csv that has a local video.

    Matches whose url is a remote link (download candidates) are skipped.
    run_pipeline resumes from an existing manifest, so re-running the batch
    only finishes stages that are still missing.
    """
    root = PROJECT_ROOT
    rows = read_csv_rows(matches_csv)
    results: list[dict[str, str]] = []
    processed = 0
    for row in rows:
        video = row.get('url', '')
        if not video or not os.path.exists(video):
            continue
        if max_matches is not None and processed >= max_matches:
            break
        match_id = row['match_id']
        run_id = f"batch_{match_id.lower()}"
        try:
            run_dir = run_pipeline(
                Path(video),
                run_id=run_id,
                config_path=config_path,
                root=root,
                stop_after=stop_after,
                skip_visualize=skip_visualize,
                skip_demo=skip_demo,
                force=force,
                runs_dir=runs_dir,
            )
            results.append(
                {
                    'match_id': match_id,
                    'video': str(video),
                    'run_dir': str(run_dir),
                    'status': 'done',
                    'message': f"stop_after={stop_after or 'full'}",
                }
            )
        except Exception as exc:  # pragma: no cover - runtime dependent
            results.append(
                {
                    'match_id': match_id,
                    'video': str(video),
                    'run_dir': str(RunLayout.create(root, run_id, runs_dir).run_dir),
                    'status': 'failed',
                    'message': str(exc)[:200],
                }
            )
        processed += 1
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Run the pipeline over every locally available match in metadata/matches.csv. '
            'Remote download-candidate rows are skipped.'
        )
    )
    parser.add_argument('--matches-csv', type=Path, default=None)
    parser.add_argument('--runs-dir', type=Path, default=None)
    parser.add_argument('--config', type=Path, default=None)
    parser.add_argument(
        '--stop-after',
        choices=['main_view', 'rally', 'calibrate', 'tracking'],
        default=None,
        help='Stop after this stage; omit to run the full pipeline.',
    )
    parser.add_argument('--skip-visualize', action='store_true')
    parser.add_argument('--skip-demo', action='store_true')
    parser.add_argument('--force', action='store_true', help='Ignore existing manifests and re-run all stages.')
    parser.add_argument('--max-matches', type=int, default=None)
    parser.add_argument('--summary-csv', type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = PROJECT_ROOT
    config = load_config(args.config, root=root)
    cfg = parse_config(config)
    project_paths = ProjectPaths.from_config(config, root=root)
    matches_csv = args.matches_csv or project_paths.metadata / 'matches.csv'
    runs_dir = RunLayout.create(
        root,
        '_batch_layout',
        args.runs_dir if args.runs_dir is not None else cfg.data.runs_dir,
    ).runs_dir
    results = batch_run_matches(
        matches_csv=matches_csv,
        runs_dir=runs_dir,
        stop_after=args.stop_after,
        config_path=args.config,
        skip_visualize=args.skip_visualize,
        skip_demo=args.skip_demo,
        force=args.force,
        max_matches=args.max_matches,
    )
    summary_csv = args.summary_csv or runs_dir / 'batch_summary.csv'
    if results:
        write_csv_rows(summary_csv, BATCH_FIELDNAMES, results)
    for row in results:
        print(f"{row['match_id']}: {row['status']} -> {row['run_dir']}")
    print(
        f"Batch processed {len(results)} locally available match(es); "
        f"summary: {summary_csv}"
    )
    return 0 if not any(row['status'] == 'failed' for row in results) else 2


if __name__ == '__main__':
    raise SystemExit(main())
