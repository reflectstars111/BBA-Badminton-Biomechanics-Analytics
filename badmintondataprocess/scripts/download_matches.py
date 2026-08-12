from __future__ import annotations

import argparse
from datetime import datetime, UTC
from pathlib import Path
from typing import Any
import re

from common import ensure_dir, read_csv_rows, write_csv_rows

import imageio_ffmpeg
import yt_dlp


STATUS_FIELDS = [
    'match_id',
    'url',
    'status',
    'is_playlist',
    'output_path',
    'downloaded_items',
    'message',
    'updated_at_utc',
]


def is_playlist_row(row: dict[str, str]) -> bool:
    return row.get('round', '').strip().lower() == 'playlist'


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec='seconds')


def clean_message(message: str) -> str:
    return re.sub(r'\x1b\[[0-9;]*m', '', message).strip()


def select_cookie_file(
    url: str,
    default_cookies: Path | None,
    youtube_cookies: Path | None,
    olympics_cookies: Path | None,
) -> Path | None:
    lowered = url.lower()
    if 'youtube.com' in lowered or 'youtu.be' in lowered:
        return youtube_cookies or default_cookies
    if 'olympics.com' in lowered:
        return olympics_cookies or default_cookies
    return default_cookies


def build_output_template(output_dir: Path, match_id: str, is_playlist: bool) -> Path:
    if is_playlist:
        playlist_dir = output_dir / match_id
        ensure_dir(playlist_dir)
        return playlist_dir / '%(playlist_index|NA)s_%(title).180B_[%(id)s].%(ext)s'
    return output_dir / f'{match_id}.%(ext)s'


def count_download_items(info: dict[str, Any]) -> int:
    entries = info.get('entries')
    if entries is None:
        return 1
    return len([item for item in entries if item])


def download_one(
    row: dict[str, str],
    output_dir: Path,
    archive_file: Path,
    cookies: Path | None,
    youtube_cookies: Path | None,
    olympics_cookies: Path | None,
    cookies_from_browser: str | None,
) -> dict[str, object]:
    match_id = row['match_id']
    url = row['url']
    playlist_mode = is_playlist_row(row)
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    outtmpl = build_output_template(output_dir, match_id, playlist_mode)

    ydl_opts: dict[str, Any] = {
        'outtmpl': str(outtmpl),
        'ffmpeg_location': ffmpeg_path,
        'restrictfilenames': True,
        'merge_output_format': 'mp4',
        'download_archive': str(archive_file),
        'ignoreerrors': playlist_mode,
        'noplaylist': not playlist_mode,
        'quiet': False,
        'no_warnings': False,
        'format': 'bv*+ba/b',
        'windowsfilenames': True,
    }
    selected_cookies = select_cookie_file(
        url=url,
        default_cookies=cookies,
        youtube_cookies=youtube_cookies,
        olympics_cookies=olympics_cookies,
    )
    if selected_cookies is not None:
        ydl_opts['cookiefile'] = str(selected_cookies)
    if cookies_from_browser:
        ydl_opts['cookiesfrombrowser'] = (cookies_from_browser,)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
        if info is None:
            raise RuntimeError('No media info returned by yt-dlp')

        downloaded_items = count_download_items(info)
        output_path = str(outtmpl.parent if playlist_mode else output_dir / f'{match_id}.mp4')
        return {
            'match_id': match_id,
            'url': url,
            'status': 'success',
            'is_playlist': playlist_mode,
            'output_path': output_path,
            'downloaded_items': downloaded_items,
            'message': 'download completed',
            'updated_at_utc': now_utc(),
        }
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        output_path = str(outtmpl.parent if playlist_mode else output_dir / f'{match_id}.mp4')
        return {
            'match_id': match_id,
            'url': url,
            'status': 'failed',
            'is_playlist': playlist_mode,
            'output_path': output_path,
            'downloaded_items': 0,
            'message': clean_message(str(exc)),
            'updated_at_utc': now_utc(),
        }


def filter_rows(rows: list[dict[str, str]], match_ids: set[str]) -> list[dict[str, str]]:
    if not match_ids:
        return rows
    return [row for row in rows if row['match_id'] in match_ids]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Download match videos from metadata/matches.csv')
    parser.add_argument('--csv', type=Path, default=Path('metadata/matches.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('raw_videos'))
    parser.add_argument('--status-csv', type=Path, default=Path('metadata/download_status.csv'))
    parser.add_argument('--archive-file', type=Path, default=Path('metadata/yt_dlp_archive.txt'))
    parser.add_argument(
        '--cookies',
        type=Path,
        default=None,
        help='Path to a Netscape-format cookies.txt file for authenticated downloads.',
    )
    parser.add_argument(
        '--youtube-cookies',
        type=Path,
        default=None,
        help='Cookie file used for YouTube URLs.',
    )
    parser.add_argument(
        '--olympics-cookies',
        type=Path,
        default=None,
        help='Cookie file used for Olympics URLs.',
    )
    parser.add_argument(
        '--cookies-from-browser',
        type=str,
        default=None,
        help='Browser name for yt-dlp cookie extraction, e.g. chrome, chromium, firefox, edge.',
    )
    parser.add_argument(
        '--match-id',
        action='append',
        default=[],
        help='Only download selected match_id values. Can be repeated.',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=0,
        help='Only process the first N rows after filtering. 0 means no limit.',
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = read_csv_rows(args.csv)
    if not rows:
        print(f'No rows found in: {args.csv}')
        return 1

    ensure_dir(args.output_dir)
    ensure_dir(args.archive_file.parent)
    selected_rows = filter_rows(rows, set(args.match_id))
    if args.limit > 0:
        selected_rows = selected_rows[:args.limit]

    if not selected_rows:
        print('No rows selected for download.')
        return 1

    results: list[dict[str, object]] = []
    for row in selected_rows:
        print(f"Downloading: {row['match_id']} -> {row['url']}")
        result = download_one(
            row=row,
            output_dir=args.output_dir,
            archive_file=args.archive_file,
            cookies=args.cookies,
            youtube_cookies=args.youtube_cookies,
            olympics_cookies=args.olympics_cookies,
            cookies_from_browser=args.cookies_from_browser,
        )
        print(f"Result: {result['status']} ({result['message']})")
        results.append(result)

    write_csv_rows(args.status_csv, STATUS_FIELDS, results)
    print(f'Status written to: {args.status_csv}')
    success_count = len([row for row in results if row['status'] == 'success'])
    print(f'Success: {success_count}/{len(results)}')
    return 0 if success_count == len(results) else 2


if __name__ == '__main__':
    raise SystemExit(main())
