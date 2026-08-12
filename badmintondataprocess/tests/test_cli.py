from __future__ import annotations

from badminton_data_process.cli import main


def test_cli_help_returns_success() -> None:
    assert main(["--help"]) == 0

