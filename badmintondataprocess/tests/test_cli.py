from __future__ import annotations

import pytest

from badminton_data_process.cli import main


def test_cli_help_returns_success() -> None:
    assert main(["--help"]) == 0


def test_render_demo_help_returns_success() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["render", "demo", "--help"])
    assert exc_info.value.code == 0


def test_full_analysis_help_returns_success() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["analyze", "--help"])
    assert exc_info.value.code == 0
