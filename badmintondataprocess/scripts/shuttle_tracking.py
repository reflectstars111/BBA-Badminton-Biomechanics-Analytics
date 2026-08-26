from __future__ import annotations

"""Thin wrapper over the shuttle tracking implementation in src/.

The tracking logic now lives in badminton_data_process.tracking.shuttle.tracking;
this script is kept so `python scripts/shuttle_tracking.py ...` and legacy
module loading (load_legacy_module) keep working unchanged.
"""

from badminton_data_process.tracking.shuttle.tracking import *  # noqa: F401,F403


if __name__ == '__main__':
    raise SystemExit(main())
