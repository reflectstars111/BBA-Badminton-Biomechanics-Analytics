from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

from .io import ensure_dir, write_json
from .schemas import PipelineStageReport


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_run_id(prefix: str = "run") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}"


@dataclass(slots=True)
class RunContext:
    root: Path
    run_id: str
    config: dict[str, Any]
    reports: list[PipelineStageReport] = field(default_factory=list)

    @property
    def run_dir(self) -> Path:
        return self.root / "runs" / self.run_id

    def ensure(self) -> None:
        ensure_dir(self.run_dir)

    def add_report(self, report: PipelineStageReport) -> None:
        self.reports.append(report)
        self.write_manifest()

    def write_manifest(self) -> None:
        self.ensure()
        payload = {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "config": self.config,
            "stages": [asdict(report) for report in self.reports],
        }
        write_json(self.run_dir / "manifest.json", payload)
        write_json(self.run_dir / "report.json", payload)


@contextmanager
def stage_report(
    context: RunContext,
    name: str,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    parameters: dict[str, Any] | None = None,
) -> Iterator[None]:
    started_at = utc_now_iso()
    start = perf_counter()
    status = "success"
    message = ""
    try:
        yield
    except Exception as exc:
        status = "failed"
        message = str(exc)
        raise
    finally:
        finished_at = utc_now_iso()
        context.add_report(
            PipelineStageReport(
                name=name,
                status=status,
                started_at=started_at,
                finished_at=finished_at,
                duration_seconds=round(perf_counter() - start, 3),
                inputs=inputs or [],
                outputs=outputs or [],
                parameters=parameters or {},
                message=message,
            )
        )
