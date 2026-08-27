from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

from .io import ensure_dir, read_json, write_json
from .schemas import STAGE_ORDER, PipelineStageReport, StageName, StageStatus


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_run_id(prefix: str = "run") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}"


def _report_from_dict(data: dict[str, Any]) -> PipelineStageReport:
    return PipelineStageReport(
        name=StageName(data["name"]),
        status=StageStatus(data["status"]),
        started_at=data.get("started_at", ""),
        finished_at=data.get("finished_at", ""),
        duration_seconds=data.get("duration_seconds", 0.0),
        inputs=list(data.get("inputs", [])),
        outputs=list(data.get("outputs", [])),
        parameters=dict(data.get("parameters", {})),
        message=data.get("message", ""),
    )


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

    def resume(self) -> set[StageName]:
        manifest_path = self.run_dir / "manifest.json"
        if not manifest_path.exists():
            return set()
        payload = read_json(manifest_path)
        completed: list[PipelineStageReport] = []
        for stage in payload.get("stages", []):
            report = _report_from_dict(stage)
            if report.status != StageStatus.SUCCESS:
                break
            completed.append(report)
        self.reports = completed
        return {report.name for report in completed}

    def add_report(self, report: PipelineStageReport) -> None:
        self._validate_stage_order(report.name)
        self.reports.append(report)
        self.write_manifest()

    def _validate_stage_order(self, name: StageName) -> None:
        # Stages may be skipped (e.g. visualization via --skip-visualize), so
        # the recorded sequence is a subsequence of STAGE_ORDER rather than the
        # full list. Enforce forward progression and forbid duplicates; anything
        # else in order is fine.
        if not self.reports:
            return
        last = self.reports[-1]
        if STAGE_ORDER.index(name) <= STAGE_ORDER.index(last.name):
            raise ValueError(
                f"Stage out of order: expected a stage after {last.name.value!r}, "
                f"got {name.value!r}"
            )

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
    name: StageName,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    parameters: dict[str, Any] | None = None,
) -> Iterator[None]:
    started_at = utc_now_iso()
    start = perf_counter()
    status = StageStatus.SUCCESS
    message = ""
    try:
        yield
    except Exception as exc:
        status = StageStatus.FAILED
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
