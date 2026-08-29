from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

from .io import ensure_dir, read_json, write_json
from .paths import RunLayout
from .schemas import (
    STAGE_ORDER,
    ArtifactKind,
    ArtifactReport,
    ArtifactStatus,
    PipelineStageReport,
    StageName,
    StageResult,
    StageStatus,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_run_id(prefix: str = "run") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}"


def _report_from_dict(data: dict[str, Any]) -> PipelineStageReport:
    artifacts = [
        ArtifactReport(
            name=str(item.get("name", "")),
            path=str(item.get("path", "")),
            kind=ArtifactKind(item.get("kind", ArtifactKind.FILE.value)),
            status=ArtifactStatus(item.get("status", ArtifactStatus.INVALID.value)),
            message=str(item.get("message", "")),
            details=dict(item.get("details", {})),
        )
        for item in data.get("artifacts", [])
    ]
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
        exit_code=(int(data["exit_code"]) if data.get("exit_code") is not None else None),
        artifacts=artifacts,
    )


class StageExecutionError(RuntimeError):
    """Raised when a stage reports failure without raising its own exception."""


@dataclass(slots=True)
class StageExecution:
    """Collect a Stage Result while preserving compatibility with integer-return stages."""

    name: StageName
    result: StageResult = field(
        default_factory=lambda: StageResult(status=StageStatus.SUCCESS)
    )

    def complete(self, message: str = "") -> StageResult:
        """Record a successful structured result after stage-specific handling."""
        self.result = StageResult(
            status=StageStatus.SUCCESS,
            message=message,
            artifacts=self.result.artifacts,
        )
        return self.result

    def reject(self, message: str) -> None:
        """Record a quality rejection and stop downstream execution."""
        self.result.status = StageStatus.REJECTED
        self.result.message = message
        raise StageExecutionError(message)

    def require_artifact(self, artifact: ArtifactReport) -> ArtifactReport:
        """Attach an artifact check and stop the stage when it is unusable."""
        self.result.artifacts.append(artifact)
        if artifact.status == ArtifactStatus.VALID:
            return artifact

        status = (
            StageStatus.EMPTY
            if artifact.status == ArtifactStatus.EMPTY
            else StageStatus.FAILED
        )
        message = (
            f"artifact {artifact.name!r} is {artifact.status.value}: "
            f"{artifact.message or artifact.path}"
        )
        self.result.status = status
        self.result.message = message
        raise StageExecutionError(message)

    def accept_legacy(self, return_code: int | None, operation: str | None = None) -> StageResult:
        """Convert a legacy ``None``/integer return into a Stage Result.

        ``None`` and zero are successful. Any non-zero integer is recorded and
        raised immediately so the surrounding stage report cannot be written as
        successful. Unexpected return types are failures as well.
        """
        label = operation or self.name.value
        artifacts = self.result.artifacts
        if return_code is None:
            self.result = StageResult(status=StageStatus.SUCCESS, artifacts=artifacts)
            return self.result
        if isinstance(return_code, bool) or not isinstance(return_code, int):
            self.result = StageResult(
                status=StageStatus.FAILED,
                message=(
                    f"{label} returned unsupported result type "
                    f"{type(return_code).__name__}; expected int or None"
                ),
                artifacts=artifacts,
            )
            raise StageExecutionError(self.result.message)
        if return_code == 0:
            self.result = StageResult(
                status=StageStatus.SUCCESS,
                exit_code=0,
                artifacts=artifacts,
            )
            return self.result

        self.result = StageResult(
            status=StageStatus.FAILED,
            exit_code=return_code,
            message=f"{label} returned non-zero exit code {return_code}",
            artifacts=artifacts,
        )
        raise StageExecutionError(self.result.message)


@dataclass(slots=True)
class RunContext:
    root: Path
    run_id: str
    config: dict[str, Any]
    layout: RunLayout | None = None
    reports: list[PipelineStageReport] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.layout is None:
            self.layout = RunLayout.create(self.root, self.run_id)
        elif self.layout.run_id != self.run_id:
            raise ValueError(
                f"RunContext run_id {self.run_id!r} does not match layout "
                f"run_id {self.layout.run_id!r}"
            )

    @property
    def run_dir(self) -> Path:
        assert self.layout is not None
        return self.layout.run_dir

    def ensure(self) -> None:
        ensure_dir(self.run_dir)

    def resume(self) -> set[StageName]:
        assert self.layout is not None
        manifest_path = self.layout.manifest_json
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
        assert self.layout is not None
        self.ensure()
        payload = {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "config": self.config,
            "stages": [asdict(report) for report in self.reports],
        }
        write_json(self.layout.manifest_json, payload)
        write_json(self.layout.report_json, payload)


@contextmanager
def stage_report(
    context: RunContext,
    name: StageName,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    parameters: dict[str, Any] | None = None,
) -> Iterator[StageExecution]:
    started_at = utc_now_iso()
    start = perf_counter()
    execution = StageExecution(name=name)
    try:
        yield execution
    except BaseException as exc:
        message = str(exc) or type(exc).__name__
        if execution.result.status == StageStatus.SUCCESS:
            exit_code = None
            if isinstance(exc, SystemExit) and isinstance(exc.code, int):
                exit_code = exc.code
            execution.result = StageResult(
                status=StageStatus.FAILED,
                exit_code=exit_code,
                message=message,
                artifacts=execution.result.artifacts,
            )
        else:
            execution.result.message = message
        raise
    finally:
        finished_at = utc_now_iso()
        context.add_report(
            PipelineStageReport(
                name=name,
                status=execution.result.status,
                started_at=started_at,
                finished_at=finished_at,
                duration_seconds=round(perf_counter() - start, 3),
                inputs=inputs or [],
                outputs=outputs or [],
                parameters=parameters or {},
                message=execution.result.message,
                exit_code=execution.result.exit_code,
                artifacts=execution.result.artifacts,
            )
        )
