from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_MARKERS = ("pyproject.toml", "configs", "scripts", "md_summary")


def discover_project_root(start: str | Path | None = None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "configs").exists() and (candidate / "scripts").exists():
            return candidate
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    return current


@dataclass(slots=True)
class ProjectPaths:
    root: Path
    raw_videos: Path
    rallies: Path
    metadata: Path
    annotations: Path
    outputs: Path
    runs: Path

    @classmethod
    def from_config(cls, config: dict, root: str | Path | None = None) -> "ProjectPaths":
        project_root = Path(root) if root is not None else discover_project_root()
        data = config.get("data", {})
        return cls(
            root=project_root,
            raw_videos=project_root / data.get("raw_videos_dir", "raw_videos"),
            rallies=project_root / data.get("rallies_dir", "rallies"),
            metadata=project_root / data.get("metadata_dir", "metadata"),
            annotations=project_root / data.get("annotations_dir", "annotations"),
            outputs=project_root / data.get("outputs_dir", "outputs"),
            runs=project_root / data.get("runs_dir", "runs"),
        )

    def resolve(self, path: str | Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else self.root / path


@dataclass(frozen=True, slots=True)
class RunLayout:
    """Single Interface for every path owned by one pipeline run."""

    project_root: Path
    runs_dir: Path
    run_id: str

    @classmethod
    def create(
        cls,
        project_root: str | Path,
        run_id: str,
        runs_dir: str | Path = "runs",
    ) -> "RunLayout":
        root = Path(project_root).resolve()
        run_name = str(run_id).strip()
        if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
            raise ValueError(f"run_id must be a non-empty path segment, got {run_id!r}")
        configured_runs = Path(runs_dir)
        resolved_runs = (
            configured_runs.resolve()
            if configured_runs.is_absolute()
            else (root / configured_runs).resolve()
        )
        return cls(project_root=root, runs_dir=resolved_runs, run_id=run_name)

    @property
    def run_dir(self) -> Path:
        return self.runs_dir / self.run_id

    @property
    def manifest_json(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def report_json(self) -> Path:
        return self.run_dir / "report.json"

    @property
    def annotations_dir(self) -> Path:
        return self.run_dir / "annotations"

    @property
    def outputs_dir(self) -> Path:
        return self.run_dir / "outputs"

    @property
    def main_view_dir(self) -> Path:
        return self.run_dir / "main_view"

    @property
    def rallies_dir(self) -> Path:
        return self.run_dir / "rallies"

    @property
    def court_calibration_dir(self) -> Path:
        return self.annotations_dir / "court_calibration"

    @property
    def court_calibration_debug_dir(self) -> Path:
        return self.outputs_dir / "court_calibration_debug"

    @property
    def player_tracking_debug_dir(self) -> Path:
        return self.outputs_dir / "player_tracking_debug"

    @property
    def shuttle_tracking_debug_dir(self) -> Path:
        return self.outputs_dir / "shuttle_tracking_debug"

    @property
    def tracking_charts_dir(self) -> Path:
        return self.outputs_dir / "tracking_charts"

    @property
    def tactics_dir(self) -> Path:
        return self.outputs_dir / "tactics"

    @property
    def demo_dir(self) -> Path:
        return self.outputs_dir / "demo"

    @property
    def review_dir(self) -> Path:
        return self.run_dir / "review"

    @property
    def main_view_frame_scores_csv(self) -> Path:
        return self.main_view_dir / "main_view_frame_scores.csv"

    @property
    def main_view_segments_csv(self) -> Path:
        return self.main_view_dir / "main_view_segments.csv"

    @property
    def main_view_quality_csv(self) -> Path:
        return self.main_view_dir / "main_view_quality.csv"

    @property
    def main_view_timeline_json(self) -> Path:
        return self.main_view_dir / "main_view_timeline.json"

    @property
    def rallies_csv(self) -> Path:
        return self.run_dir / "rallies.csv"

    @property
    def rally_decisions_csv(self) -> Path:
        return self.run_dir / "rally_decisions.csv"

    @property
    def court_calibration_summary_csv(self) -> Path:
        return self.annotations_dir / "court_calibration_summary.csv"

    @property
    def player_tracks_csv(self) -> Path:
        return self.annotations_dir / "player_tracks.csv"

    @property
    def player_tracking_summary_csv(self) -> Path:
        return self.annotations_dir / "player_tracking_summary.csv"

    @property
    def shuttle_tracks_csv(self) -> Path:
        return self.annotations_dir / "shuttle_tracks.csv"

    @property
    def shuttle_tracking_summary_csv(self) -> Path:
        return self.annotations_dir / "shuttle_tracking_summary.csv"

    @property
    def player_tracks_smoothed_csv(self) -> Path:
        return self.annotations_dir / "player_tracks_smoothed.csv"

    @property
    def shuttle_tracks_smoothed_csv(self) -> Path:
        return self.annotations_dir / "shuttle_tracks_smoothed.csv"

    @property
    def player_smoothing_summary_csv(self) -> Path:
        return self.annotations_dir / "player_smoothing_summary.csv"

    @property
    def shuttle_smoothing_summary_csv(self) -> Path:
        return self.annotations_dir / "shuttle_smoothing_summary.csv"

    @property
    def tactics_summary_csv(self) -> Path:
        return self.tactics_dir / "tactics_summary.csv"

    @property
    def tactics_events_csv(self) -> Path:
        return self.tactics_dir / "tactics_events.csv"

    def demo_output(self, filename: str) -> Path:
        if not filename or Path(filename).name != filename:
            raise ValueError(f"demo filename must be a path-free filename, got {filename!r}")
        return self.demo_dir / filename

    def demo_intermediate_output(self, filename: str) -> Path:
        if not filename or Path(filename).name != filename:
            raise ValueError(f"demo filename must be a path-free filename, got {filename!r}")
        return self.demo_dir / f"{Path(filename).stem}.opencv.mp4"
