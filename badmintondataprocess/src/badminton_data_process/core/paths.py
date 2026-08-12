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

