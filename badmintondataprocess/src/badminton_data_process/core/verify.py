from __future__ import annotations

import importlib.metadata
import json
import sys
from dataclasses import asdict, dataclass

STATUS_OK = "ok"
STATUS_MISSING = "missing"
STATUS_UNAVAILABLE = "unavailable"

# import name -> distribution name (differs when the wheel is hyphenated)
_PACKAGES = {
    "numpy": "numpy",
    "cv2": "opencv-python",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "torch": "torch",
    "ultralytics": "ultralytics",
    "yt_dlp": "yt-dlp",
    "imageio_ffmpeg": "imageio-ffmpeg",
}


@dataclass(slots=True)
class ComponentCheck:
    name: str
    status: str
    version: str = ""
    message: str = ""


def check_python() -> ComponentCheck:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info < (3, 10):
        return ComponentCheck(
            name="python", status=STATUS_UNAVAILABLE, version=version,
            message="Python >= 3.10 required",
        )
    return ComponentCheck(name="python", status=STATUS_OK, version=version)


def check_package(import_name: str, dist_name: str) -> ComponentCheck:
    try:
        __import__(import_name)
    except Exception as exc:  # noqa: BLE001 - report any import failure
        return ComponentCheck(name=import_name, status=STATUS_MISSING, message=str(exc))
    try:
        version = importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        version = ""
    return ComponentCheck(name=import_name, status=STATUS_OK, version=version)


def check_ffmpeg() -> ComponentCheck:
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001
        return ComponentCheck(name="ffmpeg", status=STATUS_UNAVAILABLE, message=str(exc))
    return ComponentCheck(name="ffmpeg", status=STATUS_OK, version=str(exe))


def check_torch_cuda() -> ComponentCheck:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return ComponentCheck(
            name="torch-cuda", status=STATUS_UNAVAILABLE, message=f"torch not installed: {exc}"
        )
    if not torch.cuda.is_available():
        return ComponentCheck(name="torch-cuda", status=STATUS_UNAVAILABLE, message="CUDA not available")
    return ComponentCheck(
        name="torch-cuda",
        status=STATUS_OK,
        version=str(torch.version.cuda),
        message=torch.cuda.get_device_name(0),
    )


def collect_checks() -> list[ComponentCheck]:
    checks = [check_python()]
    checks.extend(check_package(import_name, dist_name) for import_name, dist_name in _PACKAGES.items())
    checks.append(check_ffmpeg())
    checks.append(check_torch_cuda())
    return checks


def render_text(checks: list[ComponentCheck]) -> str:
    marker = {STATUS_OK: "OK", STATUS_MISSING: "MISSING", STATUS_UNAVAILABLE: "UNAVAILABLE"}
    lines = []
    for check in checks:
        detail = check.version or check.message
        lines.append(f"[{marker[check.status]:11}] {check.name:<16} {detail}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    checks = collect_checks()
    if "--json" in argv:
        print(json.dumps([asdict(check) for check in checks], ensure_ascii=False, indent=2))
    else:
        print(render_text(checks))
    return 0
