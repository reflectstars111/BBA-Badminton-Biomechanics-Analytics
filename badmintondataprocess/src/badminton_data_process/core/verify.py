from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

STATUS_OK = "ok"
STATUS_MISSING = "missing"
STATUS_UNAVAILABLE = "unavailable"

# Keep the core profile useful for lightweight development checks. The production
# profile is the contract used by setup_runtime.ps1 and start_webui.ps1.
CORE_PACKAGES = {
    "numpy": "numpy",
    "cv2": "opencv-python",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "pandas": "pandas",
    "torch": "torch",
    "torchvision": "torchvision",
    "ultralytics": "ultralytics",
    "yt_dlp": "yt-dlp",
    "imageio": "imageio",
    "imageio_ffmpeg": "imageio-ffmpeg",
    "yaml": "PyYAML",
}

PRODUCTION_PACKAGES = {
    "gradio": "gradio",
    "rtmlib": "rtmlib",
    "onnxruntime": "onnxruntime-gpu",
    "sklearn": "scikit-learn",
    "seaborn": "seaborn",
    "openpyxl": "openpyxl",
    "moviepy": "moviepy",
    "transformers": "transformers",
    "torcheval": "torcheval",
    "einops": "einops",
    "parse": "parse",
    "positional_encodings": "positional-encodings",
    "torchinfo": "torchinfo",
    "gdown": "gdown",
}


@dataclass(slots=True)
class ComponentCheck:
    name: str
    status: str
    version: str = ""
    message: str = ""


def check_python(*, production: bool = False) -> ComponentCheck:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    minimum = (3, 12) if production else (3, 10)
    if sys.version_info[:2] < minimum:
        return ComponentCheck(
            name="python",
            status=STATUS_UNAVAILABLE,
            version=version,
            message=f"Python >= {minimum[0]}.{minimum[1]} required",
        )
    if production and sys.version_info[:2] != (3, 12):
        return ComponentCheck(
            name="python",
            status=STATUS_UNAVAILABLE,
            version=version,
            message="the good-badminton production environment is pinned to Python 3.12",
        )
    return ComponentCheck(name="python", status=STATUS_OK, version=version)


def check_package(import_name: str, dist_name: str) -> ComponentCheck:
    try:
        __import__(import_name)
    except Exception as exc:  # noqa: BLE001 - imports may fail for binary/runtime reasons
        return ComponentCheck(name=import_name, status=STATUS_MISSING, message=str(exc))
    try:
        version = importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        version = ""
    return ComponentCheck(name=import_name, status=STATUS_OK, version=version)


def check_ffmpeg() -> ComponentCheck:
    try:
        import imageio_ffmpeg

        executable = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001
        return ComponentCheck(name="ffmpeg", status=STATUS_UNAVAILABLE, message=str(exc))
    return ComponentCheck(name="ffmpeg", status=STATUS_OK, version=str(executable))


def check_torch_cuda() -> ComponentCheck:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return ComponentCheck(
            name="torch-cuda", status=STATUS_UNAVAILABLE, message=f"torch unavailable: {exc}"
        )
    if not torch.cuda.is_available():
        return ComponentCheck(
            name="torch-cuda", status=STATUS_UNAVAILABLE, message="CUDA is not available"
        )
    return ComponentCheck(
        name="torch-cuda",
        status=STATUS_OK,
        version=str(torch.version.cuda),
        message=torch.cuda.get_device_name(0),
    )


def check_onnx_cuda() -> ComponentCheck:
    try:
        import onnxruntime as ort
    except Exception as exc:  # noqa: BLE001
        return ComponentCheck(
            name="onnx-cuda", status=STATUS_UNAVAILABLE, message=f"onnxruntime unavailable: {exc}"
        )
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        return ComponentCheck(
            name="onnx-cuda",
            status=STATUS_UNAVAILABLE,
            version=getattr(ort, "__version__", ""),
            message=f"CUDAExecutionProvider missing; providers={providers}",
        )
    return ComponentCheck(
        name="onnx-cuda",
        status=STATUS_OK,
        version=getattr(ort, "__version__", ""),
        message=", ".join(providers),
    )


def check_dependency_metadata() -> ComponentCheck:
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    # RTMLib imports the module `onnxruntime`, which is supplied by the GPU wheel.
    # Its metadata nevertheless requires the distribution named `onnxruntime`.
    # Installing that CPU distribution alongside `onnxruntime-gpu` would overwrite
    # the same module and is less correct than this explicit, runtime-verified exception.
    ignored = [
        line for line in lines
        if line.lower().startswith("rtmlib 0.0.16 requires onnxruntime, which is not installed")
    ]
    failures = [line for line in lines if line not in ignored and line != "No broken requirements found."]
    if failures:
        return ComponentCheck(
            name="dependency-metadata",
            status=STATUS_UNAVAILABLE,
            message="; ".join(failures),
        )
    message = "pip check passed"
    if ignored:
        message += "; accepted RTMLib/onnxruntime-gpu metadata exception"
    return ComponentCheck(name="dependency-metadata", status=STATUS_OK, message=message)


def check_bst_runtime(repository: Path, weights: Path) -> ComponentCheck:
    try:
        from badminton_data_process.analysis.biomechanics.bst import BSTRuntime

        runtime = BSTRuntime(
            repository=repository,
            weights=weights,
            device="cuda",
            model_name="BST_AP",
            pose_style="JnB_bone",
            seq_len=30,
            num_classes=25,
        )
    except Exception as exc:  # noqa: BLE001 - strict checkpoint/runtime smoke test
        return ComponentCheck(name="bst-runtime", status=STATUS_UNAVAILABLE, message=str(exc))
    return ComponentCheck(
        name="bst-runtime",
        status=STATUS_OK,
        version=runtime.model_id,
        message=f"device={runtime.device}",
    )


def collect_checks(
    profile: str = "core",
    *,
    bst_repository: Path | None = None,
    bst_weights: Path | None = None,
) -> list[ComponentCheck]:
    production = profile == "production"
    checks = [check_python(production=production)]
    packages = dict(CORE_PACKAGES)
    if production:
        packages.update(PRODUCTION_PACKAGES)
    checks.extend(check_package(import_name, dist_name) for import_name, dist_name in packages.items())
    checks.append(check_ffmpeg())
    checks.append(check_torch_cuda())
    if production:
        checks.append(check_onnx_cuda())
        checks.append(check_dependency_metadata())
    if bst_repository is not None or bst_weights is not None:
        if bst_repository is None or bst_weights is None:
            checks.append(
                ComponentCheck(
                    name="bst-runtime",
                    status=STATUS_UNAVAILABLE,
                    message="--bst-repository and --bst-weights must be supplied together",
                )
            )
        else:
            checks.append(check_bst_runtime(bst_repository, bst_weights))
    return checks


def render_text(checks: list[ComponentCheck]) -> str:
    marker = {STATUS_OK: "OK", STATUS_MISSING: "MISSING", STATUS_UNAVAILABLE: "UNAVAILABLE"}
    lines: list[str] = []
    for check in checks:
        details = " | ".join(value for value in (check.version, check.message) if value)
        lines.append(f"[{marker[check.status]:11}] {check.name:<22} {details}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the BBA runtime environment.")
    parser.add_argument("--profile", choices=("core", "production"), default="core")
    parser.add_argument("--strict", action="store_true", help="return non-zero when any check fails")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--bst-repository", type=Path)
    parser.add_argument("--bst-weights", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checks = collect_checks(
        args.profile,
        bst_repository=args.bst_repository,
        bst_weights=args.bst_weights,
    )
    if args.json:
        print(json.dumps([asdict(check) for check in checks], ensure_ascii=False, indent=2))
    else:
        print(render_text(checks))
    failed = any(check.status != STATUS_OK for check in checks)
    return 1 if args.strict and failed else 0
