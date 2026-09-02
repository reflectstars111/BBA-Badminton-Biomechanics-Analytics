from __future__ import annotations

from badminton_data_process.core import verify


def test_production_profile_declares_complete_analysis_runtime() -> None:
    expected = {
        "gradio",
        "rtmlib",
        "onnxruntime",
        "sklearn",
        "openpyxl",
        "moviepy",
        "transformers",
        "torcheval",
        "positional_encodings",
        "torchinfo",
        "gdown",
    }

    assert expected <= set(verify.PRODUCTION_PACKAGES)


def test_strict_verification_returns_failure_for_unavailable_component(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        verify,
        "collect_checks",
        lambda *args, **kwargs: [
            verify.ComponentCheck(
                name="onnx-cuda",
                status=verify.STATUS_UNAVAILABLE,
                message="CUDAExecutionProvider missing",
            )
        ],
    )

    assert verify.main(["--profile", "production", "--strict"]) == 1
    assert verify.main(["--profile", "production"]) == 0


def test_bst_verification_requires_repository_and_weights_together() -> None:
    checks = verify.collect_checks("core", bst_repository=None, bst_weights=None)
    assert all(check.name != "bst-runtime" for check in checks)

    checks = verify.collect_checks("core", bst_repository=None, bst_weights=verify.Path("model.pt"))
    bst = next(check for check in checks if check.name == "bst-runtime")
    assert bst.status == verify.STATUS_UNAVAILABLE
    assert "supplied together" in bst.message
