from __future__ import annotations

import json
from pathlib import Path

from badminton_data_process.core.paths import discover_project_root
from badminton_data_process.core.run import make_run_id
from badminton_data_process.webui.adapter import RunSpecification, read_run_state, submit_run


def build_app():
    """Build the optional Gradio shell around the shared pipeline Adapter."""

    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - optional UI dependency
        raise RuntimeError("WebUI requires the optional dependency: pip install '.[ui]'") from exc

    def run(video_path: str, run_id: str, config_path: str, stop_after: str):
        root = discover_project_root()
        selected_run_id = run_id.strip() or make_run_id("webui")
        spec = RunSpecification(
            input_video=Path(video_path),
            run_id=selected_run_id,
            root=root,
            config_path=Path(config_path) if config_path.strip() else None,
            stop_after=None if stop_after == "complete" else stop_after,
        )
        try:
            submit_run(spec)
        except Exception:
            # The manifest is authoritative and preserves failed/rejected/empty
            # semantics; expose it instead of inventing a UI success flag.
            pass
        state = read_run_state(root, selected_run_id)
        artifact_paths = [
            item["path"]
            for item in state["artifacts"]
            if item.get("status") == "valid" and Path(item.get("path", "")).is_file()
        ]
        return json.dumps(state, ensure_ascii=False, indent=2), artifact_paths

    with gr.Blocks(title="Badminton Research Pipeline") as app:
        gr.Markdown(
            "# 羽毛球研究管线\n"
            "默认能力：近端分析；远端双人观测仍为实验能力。界面与 CLI 使用同一流水线。"
        )
        video = gr.File(label="输入视频", type="filepath")
        run_id = gr.Textbox(label="Run ID（留空自动生成）")
        config_path = gr.Textbox(label="配置 YAML（可选）")
        stop_after = gr.Dropdown(
            ["complete", "main_view", "rally", "calibrate", "tracking"],
            value="complete",
            label="运行到",
        )
        submit = gr.Button("运行分析", variant="primary")
        state = gr.Code(label="Run Manifest 状态", language="json")
        artifacts = gr.Files(label="有效 Artifacts")
        submit.click(run, [video, run_id, config_path, stop_after], [state, artifacts])
    return app


def main() -> int:
    build_app().launch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
