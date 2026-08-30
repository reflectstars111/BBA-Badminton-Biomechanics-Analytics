from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Any, Iterator

from badminton_data_process.core.paths import discover_project_root
from badminton_data_process.core.run import make_run_id
from badminton_data_process.core.schemas import STAGE_ORDER
from badminton_data_process.webui.adapter import (
    WebAnalysisRequest,
    read_run_state,
    submit_web_analysis,
)
from badminton_data_process.webui.court_annotation import (
    MODEL_LINE_LABELS,
    accept_auto_annotation,
    add_clicked_model_line,
    apply_manual_annotation,
    clear_manual_annotation,
    format_reference_points,
    prepare_court_preview,
)
from badminton_data_process.webui.styles import WEBUI_CSS

INPUT_LABELS = {
    "未裁切：包含采访、回放或切镜头": "uncut",
    "已裁切：仅包含实际打球片段": "clipped",
}
VIEW_LABELS = {
    "俯视 / 标准转播视角": "overhead",
    "低视角 / 侧面固定机位": "low",
}

STAGE_LABELS = {
    "main_view": "主视角检测与素材清洗",
    "rally_segmentation": "有效回合切分",
    "court_calibration": "白色边线与球场标定",
    "player_tracking": "球员检测与骨骼跟踪",
    "shuttle_tracking": "TrackNet 羽毛球跟踪",
    "trajectory_smoothing": "轨迹平滑与异常插值过滤",
    "visualization": "热力图与轨迹图生成",
    "tactical_analysis": "战术与统计分析",
    "demo_rendering": "最终分析视频渲染",
}
TOTAL_PIPELINE_STAGES = len(STAGE_ORDER)

# Relative GPU/CPU cost learned from the current pipeline shape. The estimate
# is calibrated with completed stages from the active run, so it adapts to the
# uploaded video and machine instead of promising a fixed global speed.
STAGE_ETA_WEIGHTS = {
    "main_view": 0.08,
    "rally_segmentation": 0.05,
    "court_calibration": 0.04,
    "player_tracking": 0.27,
    "shuttle_tracking": 0.30,
    "trajectory_smoothing": 0.04,
    "visualization": 0.06,
    "tactical_analysis": 0.05,
    "demo_rendering": 0.11,
}

PLAYER_COLUMNS = [
    "场地角色", "回合数", "有效帧", "跟踪覆盖率", "骨骼有效率",
    "总移动距离(m)", "平均速度(m/s)", "最高速度(m/s)",
    "末个有效步速度(m/s)", "平均重心相对高度", "前场占比", "中场占比", "后场占比",
]
RALLY_PLAYER_COLUMNS = [
    "回合", "场地角色", "有效帧", "跟踪覆盖率", "骨骼有效率", "移动距离(m)",
    "平均速度(m/s)", "最高速度(m/s)", "末个有效步速度(m/s)", "平均重心相对高度",
]
SHUTTLE_COLUMNS = [
    "回合", "有效观测帧", "可见率", "平均图像速度(px/s)", "最高图像速度(px/s)",
    "末个有效步速度(px/s)", "平均屏幕对角线/秒",
]


def _number(value: Any, digits: int = 2) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _percent(value: Any) -> str:
    return "—" if value is None else f"{float(value) * 100.0:.1f}%"


def player_overview_table(report: dict[str, Any]) -> list[list[str]]:
    return [
        [
            row.get("player_id", ""), str(row.get("rallies", 0)), str(row.get("valid_frames", 0)),
            _percent(row.get("tracking_coverage_ratio")), _percent(row.get("pose_valid_ratio")),
            _number(row.get("total_distance_m")), _number(row.get("average_speed_m_s")),
            _number(row.get("maximum_speed_m_s")), _number(row.get("current_speed_m_s")),
            _percent(row.get("average_body_center_height_ratio")),
            _percent(row.get("front_court_ratio")), _percent(row.get("mid_court_ratio")),
            _percent(row.get("back_court_ratio")),
        ]
        for row in report.get("players", [])
    ]


def player_rally_table(report: dict[str, Any]) -> list[list[str]]:
    return [
        [
            str(row.get("rally_id", "")), row.get("player_id", ""), str(row.get("valid_frames", 0)),
            _percent(row.get("tracking_coverage_ratio")), _percent(row.get("pose_valid_ratio")),
            _number(row.get("total_distance_m")), _number(row.get("average_speed_m_s")),
            _number(row.get("maximum_speed_m_s")), _number(row.get("current_speed_m_s")),
            _percent(row.get("average_body_center_height_ratio")),
        ]
        for row in report.get("player_rallies", [])
    ]


def shuttle_rally_table(report: dict[str, Any]) -> list[list[str]]:
    return [
        [
            str(row.get("rally_id", "")), str(row.get("valid_observed_frames", 0)),
            _percent(row.get("visibility_ratio")), _number(row.get("average_image_speed_px_s"), 1),
            _number(row.get("maximum_image_speed_px_s"), 1),
            _number(row.get("current_image_speed_px_s"), 1),
            _number(row.get("average_screen_diagonals_s"), 3),
        ]
        for row in report.get("shuttle_rallies", [])
    ]


def report_summary_markdown(report: dict[str, Any]) -> str:
    match = report.get("match", {})
    shuttle = report.get("shuttle", {})
    return (
        "### 分析结果\n\n"
        f"- 有效回合：**{match.get('usable_rallies', 0)}**\n"
        f"- 有效比赛时长：**{_number(match.get('analyzed_duration_seconds'), 1)} 秒**\n"
        f"- 羽毛球可见率：**{_percent(shuttle.get('visibility_ratio'))}**\n"
        f"- 羽毛球平均图像速度：**{_number(shuttle.get('average_image_speed_px_s'), 1)} px/s**\n"
        f"- 羽毛球稳健最高图像速度：**{_number(shuttle.get('maximum_image_speed_px_s'), 1)} px/s**\n\n"
        "> 球员移动使用已验证球场平面的米制坐标。羽毛球在空中运动，单目视频不能可靠恢复三维米制球速，"
        "因此这里展示经过异常值约束的图像平面速度，避免输出伪精确的 m/s。"
    )


def quality_markdown(report: dict[str, Any]) -> str:
    quality = report.get("quality", {})
    return (
        "### 指标口径与能力边界\n\n"
        f"- {quality.get('metric_contract', '')}\n"
        f"- {quality.get('body_center_contract', '')}\n"
        f"- {quality.get('dual_side_capability', '')}\n\n"
        "### 骨骼动作细节分析 · 正在开发中\n\n"
        "计划加入：击球动作分类、挥拍阶段分解、关节角度与稳定性、步法与启动模式。"
    )


def stage_status_markdown(state: dict[str, Any], *, running: bool) -> str:
    stages = state.get("stages", [])
    lines = ["### 阶段记录"]
    if not stages:
        lines.append("- 正在进行视频、模型与 GPU 预检……" if running else "- 尚未开始")
    for stage in stages:
        status = stage.get("status", "pending")
        icon = "✅" if status == "success" else "❌" if status in {"failed", "rejected"} else "⏳"
        duration = stage.get("duration_seconds")
        suffix = f" · {_number(duration, 1)} 秒" if duration not in (None, "") else ""
        lines.append(f"- {icon} `{stage.get('name', '')}`：{status}{suffix}")
    if running:
        lines.append("\n分析正在后台运行，请保持本页面打开。长视频可能需要较长时间。")
    return "\n".join(lines)


def pipeline_progress(state: dict[str, Any], *, running: bool) -> dict[str, Any]:
    """Derive honest stage-level progress from the persisted Run Manifest."""

    stages = list(state.get("stages", []))
    completed_statuses = {"success", "skipped"}
    completed = sum(stage.get("status") in completed_statuses for stage in stages)
    terminal = next(
        (
            stage
            for stage in stages
            if stage.get("status") in {"failed", "rejected", "empty"}
        ),
        None,
    )

    if terminal is not None:
        stage_name = str(terminal.get("name", ""))
        current = STAGE_LABELS.get(stage_name, stage_name or "未知阶段")
        state_label = {
            "failed": "分析失败",
            "rejected": "输入被拒绝",
            "empty": "没有可分析结果",
        }.get(str(terminal.get("status")), "分析停止")
    elif running and completed >= TOTAL_PIPELINE_STAGES:
        current = "生成详细分析报告"
        state_label = "正在收尾"
    elif running:
        next_stage = STAGE_ORDER[min(completed, TOTAL_PIPELINE_STAGES - 1)].value
        current = (
            "环境预检 / 主视角检测与素材清洗"
            if not stages
            else STAGE_LABELS.get(next_stage, next_stage)
        )
        state_label = "正在分析"
    elif completed >= TOTAL_PIPELINE_STAGES:
        current = "全部阶段已完成"
        state_label = "分析完成"
    else:
        current = "等待上传视频"
        state_label = "尚未开始"

    percent = round(completed / TOTAL_PIPELINE_STAGES * 100)
    if running and completed >= TOTAL_PIPELINE_STAGES:
        percent = 99
    if not running and completed >= TOTAL_PIPELINE_STAGES and terminal is None:
        percent = 100
    return {
        "completed": completed,
        "total": TOTAL_PIPELINE_STAGES,
        "percent": percent,
        "current_stage": current,
        "state_label": state_label,
        "failed": terminal is not None,
    }


def estimate_remaining_seconds(
    state: dict[str, Any],
    live_progress: dict[str, Any] | None = None,
) -> float | None:
    """Estimate remaining work only when the active stage exposes real units."""

    stages = list(state.get("stages", []))
    statuses = {str(stage.get("name", "")): str(stage.get("status", "")) for stage in stages}
    live = live_progress or {}
    live_stage = str(live.get("stage", ""))
    live_completed = max(0, int(live.get("completed_units", 0) or 0))
    live_total = max(0, int(live.get("total_units", 0) or 0))
    live_elapsed = max(0.0, float(live.get("stage_elapsed_seconds", 0.0) or 0.0))
    if live_stage not in STAGE_ETA_WEIGHTS or live_total <= 0 or live_completed <= 0:
        return None
    live_fraction = min(1.0, live_completed / live_total)
    if live_fraction < 0.02 or live_elapsed < 2.0:
        return None

    observed_duration = 0.0
    observed_weight = 0.0
    for stage in stages:
        name = str(stage.get("name", ""))
        duration = stage.get("duration_seconds")
        if stage.get("status") != "success" or duration in (None, ""):
            continue
        duration_value = max(0.0, float(duration))
        weight = STAGE_ETA_WEIGHTS.get(name, 0.0)
        if duration_value > 0.0 and weight > 0.0:
            observed_duration += duration_value
            observed_weight += weight
    current_weight = STAGE_ETA_WEIGHTS[live_stage]
    measured_duration = observed_duration + live_elapsed
    measured_weight = observed_weight + current_weight * live_fraction
    remaining_weight = sum(
        weight
        for name, weight in STAGE_ETA_WEIGHTS.items()
        if statuses.get(name) not in {"success", "skipped"}
    )
    remaining_weight = max(0.0, remaining_weight - current_weight * live_fraction)
    if measured_duration <= 0.0 or measured_weight <= 0.0:
        return None
    return measured_duration / measured_weight * remaining_weight


def duration_label(seconds: float | None, *, approximate: bool = False) -> str:
    if seconds is None:
        return "—"
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        value = "不足 1 分钟" if approximate else f"{int(seconds)} 秒"
    else:
        total_minutes = max(1, int(seconds / 60.0 + 0.5))
        hours, minutes = divmod(total_minutes, 60)
        value = f"{hours} 小时 {minutes} 分钟" if hours else f"{minutes} 分钟"
    return f"约 {value}" if approximate and seconds >= 60.0 else value


def progress_html(
    state: dict[str, Any],
    *,
    running: bool,
    elapsed_seconds: float | None = None,
    live_progress: dict[str, Any] | None = None,
) -> str:
    progress = pipeline_progress(state, running=running)
    color_class = " pipeline-progress-error" if progress["failed"] else ""
    remaining_seconds = (
        estimate_remaining_seconds(state, live_progress) if running else None
    )
    live = live_progress or {}
    live_completed = max(0, int(live.get("completed_units", 0) or 0))
    live_total = max(0, int(live.get("total_units", 0) or 0))
    live_fraction = min(1.0, live_completed / live_total) if live_total else 0.0
    current_stage_label = progress["current_stage"]
    if running and live_total > 0:
        current_stage_label += (
            f" · {live_completed} / {live_total} 帧（{live_fraction * 100.0:.1f}%）"
        )
    if running and remaining_seconds is None:
        remaining_label = "正在测量当前阶段"
    elif running:
        remaining_label = duration_label(remaining_seconds, approximate=True)
    elif progress["percent"] == 100:
        remaining_label = "已完成"
    else:
        remaining_label = "—"
    elapsed_label = duration_label(elapsed_seconds)
    segments = "".join(
        (
            '<span class="progress-segment is-complete"></span>'
            if index < progress["completed"]
            else '<span class="progress-segment is-active"></span>'
            if running and index == progress["completed"]
            else '<span class="progress-segment"></span>'
        )
        for index in range(progress["total"])
    )
    return (
        f'<section class="pipeline-progress-card{color_class}">'
        '<div class="pipeline-progress-heading"><div>'
        '<span class="progress-kicker">ANALYSIS PIPELINE</span>'
        f'<strong>{progress["state_label"]}</strong></div>'
        f'<span class="progress-percent">{progress["percent"]}%</span></div>'
        f'<progress class="pipeline-progress" value="{progress["percent"]}" max="100" '
        f'aria-label="分析进度 {progress["percent"]}%"></progress>'
        f'<div class="progress-segments" aria-hidden="true">{segments}</div>'
        '<div class="pipeline-progress-detail">'
        f'<span><small>当前阶段</small><b>{current_stage_label}</b></span>'
        f'<span><small>预计剩余</small><b>{remaining_label}</b></span>'
        f'<span><small>{"总用时" if not running and elapsed_seconds is not None else "已运行"}</small>'
        f'<b>{elapsed_label}</b></span>'
        f'<span class="progress-count"><small>流程进度</small><b>已完成 {progress["completed"]} / {progress["total"]} 阶段</b></span>'
        '</div></section>'
    )


def _result_files(report: dict[str, Any]) -> list[str]:
    candidates = [
        report.get("outputs", {}).get("web_report"), report.get("outputs", {}).get("tactical_summary"),
        report.get("outputs", {}).get("player_tracks"), report.get("outputs", {}).get("shuttle_tracks"),
        report.get("outputs", {}).get("manifest"),
    ]
    return [str(path) for path in candidates if path and Path(path).is_file()]


def _chart_files(report: dict[str, Any]) -> list[str]:
    chart_value = report.get("outputs", {}).get("tracking_charts")
    if not chart_value:
        return []
    chart_dir = Path(chart_value)
    return [str(path) for path in sorted(chart_dir.glob("*.png"))] if chart_dir.is_dir() else []


def build_app():
    """Build the Gradio shell around the shared full-analysis Adapter."""

    global gr  # Gradio resolves event type hints from the module namespace.
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - optional UI dependency
        raise RuntimeError("WebUI requires the optional dependency: pip install '.[ui]'") from exc

    def run(
        video_path: str | None,
        input_label: str,
        view_label: str,
        manual_reference_points: list[float] | None,
    ) -> Iterator[tuple[Any, ...]]:
        empty = (None, "", [], [], [], "", [], [])
        if not video_path:
            missing_state: dict[str, Any] = {"stages": []}
            yield (
                progress_html(missing_state, running=False),
                "### 无法开始\n\n请先上传视频。",
                *empty,
            )
            return
        if not manual_reference_points:
            waiting_state: dict[str, Any] = {"stages": []}
            yield (
                progress_html(waiting_state, running=False),
                "### 尚未开始\n\n请先检查代表帧中的场地标定，并选择“接受自动标定”或应用手动四点标注。",
                *empty,
            )
            return
        root = discover_project_root()
        run_id = make_run_id("webui")
        request = WebAnalysisRequest(
            input_video=Path(video_path), input_kind=INPUT_LABELS[input_label],
            view_kind=VIEW_LABELS[view_label], run_id=run_id, root=root,
            manual_reference_points=manual_reference_points,
        )
        result: dict[str, Any] = {}
        live_progress: dict[str, Any] = {}
        progress_lock = threading.Lock()

        def update_live_progress(stage: str, completed: int, total: int) -> None:
            now = time.monotonic()
            with progress_lock:
                if live_progress.get("stage") != stage:
                    live_progress.clear()
                    live_progress["stage"] = stage
                    live_progress["stage_started_at"] = now
                live_progress["completed_units"] = max(0, int(completed))
                live_progress["total_units"] = max(0, int(total))

        def live_progress_snapshot(now: float) -> dict[str, Any]:
            with progress_lock:
                snapshot = dict(live_progress)
            stage_started_at = snapshot.pop("stage_started_at", None)
            if stage_started_at is not None:
                snapshot["stage_elapsed_seconds"] = max(0.0, now - stage_started_at)
            return snapshot

        def worker() -> None:
            try:
                result["value"] = submit_web_analysis(
                    request,
                    progress_callback=update_live_progress,
                )
            except Exception as exc:  # noqa: BLE001 - surface pipeline failure in UI
                result["error"] = exc

        thread = threading.Thread(target=worker, name=f"analysis-{run_id}", daemon=True)
        started_at = time.monotonic()
        thread.start()
        while thread.is_alive():
            state = read_run_state(root, run_id)
            now = time.monotonic()
            yield (
                progress_html(
                    state,
                    running=True,
                    elapsed_seconds=now - started_at,
                    live_progress=live_progress_snapshot(now),
                ),
                stage_status_markdown(state, running=True),
                *empty,
            )
            time.sleep(2.0)
        thread.join()
        state = read_run_state(root, run_id)
        elapsed_seconds = time.monotonic() - started_at
        if "error" in result:
            message = str(result["error"]) or type(result["error"]).__name__
            yield (
                progress_html(state, running=False, elapsed_seconds=elapsed_seconds),
                stage_status_markdown(state, running=False)
                + f"\n\n### 分析未完成\n\n`{message}`\n\n请检查 Run Manifest 或终端日志。",
                *empty,
            )
            return
        _, _, report = result["value"]
        yield (
            progress_html(state, running=False, elapsed_seconds=elapsed_seconds),
            stage_status_markdown(state, running=False) + "\n\n### ✅ 分析完成",
            report.get("match", {}).get("analysis_video"), report_summary_markdown(report),
            player_overview_table(report), player_rally_table(report), shuttle_rally_table(report),
            quality_markdown(report), _result_files(report), _chart_files(report),
        )

    with gr.Blocks(
        title="BBA · Badminton Biomechanics Analytics",
        theme=gr.themes.Base(primary_hue="orange", neutral_hue="slate"),
        css=WEBUI_CSS,
    ) as app:
        gr.HTML(
            "<header class='bba-topbar'>"
            "<div class='bba-brand'><span class='bba-mark'>BBA</span><span class='bba-name'>"
            "<b>Badminton Biomechanics Analytics</b><small>羽毛球生物力学分析系统</small>"
            "</span></div><div class='runtime-badge'><i></i>LOCAL GPU WORKSPACE</div></header>"
        )
        gr.HTML(
            "<section class='bba-intro'><span class='overline'>MATCH INTELLIGENCE / 01</span>"
            "<h1>从比赛画面，读取运动表现。</h1>"
            "<p>自动清洗复杂视频，建立标准球场坐标，追踪球员骨骼与羽毛球轨迹，最终形成可复核的数据报告。</p>"
            "</section>"
        )
        gr.HTML(
            "<nav class='stage-rail' aria-label='分析流程'>"
            "<div class='stage-node is-current'><b>01</b><span>素材配置<small>UPLOAD</small></span></div>"
            "<div class='stage-node'><b>02</b><span>场地校准<small>CALIBRATE</small></span></div>"
            "<div class='stage-node'><b>03</b><span>模型分析<small>PROCESS</small></span></div>"
            "<div class='stage-node'><b>04</b><span>结果报告<small>REPORT</small></span></div>"
            "</nav>"
        )

        gr.HTML(
            "<header class='stage-heading'><span>01</span><div><small>INPUT</small>"
            "<h2>建立分析任务</h2><p>上传一段比赛视频，并选择与素材相符的处理模式。</p></div></header>"
        )
        with gr.Group(elem_classes=["workspace-panel", "upload-panel"]):
            video = gr.Video(
                label="上传比赛视频",
                sources=["upload"],
                elem_classes=["video-upload"],
            )
            gr.HTML("<div class='panel-divider'><span>任务参数</span></div>")
            with gr.Row(elem_classes=["config-row"]):
                input_kind = gr.Radio(
                    list(INPUT_LABELS),
                    value=list(INPUT_LABELS)[0],
                    label="素材状态",
                    info="未裁切素材会自动排除采访、回放、特写及切镜头。",
                )
                view_kind = gr.Radio(
                    list(VIEW_LABELS),
                    value=list(VIEW_LABELS)[0],
                    label="固定机位",
                    info="低视角支持画外角点建模，目前属于实验配置。",
                )
            gr.HTML(
                "<aside class='privacy-note'><b>本地分析</b><span>视频与结果保留在当前计算机；"
                "首次运行会自动检查 CUDA、模型与环境。</span></aside>"
            )

        base_court_frame = gr.State(value=None)
        clicked_model_points = gr.State(value=[])
        auto_reference_points = gr.State(value=None)
        manual_reference_points = gr.State(value=None)
        gr.HTML(
            "<header class='stage-heading'><span>02</span><div><small>CALIBRATION</small>"
            "<h2>核对球场坐标</h2><p>这一步决定球员位置、移动距离和速度指标是否可信。</p></div></header>"
        )
        with gr.Group(elem_classes=["workspace-panel", "calibration-panel"]):
            corner_status = gr.Markdown(
                "上传视频后，系统会自动挑选代表帧并给出球场建议。",
                elem_classes=["calibration-status"],
            )
            court_preview = gr.Image(
                label="代表帧 / 标准球场投影",
                type="numpy",
                interactive=False,
                elem_classes=["court-preview"],
            )
            with gr.Row(elem_classes=["primary-actions"]):
                accept_auto_button = gr.Button(
                    "确认自动标定",
                    variant="primary",
                    interactive=False,
                )
                preview_button = gr.Button("重新选择代表帧", variant="secondary")
                clear_corners_button = gr.Button("清除确认", variant="secondary")
            with gr.Accordion(
                "自动结果不准确？打开标准球场模型手动修正",
                open=False,
                elem_classes=["manual-calibration"],
            ):
                gr.Markdown(
                    "选择画面中实际可见的两条纵线和两条横线，然后依次在每条线上点两个相距较远的点，"
                    "共 8 点。系统会延长直线，并使用 6.10 m × 13.40 m 标准模型推导画外角点。",
                    elem_classes=["manual-help"],
                )
                with gr.Row(elem_classes=["line-selectors"]):
                    left_model_line = gr.Dropdown(
                        choices=[
                            (MODEL_LINE_LABELS["left_doubles_sideline"], "left_doubles_sideline"),
                            (MODEL_LINE_LABELS["left_singles_sideline"], "left_singles_sideline"),
                        ],
                        value="left_doubles_sideline",
                        label="左侧纵线",
                    )
                    right_model_line = gr.Dropdown(
                        choices=[
                            (MODEL_LINE_LABELS["right_doubles_sideline"], "right_doubles_sideline"),
                            (MODEL_LINE_LABELS["right_singles_sideline"], "right_singles_sideline"),
                        ],
                        value="right_doubles_sideline",
                        label="右侧纵线",
                    )
                    far_model_line = gr.Dropdown(
                        choices=[
                            (MODEL_LINE_LABELS["far_baseline"], "far_baseline"),
                            (MODEL_LINE_LABELS["far_doubles_long_service"], "far_doubles_long_service"),
                            (MODEL_LINE_LABELS["far_short_service"], "far_short_service"),
                        ],
                        value="far_short_service",
                        label="远端横线",
                    )
                    near_model_line = gr.Dropdown(
                        choices=[
                            (MODEL_LINE_LABELS["near_short_service"], "near_short_service"),
                            (MODEL_LINE_LABELS["near_doubles_long_service"], "near_doubles_long_service"),
                            (MODEL_LINE_LABELS["near_baseline"], "near_baseline"),
                        ],
                        value="near_short_service",
                        label="近端横线",
                    )
                with gr.Accordion("高级坐标编辑", open=False, elem_classes=["advanced-corners"]):
                    corner_coordinates = gr.Textbox(
                        label="归一化外角 · TL, TR, BR, BL",
                        placeholder="0.10,0.30; 0.90,0.30; 1.20,0.95; 0.05,0.95",
                    )
                apply_corners_button = gr.Button("应用手动模型", variant="primary")

        with gr.Group(elem_classes=["launch-panel"]):
            gr.HTML(
                "<div class='launch-copy'><small>READY WHEN CALIBRATED</small>"
                "<b>场地确认后即可运行完整分析</b><span>同一台机器一次处理一个任务，请保持页面打开。</span></div>"
            )
            submit = gr.Button(
                "等待场地确认",
                variant="primary",
                elem_id="run-button",
                interactive=False,
            )

        gr.HTML(
            "<header class='stage-heading'><span>03</span><div><small>PROCESS</small>"
            "<h2>分析运行状态</h2><p>九个阶段均来自实际运行清单，不使用虚假加载进度。</p></div></header>"
        )
        progress_bar = gr.HTML(progress_html({"stages": []}, running=False))
        with gr.Accordion("运行日志与阶段记录", open=False, elem_classes=["stage-log"]):
            status = gr.Markdown("### 阶段记录\n\n- 等待分析任务")

        gr.HTML(
            "<header class='stage-heading report-heading'><span>04</span><div><small>REPORT</small>"
            "<h2>比赛分析结果</h2><p>视频、球员指标、逐回合数据和原始文件集中在这里。</p></div></header>"
        )
        with gr.Tabs(elem_classes=["results-tabs"]):
            with gr.Tab("视频与核心指标"):
                output_video = gr.Video(label="骨骼与轨迹分析视频", elem_classes=["results-video"])
                summary = gr.Markdown(elem_classes=["report-summary"])
                player_overview = gr.Dataframe(
                    headers=PLAYER_COLUMNS,
                    datatype=["str"] * len(PLAYER_COLUMNS),
                    interactive=False,
                    wrap=True,
                    label="球员全场指标",
                    elem_classes=["report-table"],
                )
            with gr.Tab("逐回合数据"):
                rally_players = gr.Dataframe(
                    headers=RALLY_PLAYER_COLUMNS,
                    datatype=["str"] * len(RALLY_PLAYER_COLUMNS),
                    interactive=False,
                    wrap=True,
                    label="球员逐回合指标",
                    elem_classes=["report-table"],
                )
                rally_shuttle = gr.Dataframe(
                    headers=SHUTTLE_COLUMNS,
                    datatype=["str"] * len(SHUTTLE_COLUMNS),
                    interactive=False,
                    wrap=True,
                    label="羽毛球逐回合指标",
                    elem_classes=["report-table"],
                )
            with gr.Tab("可视化与文件"):
                gallery = gr.Gallery(label="轨迹、散点与热力图", columns=2, height="auto")
                downloads = gr.Files(label="下载 JSON / CSV / Run Manifest")
            with gr.Tab("动作分析路线图"):
                quality = gr.Markdown(
                    "### 骨骼动作细节分析 · 正在开发中\n\n"
                    "计划加入击球动作分类、挥拍阶段分解、关节角度与稳定性、步法与启动模式。"
                )
        outputs = [
            progress_bar, status, output_video, summary, player_overview,
            rally_players, rally_shuttle, quality, downloads, gallery,
        ]
        def prepare_annotation(video_path: str | None, view_label: str):
            if not video_path:
                return (
                    None,
                    None,
                    [],
                    None,
                    None,
                    "",
                    "请先上传视频；正式分析必须经过场地标定确认。",
                    gr.update(value="请先确认场地标定", interactive=False),
                    gr.update(interactive=False),
                )
            try:
                preview, base, auto_points, message = prepare_court_preview(
                    video_path,
                    VIEW_LABELS[view_label],
                )
            except Exception as exc:  # noqa: BLE001 - UI boundary
                raise gr.Error(str(exc)) from exc
            return (
                preview,
                base,
                [],
                auto_points,
                None,
                format_reference_points(auto_points) if auto_points else "",
                message,
                gr.update(value="请先确认场地标定", interactive=False),
                gr.update(interactive=bool(auto_points)),
            )

        def select_model_line(
            base: Any,
            points: list[tuple[float, float]] | None,
            left_line: str,
            right_line: str,
            far_line: str,
            near_line: str,
            view_label: str,
            evt: gr.SelectData,
        ):
            try:
                preview, updated, text, message = add_clicked_model_line(
                    base,
                    points,
                    tuple(evt.index),
                    [left_line, right_line, far_line, near_line],
                    VIEW_LABELS[view_label],
                )
            except Exception as exc:  # noqa: BLE001 - UI boundary
                raise gr.Error(str(exc)) from exc
            return (
                preview,
                updated,
                None,
                text,
                message,
                gr.update(value="请先确认场地标定", interactive=False),
            )

        def accept_annotation(base: Any, points: list[float] | None, view_label: str):
            try:
                preview, normalized, message = accept_auto_annotation(
                    base,
                    points,
                    VIEW_LABELS[view_label],
                )
            except Exception as exc:  # noqa: BLE001 - UI boundary
                raise gr.Error(str(exc)) from exc
            return (
                preview,
                normalized,
                message,
                gr.update(value="开始完整分析", interactive=True),
            )

        def apply_annotation(base: Any, text: str, view_label: str):
            try:
                preview, normalized, message = apply_manual_annotation(
                    base,
                    text,
                    VIEW_LABELS[view_label],
                )
            except Exception as exc:  # noqa: BLE001 - UI boundary
                raise gr.Error(str(exc)) from exc
            return (
                preview,
                normalized,
                message,
                gr.update(value="开始完整分析", interactive=True),
            )

        def clear_annotation(base: Any):
            preview, points, confirmed, text, message = clear_manual_annotation(base)
            return (
                preview,
                points,
                confirmed,
                text,
                message,
                gr.update(value="请先确认场地标定", interactive=False),
            )

        def reset_model_selection():
            return (
                [],
                None,
                "",
                "模型线选择已变更；请按当前四条线重新点击 8 个点。",
                gr.update(value="请先确认场地标定", interactive=False),
            )

        annotation_prepare_outputs = [
            court_preview,
            base_court_frame,
            clicked_model_points,
            auto_reference_points,
            manual_reference_points,
            corner_coordinates,
            corner_status,
            submit,
            accept_auto_button,
        ]

        preview_button.click(
            prepare_annotation,
            [video, view_kind],
            annotation_prepare_outputs,
        )
        video.change(
            prepare_annotation,
            [video, view_kind],
            annotation_prepare_outputs,
        )
        view_kind.change(
            prepare_annotation,
            [video, view_kind],
            annotation_prepare_outputs,
        )
        court_preview.select(
            select_model_line,
            [
                base_court_frame,
                clicked_model_points,
                left_model_line,
                right_model_line,
                far_model_line,
                near_model_line,
                view_kind,
            ],
            [
                court_preview,
                clicked_model_points,
                manual_reference_points,
                corner_coordinates,
                corner_status,
                submit,
            ],
        )
        accept_auto_button.click(
            accept_annotation,
            [base_court_frame, auto_reference_points, view_kind],
            [court_preview, manual_reference_points, corner_status, submit],
        )
        apply_corners_button.click(
            apply_annotation,
            [base_court_frame, corner_coordinates, view_kind],
            [court_preview, manual_reference_points, corner_status, submit],
        )
        clear_corners_button.click(
            clear_annotation,
            [base_court_frame],
            [
                court_preview,
                clicked_model_points,
                manual_reference_points,
                corner_coordinates,
                corner_status,
                submit,
            ],
        )
        for model_line_selector in (
            left_model_line,
            right_model_line,
            far_model_line,
            near_model_line,
        ):
            model_line_selector.change(
                reset_model_selection,
                [],
                [
                    clicked_model_points,
                    manual_reference_points,
                    corner_coordinates,
                    corner_status,
                    submit,
                ],
            )
        submit.click(
            run,
            [video, input_kind, view_kind, manual_reference_points],
            outputs,
            concurrency_limit=1,
        )
    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the BBA browser analysis workspace.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = discover_project_root()
    build_app().queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, share=args.share,
        inbrowser=not args.no_browser, allowed_paths=[str(root / "runs")],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
