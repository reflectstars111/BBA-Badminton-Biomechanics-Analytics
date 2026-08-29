import json
import os

import cv2
import gradio as gr
import numpy as np

from webui.pipeline import imread_safe, prepare_court, run_analysis

_MAX_VIDEO_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
_MAX_IMAGE_BYTES = 50 * 1024 * 1024  # 50 MB


def _validate_file_size(path, max_bytes, label="File"):
    if path and os.path.isfile(path):
        size = os.path.getsize(path)
        if size > max_bytes:
            max_mb = max_bytes / (1024 * 1024)
            raise gr.Error(f"{label} exceeds {max_mb:.0f} MB limit.")


def _bgr_to_rgb(img):
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def detect_court(template_file):
    if template_file is None:
        raise gr.Error("Please upload a court template image first.")
    _validate_file_size(template_file, _MAX_IMAGE_BYTES, "Template image")

    result = prepare_court(template_file)
    preview_rgb = _bgr_to_rgb(result["preview_bgr"])
    corners = result["corners"]
    if corners is None:
        gr.Warning("Auto-detection failed. Click 4 court corners on the image "
                   "(top-left, top-right, bottom-right, bottom-left) to annotate manually.")
        template_img = imread_safe(template_file)
        preview_rgb = _bgr_to_rgb(template_img)
    return preview_rgb, corners


def on_court_image_select(corners_state, template_file, evt: gr.SelectData):
    """Accumulate clicked points and redraw markers on the template."""
    if corners_state is None:
        corners_state = []

    if len(corners_state) >= 4:
        corners_state = []

    x, y = evt.index
    corners_state.append((x, y))

    template_img = imread_safe(template_file)
    preview = template_img.copy()
    for idx, pt in enumerate(corners_state, 1):
        cv2.circle(preview, pt, 6, (0, 0, 255), -1)
        cv2.putText(preview, str(idx), (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    if len(corners_state) > 1:
        cv2.polylines(preview, [np.array(corners_state, dtype=np.int32)],
                      len(corners_state) == 4, (0, 255, 0), 2)

    status = f"Corners selected: {len(corners_state)}/4"
    if len(corners_state) == 4:
        status += " — corners locked. Click 'Apply Manual Corners' or re-click to restart."

    corners_out = corners_state if len(corners_state) == 4 else None
    return _bgr_to_rgb(preview), corners_state, corners_out, status


def apply_manual_corners(template_file, corners_state):
    """Re-run court resolution with manually clicked corners."""
    if not corners_state or len(corners_state) != 4:
        raise gr.Error("Please click exactly 4 corners on the court image first.")

    result = prepare_court(template_file, manual_corners=corners_state)
    preview_rgb = _bgr_to_rgb(result["preview_bgr"])
    corners = result["corners"]
    if corners is None:
        raise gr.Error("Failed to resolve corners. Please try again.")
    return preview_rgb, corners


def run_full_analysis(video_file, template_file, corners,
                      pose_family, pose_mode, language, audio,
                      show_skeletons, show_player_trajectories,
                      show_court_trajectory, show_shuttlecock_trajectory,
                      show_player_stats, show_pose_roi, visualize_positions,
                      yolo_pose_model, ball_model,
                      progress=gr.Progress(track_tqdm=False)):
    if video_file is None:
        raise gr.Error("Please upload a video file.")
    if template_file is None:
        raise gr.Error("Please upload a court template image.")
    if not corners or len(corners) != 4:
        raise gr.Error("Please detect or manually annotate court corners first.")

    _validate_file_size(video_file, _MAX_VIDEO_BYTES, "Video")
    _validate_file_size(template_file, _MAX_IMAGE_BYTES, "Template image")

    options = {
        "pose_family": pose_family,
        "pose_mode": pose_mode,
        "language": language,
        "audio": audio,
        "show_skeletons": show_skeletons,
        "show_player_trajectories": show_player_trajectories,
        "show_court_trajectory": show_court_trajectory,
        "show_shuttlecock_trajectory": show_shuttlecock_trajectory,
        "show_player_stats": show_player_stats,
        "show_pose_roi": show_pose_roi,
        "visualize_positions": visualize_positions,
        "yolo_pose_model": yolo_pose_model or "weights/yolo11n-pose.pt",
        "ball_model": ball_model or "weights/yolo11s-ball.pt",
    }

    def progress_cb(frame, total):
        progress(frame / total, desc=f"Processing frame {frame}/{total}")

    result = run_analysis(
        video_path=video_file,
        template_path=template_file,
        corners=corners,
        options=options,
        progress_cb=progress_cb,
    )

    output_video = result["video"] if os.path.isfile(result["video"]) else None
    viz_images = [img for img in result["visualizations"] if os.path.isfile(img)]

    metadata_content = None
    if os.path.isfile(result["metadata"]):
        with open(result["metadata"], "r", encoding="utf-8") as f:
            metadata_content = json.load(f)

    detections_file = result["detections"] if os.path.isfile(result["detections"]) else None

    return output_video, viz_images or None, metadata_content, detections_file


_UI_TEXT = {
    "zh": {
        "title": "# Good Badminton — AI 羽毛球分析系统",
        "inputs": "### 输入",
        "video": "比赛视频",
        "template": "球场模板图像",
        "settings": "### 分析设置",
        "pose_family": "姿态模型",
        "pose_mode": "姿态模式",
        "language": "语言 / Language",
        "audio": "保留音频",
        "advanced": "高级选项",
        "skeletons": "显示骨架",
        "player_traj": "显示球员轨迹",
        "court_traj": "显示球场轨迹",
        "shuttle_traj": "显示羽毛球轨迹",
        "player_stats": "显示球员统计",
        "pose_roi": "显示姿态 ROI",
        "viz_positions": "生成热力图和散点图",
        "yolo_pose_path": "YOLO 姿态模型路径",
        "ball_path": "羽毛球检测模型路径",
        "step1": "### 第一步 — 球场检测",
        "detect_btn": "检测球场",
        "court_preview": "球场预览（点击标注角点）",
        "corner_status": "角点状态",
        "corner_none": "尚未检测到角点。",
        "apply_btn": "应用手动角点",
        "step2": "### 第二步 — 运行分析",
        "run_btn": "运行分析",
        "results": "### 结果",
        "out_video": "标注视频",
        "out_gallery": "热力图和散点图",
        "out_metadata": "元数据",
        "out_detections": "检测数据 (JSONL)",
        "auto_ok": "自动检测到 {} 个角点。",
        "auto_fail": "自动检测失败 — 请手动点击 4 个角点。",
        "manual_ok": "已应用手动角点（{} 个点）。",
        "manual_fail": "失败。",
    },
    "en": {
        "title": "# Good Badminton — AI Badminton Analysis",
        "inputs": "### Inputs",
        "video": "Match Video",
        "template": "Court Template Image",
        "settings": "### Analysis Settings",
        "pose_family": "Pose Model Family",
        "pose_mode": "Pose Mode",
        "language": "Language / 语言",
        "audio": "Keep Audio",
        "advanced": "Advanced Options",
        "skeletons": "Show Skeletons",
        "player_traj": "Show Player Trajectories",
        "court_traj": "Show Court Trajectory",
        "shuttle_traj": "Show Shuttlecock Trajectory",
        "player_stats": "Show Player Stats",
        "pose_roi": "Show Pose ROI",
        "viz_positions": "Generate Heatmaps & Scatter Plots",
        "yolo_pose_path": "YOLO Pose Model Path",
        "ball_path": "Shuttlecock Model Path",
        "step1": "### Step 1 — Court Detection",
        "detect_btn": "Detect Court",
        "court_preview": "Court Preview (click to annotate corners)",
        "corner_status": "Corner Status",
        "corner_none": "No corners detected yet.",
        "apply_btn": "Apply Manual Corners",
        "step2": "### Step 2 — Run Analysis",
        "run_btn": "Run Analysis",
        "results": "### Results",
        "out_video": "Annotated Video",
        "out_gallery": "Heatmaps & Scatter Plots",
        "out_metadata": "Metadata",
        "out_detections": "Detections (JSONL)",
        "auto_ok": "Auto-detected {} corners.",
        "auto_fail": "Auto-detection failed — click 4 corners manually.",
        "manual_ok": "Manual corners applied ({} points).",
        "manual_fail": "Failed.",
    },
}


def _switch_language(lang):
    t = _UI_TEXT.get(lang, _UI_TEXT["zh"])
    return [
        gr.update(value=t["title"]),
        gr.update(value=t["inputs"]),
        gr.update(label=t["video"]),
        gr.update(label=t["template"]),
        gr.update(value=t["settings"]),
        gr.update(label=t["pose_family"]),
        gr.update(label=t["pose_mode"]),
        gr.update(label=t["audio"]),
        gr.update(label=t["advanced"]),
        gr.update(label=t["skeletons"]),
        gr.update(label=t["player_traj"]),
        gr.update(label=t["court_traj"]),
        gr.update(label=t["shuttle_traj"]),
        gr.update(label=t["player_stats"]),
        gr.update(label=t["pose_roi"]),
        gr.update(label=t["viz_positions"]),
        gr.update(label=t["yolo_pose_path"]),
        gr.update(label=t["ball_path"]),
        gr.update(value=t["step1"]),
        gr.update(value=t["detect_btn"]),
        gr.update(label=t["court_preview"]),
        gr.update(label=t["corner_status"]),
        gr.update(value=t["apply_btn"]),
        gr.update(value=t["step2"]),
        gr.update(value=t["run_btn"]),
        gr.update(value=t["results"]),
        gr.update(label=t["out_video"]),
        gr.update(label=t["out_gallery"]),
        gr.update(label=t["out_metadata"]),
        gr.update(label=t["out_detections"]),
    ]


def build_ui():
    t = _UI_TEXT["zh"]

    with gr.Blocks(
        title="Good Badminton — AI Badminton Analysis",
    ) as demo:
        md_title = gr.Markdown(t["title"])

        corners_state = gr.State(value=None)
        click_corners_state = gr.State(value=[])

        with gr.Row():
            with gr.Column(scale=1):
                md_inputs = gr.Markdown(t["inputs"])
                video_input = gr.File(label=t["video"], file_types=["video"])
                template_input = gr.File(label=t["template"], file_types=["image"])

                md_settings = gr.Markdown(t["settings"])
                pose_family = gr.Dropdown(
                    choices=["yolo-pose", "rtmpose", "rtmo"],
                    value="yolo-pose", label=t["pose_family"],
                )
                pose_mode = gr.Dropdown(
                    choices=["lightweight", "balanced", "performance"],
                    value="balanced", label=t["pose_mode"],
                )
                language = gr.Radio(
                    choices=[("中文", "zh"), ("English", "en")],
                    value="zh", label=t["language"],
                )
                audio = gr.Checkbox(value=True, label=t["audio"])

                with gr.Accordion(t["advanced"], open=False) as adv_accordion:
                    show_skeletons = gr.Checkbox(value=True, label=t["skeletons"])
                    show_player_trajectories = gr.Checkbox(value=True, label=t["player_traj"])
                    show_court_trajectory = gr.Checkbox(value=True, label=t["court_traj"])
                    show_shuttlecock_trajectory = gr.Checkbox(value=True, label=t["shuttle_traj"])
                    show_player_stats = gr.Checkbox(value=True, label=t["player_stats"])
                    show_pose_roi = gr.Checkbox(value=True, label=t["pose_roi"])
                    visualize_positions = gr.Checkbox(value=True, label=t["viz_positions"])
                    yolo_pose_model = gr.Textbox(value="weights/yolo11n-pose.pt", label=t["yolo_pose_path"])
                    ball_model = gr.Textbox(value="weights/yolo11s-ball.pt", label=t["ball_path"])

            with gr.Column(scale=2):
                md_step1 = gr.Markdown(t["step1"])
                detect_btn = gr.Button(t["detect_btn"], variant="primary")
                court_image = gr.Image(label=t["court_preview"], interactive=False, type="numpy")
                corner_status = gr.Textbox(label=t["corner_status"], interactive=False, value=t["corner_none"])
                apply_btn = gr.Button(t["apply_btn"], variant="secondary")

                md_step2 = gr.Markdown(t["step2"])
                run_btn = gr.Button(t["run_btn"], variant="primary")

                md_results = gr.Markdown(t["results"])
                output_video = gr.Video(label=t["out_video"])
                output_gallery = gr.Gallery(label=t["out_gallery"], columns=2, height="auto")
                output_metadata = gr.JSON(label=t["out_metadata"])
                output_detections = gr.File(label=t["out_detections"])

        lang_outputs = [
            md_title, md_inputs, video_input, template_input,
            md_settings, pose_family, pose_mode, audio, adv_accordion,
            show_skeletons, show_player_trajectories, show_court_trajectory,
            show_shuttlecock_trajectory, show_player_stats, show_pose_roi,
            visualize_positions, yolo_pose_model, ball_model,
            md_step1, detect_btn, court_image, corner_status, apply_btn,
            md_step2, run_btn, md_results,
            output_video, output_gallery, output_metadata, output_detections,
        ]
        language.change(fn=_switch_language, inputs=[language], outputs=lang_outputs)

        detect_btn.click(
            fn=detect_court,
            inputs=[template_input],
            outputs=[court_image, corners_state],
        ).then(
            fn=lambda c, lang: _UI_TEXT.get(lang, _UI_TEXT["zh"])["auto_ok"].format(len(c)) if c
               else _UI_TEXT.get(lang, _UI_TEXT["zh"])["auto_fail"],
            inputs=[corners_state, language],
            outputs=[corner_status],
        )

        court_image.select(
            fn=on_court_image_select,
            inputs=[click_corners_state, template_input],
            outputs=[court_image, click_corners_state, corners_state, corner_status],
        )

        apply_btn.click(
            fn=apply_manual_corners,
            inputs=[template_input, click_corners_state],
            outputs=[court_image, corners_state],
        ).then(
            fn=lambda c, lang: _UI_TEXT.get(lang, _UI_TEXT["zh"])["manual_ok"].format(len(c)) if c
               else _UI_TEXT.get(lang, _UI_TEXT["zh"])["manual_fail"],
            inputs=[corners_state, language],
            outputs=[corner_status],
        )

        run_btn.click(
            fn=run_full_analysis,
            inputs=[
                video_input, template_input, corners_state,
                pose_family, pose_mode, language, audio,
                show_skeletons, show_player_trajectories,
                show_court_trajectory, show_shuttlecock_trajectory,
                show_player_stats, show_pose_roi, visualize_positions,
                yolo_pose_model, ball_model,
            ],
            outputs=[output_video, output_gallery, output_metadata, output_detections],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue(default_concurrency_limit=1).launch(theme=gr.themes.Soft())
