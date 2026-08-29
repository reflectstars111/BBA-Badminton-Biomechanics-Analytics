import os
import shutil
import subprocess
import time

import cv2
import numpy as np

from badminton_analysis.court.mapper import (
    CourtMapper,
    auto_detect_preview,
    compute_expanded_roi,
    resolve_court_corners,
)
from badminton_analysis.system import BadmintonAnalysisSystem, load_runtime_dependencies

_MAX_WEBUI_OUTPUTS = 10

_dependencies_loaded = False


def imread_safe(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread with fallback for Unicode paths on Windows."""
    img = cv2.imread(path, flags)
    if img is None and os.path.isfile(path):
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, flags)
        except Exception:
            pass
    return img


def _cleanup_old_outputs(base_dir="outputs", prefix="webui_", keep=_MAX_WEBUI_OUTPUTS):
    """Remove oldest webui output directories beyond *keep* count."""
    if not os.path.isdir(base_dir):
        return
    dirs = []
    for name in os.listdir(base_dir):
        if name.startswith(prefix):
            full = os.path.join(base_dir, name)
            if os.path.isdir(full):
                dirs.append((os.path.getmtime(full), full))
    dirs.sort(reverse=True)
    for _, path in dirs[keep:]:
        try:
            shutil.rmtree(path)
        except Exception:
            pass


def _ensure_dependencies():
    global _dependencies_loaded
    if not _dependencies_loaded:
        load_runtime_dependencies()
        _dependencies_loaded = True


def prepare_court(template_path, manual_corners=None):
    """Detect (or apply manual) court corners and return results + preview.

    Returns:
        dict with keys: corners, roi_corners, mid_height, preview_bgr.
        On failure corners/roi_corners/mid_height are None.
    """
    _ensure_dependencies()
    template_color = imread_safe(template_path)
    if template_color is None:
        raise FileNotFoundError(f"Cannot read template image: {template_path}")

    if manual_corners and len(manual_corners) == 4:
        corners, roi_corners, mid_height = resolve_court_corners(
            template_color, manual_corners=manual_corners
        )
        preview = template_color.copy()
        if corners:
            pts = [list(c) for c in corners]
            cv2.polylines(preview, [np.array(pts, dtype=np.int32)], True, (0, 255, 0), 3)
            for idx, pt in enumerate(corners, 1):
                cv2.circle(preview, pt, 6, (0, 0, 255), -1)
                cv2.putText(preview, str(idx), (pt[0] + 8, pt[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        corners_auto, preview = auto_detect_preview(template_color)
        if preview is not None:
            h, w = template_color.shape[:2]
            preview = cv2.resize(preview, (w, h))
        if corners_auto:
            corners, roi_corners, mid_height = resolve_court_corners(
                template_color, manual_corners=corners_auto
            )
        else:
            corners, roi_corners, mid_height = None, None, None

    return {
        "corners": corners,
        "roi_corners": roi_corners,
        "mid_height": mid_height,
        "preview_bgr": preview,
    }


def _find_ffmpeg():
    """Locate an ffmpeg binary — prefer imageio_ffmpeg (bundled with moviepy)."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    path = shutil.which("ffmpeg")
    if path:
        return path
    return None


def _reencode_for_browser(video_path, output_dir):
    """Re-encode video to H.264 so browsers can play it.

    OpenCV's mp4v codec isn't browser-compatible.  This converts to H.264
    via ffmpeg.  Returns the path of the web-friendly file (or the original
    if ffmpeg is unavailable).
    """
    if not os.path.isfile(video_path):
        return video_path

    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        return video_path

    web_path = os.path.join(output_dir, "web_" + os.path.basename(video_path))
    try:
        subprocess.run(
            [
                ffmpeg, "-y",
                "-i", video_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                "-movflags", "faststart",
                web_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300,
            check=True,
        )
        if os.path.isfile(web_path) and os.path.getsize(web_path) > 0:
            return web_path
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return video_path


def _scale_corners_to_video(corners, template_path, video_path):
    """Scale court corners from template resolution to video frame resolution."""
    template_img = imread_safe(template_path)
    if template_img is None:
        return corners

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return corners
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    tmpl_h, tmpl_w = template_img.shape[:2]
    if tmpl_w == frame_w and tmpl_h == frame_h:
        return corners

    sx = frame_w / tmpl_w
    sy = frame_h / tmpl_h
    return [(int(x * sx), int(y * sy)) for x, y in corners]


def run_analysis(video_path, template_path, corners, options, progress_cb=None):
    """Run the full analysis pipeline headlessly.

    Args:
        video_path: Path to the input video file.
        template_path: Path to the court template image.
        corners: List of 4 (x, y) court corner tuples (template resolution).
        options: dict of analysis options (mirrors CLI flags).
        progress_cb: Optional callable(frame_count, total_frames).

    Returns:
        dict with output file paths.
    """
    _ensure_dependencies()
    _cleanup_old_outputs()

    corners = _scale_corners_to_video(corners, template_path, video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    roi_corners = compute_expanded_roi(corners, (frame_h, frame_w, 3))
    mapper = CourtMapper(corners)
    mid_height = mapper.mid_height

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("outputs", f"webui_{video_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "court_annotations.txt"), "w") as f:
        f.write(f"corners={corners}\n")
        f.write(f"roi_corners={roi_corners}\n")
        f.write(f"mid_height={mid_height}\n")

    language = options.get("language", "zh")
    pose_family = options.get("pose_family", "yolo-pose")
    pose_mode = options.get("pose_mode", "balanced")
    yolo_pose_model = options.get("yolo_pose_model", "weights/yolo11n-pose.pt")
    ball_model = options.get("ball_model", "weights/yolo11s-ball.pt")
    keep_audio = options.get("audio", True)
    show_skeletons = options.get("show_skeletons", True)
    show_player_trajectories = options.get("show_player_trajectories", True)
    show_court_trajectory = options.get("show_court_trajectory", True)
    show_shuttlecock_trajectory = options.get("show_shuttlecock_trajectory", True)
    show_player_stats = options.get("show_player_stats", True)
    show_pose_roi = options.get("show_pose_roi", True)
    visualize_positions = options.get("visualize_positions", True)

    system = BadmintonAnalysisSystem(
        video_path,
        show_display=False,
        show_skeletons=show_skeletons,
        show_player_trajectories=show_player_trajectories,
        show_court_trajectory=show_court_trajectory,
        show_shuttlecock_trajectory=show_shuttlecock_trajectory,
        show_player_stats=show_player_stats,
        show_performance_stats=False,
        save_images=False,
        language=language,
        output_dir=output_dir,
        ball_model_path=ball_model,
        template_path=template_path,
        pose_mode=pose_mode,
        pose_family=pose_family,
        yolo_pose_model=yolo_pose_model,
        show_pose_roi=show_pose_roi,
    )
    system.keep_audio = keep_audio
    system.process_video(progress_callback=progress_cb)

    if visualize_positions:
        if language == "en":
            from badminton_analysis.visualization.player_positions_en import analyze_player_positions
        else:
            from badminton_analysis.visualization.player_positions_zh import analyze_player_positions
        vis_dir = os.path.join(output_dir, "position_visualizations")
        analyze_player_positions(system.detections_path, vis_dir, fps=system.fps)

    web_video_path = _reencode_for_browser(system.output_video_path, output_dir)

    result = {
        "output_dir": output_dir,
        "video": web_video_path,
        "metadata": system.metadata_path,
        "detections": system.detections_path,
        "visualizations": [],
    }

    vis_dir = os.path.join(output_dir, "position_visualizations")
    if os.path.isdir(vis_dir):
        for root, _dirs, files in os.walk(vis_dir):
            for fname in sorted(files):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    result["visualizations"].append(os.path.join(root, fname))

    return result
