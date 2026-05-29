
from __future__ import annotations

import gc
import math
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import cv2
import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from ultralytics import YOLO
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "yolo_models" / "rider_helmet_yolo11s_v6_best.pt"
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

CONF_THRESHOLD = 0.40
# 영상에서 1프레임만 잠깐 튀는 오탐을 줄이기 위한 후처리 기준
# 같은 클래스의 박스가 인접 프레임에서 한 번 더 확인될 때만 영상에 표시합니다.
VIDEO_STABLE_IOU_THRESHOLD = 0.12
VIDEO_STABLE_CENTER_RATIO = 0.85
IMAGE_SIZE = 640
VIDEO_SIZE = 480
MAX_VIDEO_SECONDS = 10
NORMALIZED_VIDEO_FPS = 6
GIF_SECONDS = 3.0
GIF_FPS = 5
GIF_MAX_WIDTH = 520

CLASS_LABELS = {
    0: "rider_no_helmet",
    1: "rider_helmet",
}

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

model = YOLO(str(MODEL_PATH)) if MODEL_PATH.exists() else None


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "service": "YOLO11 helmet detection API",
        "model_loaded": model is not None,
    })


def model_info(kind: str) -> dict:
    return {
        "model_name": "YOLO11s",
        "conf_threshold": CONF_THRESHOLD,
        "image_size": IMAGE_SIZE if kind == "image" else VIDEO_SIZE,
        "classes": ["rider_no_helmet", "rider_helmet"],
    }


def resize_keep_ratio(frame, max_width: int):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    new_h = int(h * max_width / w)
    return cv2.resize(frame, (max_width, new_h), interpolation=cv2.INTER_AREA)


def ensure_even_frame(frame):
    h, w = frame.shape[:2]
    even_h = h - (h % 2)
    even_w = w - (w % 2)
    if even_h != h or even_w != w:
        frame = frame[:even_h, :even_w]
    return frame


def extract_detections(result) -> list[dict]:
    detections = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections

    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls_id in zip(xyxy, confs, class_ids):
        if int(cls_id) not in CLASS_LABELS:
            continue
        detections.append({
            "class_id": int(cls_id),
            "class_name": CLASS_LABELS[int(cls_id)],
            "confidence": round(float(conf), 4),
            "box": [round(float(v), 2) for v in box.tolist()],
        })
    return detections


def summarize_detections(detections: list[dict]) -> dict:
    helmet = sum(1 for d in detections if d["class_id"] == 1)
    no_helmet = sum(1 for d in detections if d["class_id"] == 0)
    confs = [float(d["confidence"]) for d in detections]
    avg_conf = round(sum(confs) / len(confs) * 100, 2) if confs else None
    max_conf = round(max(confs) * 100, 2) if confs else None
    return {
        "rider_helmet": helmet,
        "rider_no_helmet": no_helmet,
        "total_riders": helmet + no_helmet,
        "avg_confidence": avg_conf,
        "max_confidence": max_conf,
    }




def box_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def box_center_distance_ratio(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]

    acx = (ax1 + ax2) / 2
    acy = (ay1 + ay2) / 2
    bcx = (bx1 + bx2) / 2
    bcy = (by1 + by2) / 2

    dist = math.hypot(acx - bcx, acy - bcy)
    scale = max(ax2 - ax1, ay2 - ay1, bx2 - bx1, by2 - by1, 1.0)
    return dist / scale


def has_temporal_support(det: dict, neighbor_detections: list[dict]) -> bool:
    """Return True when the same class appears close enough in an adjacent frame."""
    for other in neighbor_detections:
        if det["class_id"] != other["class_id"]:
            continue

        iou = box_iou(det["box"], other["box"])
        center_ratio = box_center_distance_ratio(det["box"], other["box"])
        if iou >= VIDEO_STABLE_IOU_THRESHOLD or center_ratio <= VIDEO_STABLE_CENTER_RATIO:
            return True
    return False


def filter_stable_video_detections(current: list[dict], neighbor_groups: list[list[dict]]) -> list[dict]:
    """Remove detections that appear in only one isolated frame.

    This is a general video post-processing rule, not a coordinate-based exception.
    A box is kept only when a similar same-class box also appears in the previous
    or next frame.
    """
    neighbors = [det for group in neighbor_groups for det in group]
    if not neighbors:
        return []
    return [det for det in current if has_temporal_support(det, neighbors)]


def draw_detections(frame, detections: list[dict]):
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["box"]]
        class_id = int(det["class_id"])
        class_name = det["class_name"]
        confidence = float(det["confidence"])

        color = (0, 215, 255) if class_id == 1 else (0, 90, 255)
        thickness = max(2, int(round(min(frame.shape[:2]) / 260)))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        label = f"{class_name} {confidence:.2f}"
        font_scale = max(0.45, min(frame.shape[:2]) / 900)
        font_thickness = max(1, thickness - 1)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        label_y1 = max(0, y1 - th - baseline - 6)
        label_y2 = max(th + baseline + 6, y1)
        cv2.rectangle(annotated, (x1, label_y1), (x1 + tw + 8, label_y2), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 4, label_y2 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (10, 15, 20),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    return annotated

def build_analysis_reasons(summary: dict, kind: str) -> list[str]:
    reasons = []
    avg_conf = summary.get("avg_confidence")
    max_conf = summary.get("max_confidence")

    if kind == "image":
        if not summary.get("total_riders"):
            return ["탐지 기준을 만족하는 라이더가 확인되지 않았습니다."]
        if avg_conf is not None and avg_conf < 65:
            reasons.append("라이더 또는 머리 영역이 작거나 흐릿해 평균 탐지 신뢰도가 낮아질 수 있습니다.")
        elif avg_conf is not None and avg_conf < 80:
            reasons.append("일부 영역은 선명하지만 자세, 거리, 가림 정도에 따라 신뢰도가 달라질 수 있습니다.")
        else:
            reasons.append("라이더와 헬멧 영역이 비교적 명확하게 보여 안정적인 탐지가 이루어졌습니다.")
    else:
        processed = summary.get("processed_frames") or 0
        detected = summary.get("detection_frames") or 0
        ratio = detected / processed if processed else 0
        if processed == 0:
            return ["분석 가능한 프레임을 확보하지 못했습니다."]
        if ratio == 0:
            reasons.append("이번 구간에서는 탐지 기준을 만족하는 라이더 박스가 확인되지 않았습니다.")
        elif ratio < 0.4:
            reasons.append("일부 프레임에서만 라이더가 안정적으로 감지되어 탐지 프레임 비율이 낮게 나타났습니다.")
        else:
            reasons.append("여러 프레임에서 라이더가 반복적으로 감지되어 프레임별 분석 요약을 생성했습니다.")
        if avg_conf is not None and avg_conf < 65:
            reasons.append("라이더가 멀리 있거나 측면/뒷모습에 가까운 프레임에서는 머리·헬멧 영역이 작게 보여 평균 신뢰도가 낮아질 수 있습니다.")
        elif avg_conf is not None and avg_conf < 80:
            reasons.append("움직임, 각도, 프레임 선명도 차이 때문에 평균 신뢰도는 중간 수준으로 나타날 수 있습니다.")
        if max_conf is not None and avg_conf is not None and max_conf - avg_conf >= 12:
            reasons.append("선명한 구간에서는 더 높은 신뢰도로 탐지된 프레임이 확인됩니다.")
        if summary.get("rider_no_helmet_detections", 0) > summary.get("rider_helmet_detections", 0):
            reasons.append("헬멧 착용 영역이 명확히 보이지 않는 프레임들이 미착용 후보로 더 많이 집계되었습니다.")

    return reasons


def make_result_url(path: Path) -> str:
    return f"/api/helmet/results/{path.name}?v={int(time.time() * 1000)}"


def normalize_video_for_analysis(upload_path: Path, run_id: str) -> Path:
    """Convert uploaded/sample videos to a lightweight MP4 before frame analysis.

    The output is optimized for OpenCV decoding, not for direct browser playback.
    We intentionally avoid libx264 here because some low-memory Windows setups fail
    while opening the x264 encoder. The web preview is provided as GIF, and the full
    result file is only offered as a download.
    """
    normalized_path = RESULT_DIR / f"{run_id}_normalized.mp4"
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    scale_filter = "scale='if(gt(iw,ih),480,-2)':'if(gt(iw,ih),-2,480)',fps=6"
    cmd = [
        ffmpeg_exe,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(upload_path),
        "-t",
        str(MAX_VIDEO_SECONDS),
        "-vf",
        scale_filter,
        "-an",
        "-c:v",
        "mpeg4",
        "-q:v",
        "5",
        "-pix_fmt",
        "yuv420p",
        str(normalized_path),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0 or not normalized_path.exists() or normalized_path.stat().st_size == 0:
        detail = completed.stderr.strip() or "영상 변환 실패"
        raise RuntimeError(f"영상 전처리에 실패했습니다. {detail}")
    return normalized_path


def create_cv2_video_writer(path: Path, fps: float, frame_shape) -> cv2.VideoWriter | None:
    """Create a lightweight MP4 writer with mp4v.

    This avoids x264 memory allocation failures. The browser preview still uses GIF,
    so the MP4 is mainly for download/checking the full result.
    """
    h, w = frame_shape[:2]
    w = w - (w % 2)
    h = h - (h % 2)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, max(1.0, float(fps)), (w, h))
    if not writer.isOpened():
        writer.release()
        return None
    return writer

def save_preview_gif(records: list[dict], output_path: Path, output_fps: float) -> bool:
    if not records:
        return False

    detected_records = [r for r in records if r.get("has_detection")]
    if detected_records:
        center_index = detected_records[len(detected_records) // 2]["index"]
    else:
        center_index = records[len(records) // 2]["index"]

    span = max(2, int(output_fps * GIF_SECONDS))
    start = center_index - span // 2
    end = center_index + span // 2
    selected = [r for r in records if start <= r["index"] <= end]

    if len(selected) < 2:
        mid = len(records) // 2
        half = max(2, span // 2)
        selected = records[max(0, mid - half): min(len(records), mid + half)]

    max_gif_frames = max(2, int(GIF_SECONDS * GIF_FPS))
    step = max(1, math.ceil(len(selected) / max_gif_frames))
    frames = [r["frame_rgb"] for r in selected[::step]]

    if len(frames) == 1:
        frames = frames * 2
    if not frames:
        return False

    imageio.mimsave(str(output_path), frames, duration=1 / GIF_FPS, loop=0)
    return output_path.exists() and output_path.stat().st_size > 0


def analyze_image(upload_path: Path, run_id: str):
    if model is None:
        raise RuntimeError("모델 파일을 찾을 수 없습니다.")

    results = model.predict(
        source=str(upload_path),
        imgsz=IMAGE_SIZE,
        conf=CONF_THRESHOLD,
        classes=[0, 1],
        verbose=False,
    )
    result = results[0]
    detections = extract_detections(result)
    annotated = result.plot()

    result_path = RESULT_DIR / f"{run_id}_image_result.jpg"
    cv2.imwrite(str(result_path), annotated)

    summary = summarize_detections(detections)
    reasons = build_analysis_reasons(summary, "image")

    return {
        "ok": True,
        "type": "image",
        "result_url": make_result_url(result_path),
        "detections": detections,
        "summary": summary,
        "analysis_reasons": reasons,
        "model_info": model_info("image"),
    }


def analyze_video(upload_path: Path, run_id: str):
    if model is None:
        raise RuntimeError("모델 파일을 찾을 수 없습니다.")

    normalized_path = normalize_video_for_analysis(upload_path, run_id)
    cap = cv2.VideoCapture(str(normalized_path))
    if not cap.isOpened():
        raise RuntimeError("전처리된 영상을 읽을 수 없습니다.")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or NORMALIZED_VIDEO_FPS)
    if source_fps <= 1:
        source_fps = NORMALIZED_VIDEO_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = total_frames if total_frames > 0 else int(source_fps * MAX_VIDEO_SECONDS)
    max_frames = min(max_frames, int(source_fps * MAX_VIDEO_SECONDS))
    output_fps = max(4.0, min(8.0, source_fps))

    result_video_path = RESULT_DIR / f"{run_id}_video_result.mp4"
    preview_gif_path = RESULT_DIR / f"{run_id}_preview.gif"
    representative_path = RESULT_DIR / f"{run_id}_representative.jpg"

    video_writer = None
    preview_records: list[dict] = []
    representative_candidate = None
    representative_conf = -1.0

    processed_frames = 0
    detection_frames = 0
    helmet_detections = 0
    no_helmet_detections = 0
    all_confidences = []
    raw_detection_frames = 0
    raw_detection_count = 0
    filtered_detection_count = 0

    frame_buffer: list[dict] = []
    output_index = 0

    def finalize_record(record: dict, prev_detections: list[dict], next_detections: list[dict]):
        nonlocal video_writer, representative_candidate, representative_conf
        nonlocal detection_frames, helmet_detections, no_helmet_detections
        nonlocal output_index, filtered_detection_count

        stable_detections = filter_stable_video_detections(
            record["detections"],
            [prev_detections, next_detections],
        )
        filtered_detection_count += max(0, len(record["detections"]) - len(stable_detections))

        annotated = draw_detections(record["frame"], stable_detections)
        annotated = resize_keep_ratio(annotated, VIDEO_SIZE)
        annotated = ensure_even_frame(annotated)

        if video_writer is None:
            video_writer = create_cv2_video_writer(result_video_path, output_fps, annotated.shape)

        if video_writer is not None:
            video_writer.write(annotated)

        current_max_conf = max([float(d["confidence"]) for d in stable_detections], default=0.0)
        if stable_detections:
            detection_frames += 1
            if current_max_conf > representative_conf:
                representative_candidate = annotated.copy()
                representative_conf = current_max_conf

        for det in stable_detections:
            all_confidences.append(float(det["confidence"]))
            if det["class_id"] == 1:
                helmet_detections += 1
            elif det["class_id"] == 0:
                no_helmet_detections += 1

        gif_bgr = resize_keep_ratio(annotated, GIF_MAX_WIDTH)
        gif_rgb = cv2.cvtColor(gif_bgr, cv2.COLOR_BGR2RGB)
        preview_records.append({
            "index": output_index,
            "has_detection": bool(stable_detections),
            "frame_rgb": gif_rgb,
        })
        output_index += 1

    try:
        while processed_frames < max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            frame = resize_keep_ratio(frame, VIDEO_SIZE)
            results = model.predict(
                source=frame,
                imgsz=VIDEO_SIZE,
                conf=CONF_THRESHOLD,
                classes=[0, 1],
                verbose=False,
            )
            result = results[0]
            detections = extract_detections(result)

            if detections:
                raw_detection_frames += 1
                raw_detection_count += len(detections)

            frame_buffer.append({
                "frame": frame.copy(),
                "detections": detections,
            })

            # 세 프레임이 모이면 가운데 프레임을 확정합니다.
            # 가운데 프레임은 이전/다음 프레임을 둘 다 볼 수 있어 1프레임 오탐 제거가 가능합니다.
            if len(frame_buffer) >= 3:
                finalize_record(
                    frame_buffer[1],
                    frame_buffer[0]["detections"],
                    frame_buffer[2]["detections"],
                )
                frame_buffer.pop(0)

            processed_frames += 1
            if processed_frames % 6 == 0:
                gc.collect()
    finally:
        cap.release()

        # 남은 첫/끝 프레임은 가능한 이웃 프레임만 보고 확정합니다.
        if frame_buffer:
            for i, record in enumerate(frame_buffer):
                prev_detections = frame_buffer[i - 1]["detections"] if i > 0 else []
                next_detections = frame_buffer[i + 1]["detections"] if i + 1 < len(frame_buffer) else []
                finalize_record(record, prev_detections, next_detections)

        if video_writer is not None:
            video_writer.release()

    if representative_candidate is not None:
        cv2.imwrite(str(representative_path), representative_candidate)
    elif preview_records:
        mid_rgb = preview_records[len(preview_records) // 2]["frame_rgb"]
        cv2.imwrite(str(representative_path), cv2.cvtColor(mid_rgb, cv2.COLOR_RGB2BGR))

    gif_ok = save_preview_gif(preview_records, preview_gif_path, output_fps)

    avg_conf = round(sum(all_confidences) / len(all_confidences) * 100, 2) if all_confidences else None
    max_conf = round(max(all_confidences) * 100, 2) if all_confidences else None

    summary = {
        "processed_frames": processed_frames,
        "detection_frames": detection_frames,
        "rider_helmet_detections": helmet_detections,
        "rider_no_helmet_detections": no_helmet_detections,
        "avg_confidence": avg_conf,
        "max_confidence": max_conf,
        "raw_detection_frames": raw_detection_frames,
        "raw_detection_count": raw_detection_count,
        "filtered_detection_count": filtered_detection_count,
        "temporal_filter": "same-class adjacent-frame stability filter",
    }
    reasons = build_analysis_reasons(summary, "video")

    has_result_video = result_video_path.exists() and result_video_path.stat().st_size > 0
    return {
        "ok": True,
        "type": "video",
        "preview_gif_url": make_result_url(preview_gif_path) if gif_ok else None,
        "representative_image_url": make_result_url(representative_path) if representative_path.exists() else None,
        "result_video_url": make_result_url(result_video_path) if has_result_video else None,
        "download_url": f"/api/helmet/download/{result_video_path.name}" if has_result_video else None,
        "summary": summary,
        "analysis_reasons": reasons,
        "model_info": model_info("video"),
    }


@app.route("/api/helmet/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"ok": True})

    if "file" not in request.files:
        return jsonify({"ok": False, "message": "파일이 업로드되지 않았습니다."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"ok": False, "message": "파일명이 비어 있습니다."}), 400

    original_name = secure_filename(file.filename)
    ext = Path(original_name).suffix.lower()
    if ext not in IMAGE_EXTS and ext not in VIDEO_EXTS:
        return jsonify({"ok": False, "message": "지원하지 않는 파일 형식입니다."}), 400

    run_id = uuid4().hex[:12]
    upload_path = UPLOAD_DIR / f"{run_id}_{original_name}"
    file.save(str(upload_path))

    try:
        if ext in IMAGE_EXTS:
            return jsonify(analyze_image(upload_path, run_id))
        return jsonify(analyze_video(upload_path, run_id))
    except Exception as exc:
        return jsonify({"ok": False, "message": "분석 중 오류가 발생했습니다.", "error": str(exc)}), 500


@app.get("/api/helmet/results/<path:filename>")
def result_file(filename):
    return send_from_directory(RESULT_DIR, filename)


@app.get("/api/helmet/download/<path:filename>")
def download_file(filename):
    return send_from_directory(RESULT_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
