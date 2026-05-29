from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request, url_for

try:
    import oracledb
    try:
        oracledb.init_oracle_client(
            lib_dir=r"C:\oraclexe\app\oracle\product\11.2.0\server\bin"
        )
    except Exception:
        pass
except ImportError:
    oracledb = None

from utils.predictor import get_chart_data, predict_existing_member, predict_manual_input


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_AS_ASCII"] = False

    @app.context_processor
    def inject_globals():
        return {"project_name": "RoadSafe"}

    @app.get("/")
    def home():
        return render_template("home.html")

    @app.get("/existing")
    def existing_predict():
        return render_template("existing_predict.html")

    @app.get("/direct")
    def direct_predict():
        return render_template("direct_predict.html")

    @app.get("/helmet")
    def helmet_page():
        return render_template("helmet.html")

    @app.get("/llm")
    def llm_page():
        return render_template("llm.html")

    @app.get("/api/helmet/samples")
    def api_helmet_samples():
        """헬멧 탐지 페이지에 표시할 예시 이미지/영상 목록."""
        sample_root = Path(app.static_folder) / "samples" / "helmet"
        images_dir = sample_root / "images"
        videos_dir = sample_root / "videos"
        thumbs_dir = sample_root / "video_thumbnails"

        samples = []
        for path in sorted(images_dir.glob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                rel = f"samples/helmet/images/{path.name}"
                samples.append({
                    "type": "image",
                    "label": "예시 이미지",
                    "url": url_for("static", filename=rel),
                    "thumbnail_url": url_for("static", filename=rel),
                })

        for path in sorted(videos_dir.glob("*")):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
                thumb = thumbs_dir / f"{path.stem}.jpg"
                samples.append({
                    "type": "video",
                    "label": "예시 영상",
                    "url": url_for("static", filename=f"samples/helmet/videos/{path.name}"),
                    "thumbnail_url": url_for("static", filename=f"samples/helmet/video_thumbnails/{thumb.name}") if thumb.exists() else None,
                })

        return jsonify({"ok": True, "samples": samples})

    @app.get("/api/charts/overview")
    def api_charts_overview():
        try:
            return jsonify({"ok": True, "charts": get_chart_data()})
        except Exception as exc:
            return jsonify({"ok": False, "message": str(exc)}), 500

    @app.post("/api/predict/existing")
    def api_predict_existing():
        payload = request.get_json(silent=True) or {}
        member_no = str(payload.get("member_no", "")).strip()
        if not member_no:
            return jsonify({"ok": False, "message": "회원 번호를 입력해 주시기 바랍니다."}), 400
        try:
            result = predict_existing_member(member_no)
            return jsonify({"ok": True, "result": result})
        except Exception as exc:
            return jsonify({"ok": False, "message": str(exc)}), 500

    @app.post("/api/predict/direct")
    def api_predict_direct():
        payload = request.get_json(silent=True) or {}
        try:
            result = predict_manual_input(payload)
            return jsonify({"ok": True, "result": result})
        except Exception as exc:
            return jsonify({"ok": False, "message": str(exc)}), 500

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
