from typing import Dict

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from cache import PersistentCache, create_cache_zip
from interview_simulator import InterviewSimulator

import os


app = Flask(__name__, static_folder=os.environ.get("FRONTEND_BUILD", "frontend/build"))
CORS(app)

cache = PersistentCache(cache_dir=os.environ.get("CACHE_DIR", "cache_bangla"), language="bn")
sessions: Dict[str, InterviewSimulator] = {}


@app.route("/api/start-interview", methods=["POST"])
def start_interview():
    try:
        data = request.get_json() or {}
        patient_id = data.get("patient_id")
        patient_data = data.get("patient_data")

        if not patient_id or not patient_data:
            return jsonify({"error": "Missing required data"}), 400

        simulator = InterviewSimulator(patient_data, cache)
        sessions[patient_id] = simulator

        response = simulator.start_interview()
        return jsonify({"success": True, "language": "bn", **response})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/send-message", methods=["POST"])
def send_message():
    try:
        data = request.get_json() or {}
        patient_id = data.get("patient_id")
        message = data.get("message")

        if not patient_id or patient_id not in sessions:
            return jsonify({"error": "Session not found"}), 404

        simulator = sessions[patient_id]
        response = simulator.process_user_response(message)
        return jsonify({"success": True, "language": "bn", **response})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/generate-report", methods=["POST"])
def generate_report():
    try:
        data = request.get_json() or {}
        patient_id = data.get("patient_id")

        if not patient_id or patient_id not in sessions:
            return jsonify({"error": "Session not found"}), 404

        simulator = sessions[patient_id]
        report = simulator.generate_report()
        transcript = simulator.get_transcript()

        return jsonify({
            "success": True,
            "language": "bn",
            "report": report,
            "transcript": transcript,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/get-transcript", methods=["POST"])
def get_transcript():
    try:
        data = request.get_json() or {}
        patient_id = data.get("patient_id")

        if not patient_id or patient_id not in sessions:
            return jsonify({"error": "Session not found"}), 404

        simulator = sessions[patient_id]
        transcript = simulator.get_transcript()

        return jsonify({"success": True, "language": "bn", "transcript": transcript})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/cache-stats", methods=["GET"])
def cache_stats():
    try:
        stats = cache.get_stats()
        return jsonify({"success": True, "language": "bn", "stats": stats})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/download-cache", methods=["GET"])
def download_cache():
    archive_path, error = create_cache_zip()
    if error:
        return jsonify({"error": error}), 500
    if not archive_path:
        return jsonify({"error": "Cache archive not available"}), 404
    return send_from_directory(os.path.dirname(archive_path), os.path.basename(archive_path), as_attachment=True)


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path: str):
    static_dir = app.static_folder or "frontend/build"
    target = os.path.join(static_dir, path)
    if path and os.path.exists(target):
        return send_from_directory(static_dir, path)
    return send_from_directory(static_dir, "index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
