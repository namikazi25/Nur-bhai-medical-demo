from typing import Dict

from flask import Flask, jsonify, request, send_from_directory, Response, stream_with_context
from flask_cors import CORS

from cache import PersistentCache, create_cache_zip
from interview_simulator import InterviewSimulator

import os
import json


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


def _resolve_patient_data(name: str, condition: str) -> dict:
    """Best-effort resolve of patient attributes from frontend assets; falls back to defaults."""
    static_dir = app.static_folder or "frontend/build"
    asset_path = os.path.join(static_dir, "assets", "patients_and_conditions.json")
    try:
        if os.path.isfile(asset_path):
            with open(asset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for p in data.get("patients", []):
                    if p.get("name") == name:
                        # Map available fields to our expected structure
                        return {
                            "name": p.get("name", name),
                            "age": p.get("age", 35),
                            "gender": p.get("gender", "পুরুষ"),
                            "condition": condition,
                        }
    except Exception:
        pass
    # Fallback defaults
    return {"name": name or "Patient", "age": 35, "gender": "পুরুষ", "condition": condition}


@app.route("/api/stream_conversation", methods=["GET"])
def stream_conversation():
    """Compatibility SSE endpoint for the existing frontend. Streams the Bangla interview."""

    patient = request.args.get("patient", "Patient")
    condition = request.args.get("condition", "unknown condition")

    def generate():
        try:
            simulator = InterviewSimulator(_resolve_patient_data(patient, condition), cache)
            # Start interview (assistant opens)
            start = simulator.start_interview()
            assistant_msg = start.get("message", "")
            first = json.dumps({"role": "assistant", "text": assistant_msg}, ensure_ascii=False)
            yield f"data: {first}\n\n"

            turns = 0
            complete = start.get("complete", False)
            while not complete and turns < 20:
                step = simulator.process_user_response(assistant_msg)
                conv = step.get("conversation", [])

                # Stream patient's latest reply if present
                if conv:
                    # Find last 'user' message in conversation
                    for m in reversed(conv):
                        if m.get("role") == "user":
                            patient_event = json.dumps({"role": "patient", "text": m.get("content", "")}, ensure_ascii=False)
                            yield f"data: {patient_event}\n\n"
                            break

                # Stream assistant's follow-up question
                assistant_msg = step.get("message", "")
                assistant_event = json.dumps({"role": "assistant", "text": assistant_msg}, ensure_ascii=False)
                yield f"data: {assistant_event}\n\n"

                complete = step.get("complete", False)
                turns += 1

            # Signal completion to client
            yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"
        except Exception as e:
            err = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {err}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


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
