"""Flask entrypoint that exposes encoder/decoder endpoints and serves the UI."""

from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, render_template, request

from engine.decoder import run_analysis
from engine.encoder import (
    as_data_url,
    encode_multi_channel,
    encode_payload,
    normalize_output_format,
)
from engine.tooling import get_tool_status

app = Flask(__name__, static_folder="static", template_folder="templates")


def sniff_image_mime(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    return "application/octet-stream"


def _form_flag(value: Optional[str]) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.get("/api/tools")
def api_tools():
    try:
        return jsonify({"tools": get_tool_status()})
    except Exception as exc:
        return jsonify({"error": f"Failed to get tool status: {str(exc)}"}), 500


@app.post("/api/encode")
def api_encode():
    try:
        mode = request.form.get("mode", "text")
        plane = request.form.get("plane", "RGB")
        payload_text: Optional[str] = request.form.get("text") or None
        channels_json = request.form.get("channels")

        payload_file = request.files.get("payload")
        image_file = request.files.get("image")

        if image_file is None:
            return jsonify({"error": "Image file is required"}), 400

        if not image_file.filename:
            return jsonify({"error": "Image file must have a filename"}), 400

        try:
            image_bytes = image_file.read()
        except Exception as e:
            return jsonify({"error": f"Failed to read image file: {str(e)}"}), 400

        if not image_bytes:
            return jsonify({"error": "Image file is empty"}), 400

        # Validate file size (8MB limit as suggested by the frontend)
        max_size = 8 * 1024 * 1024  # 8MB
        if len(image_bytes) > max_size:
            return jsonify({"error": f"Image file too large. Maximum size is {max_size // (1024 * 1024)}MB"}), 400

        filename = image_file.filename or "input.png"
    except Exception as e:
        return jsonify({"error": f"Failed to process request: {str(e)}"}), 400

    try:
        output_format = normalize_output_format(request.form.get("outputFormat", "png"))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Advanced multi-channel
    if channels_json:
        import json

        try:
            channels = json.loads(channels_json)
        except json.JSONDecodeError as e:
            return jsonify({"error": f"Invalid JSON in channels payload: {str(e)}"}), 400
        except Exception as e:
            return jsonify({"error": f"Failed to parse channels payload: {str(e)}"}), 400

        if not isinstance(channels, dict):
            return jsonify({"error": "Channels payload must be a JSON object"}), 400

        channel_payloads = {}
        try:
            for ch in ["R", "G", "B", "A"]:
                cfg = channels.get(ch) or {}
                enabled = bool(cfg.get("enabled"))
                if not enabled:
                    channel_payloads[ch] = {"enabled": False}
                    continue

                payload_type = cfg.get("type")
                if payload_type == "text":
                    channel_payloads[ch] = {
                        "enabled": True,
                        "type": "text",
                        "text": cfg.get("text") or "",
                    }
                elif payload_type == "file":
                    file_field = f"file_{ch}"
                    upload = request.files.get(file_field)
                    if not upload:
                        return jsonify({"error": f"Missing file upload for channel {ch}"}), 400
                    try:
                        file_data = upload.read()
                        if not file_data:
                            return jsonify({"error": f"File for channel {ch} is empty"}), 400
                        channel_payloads[ch] = {
                            "enabled": True,
                            "type": "file",
                            "file_data": file_data,
                        }
                    except Exception as e:
                        return jsonify({"error": f"Failed to read file for channel {ch}: {str(e)}"}), 400
                else:
                    return jsonify({"error": f"Invalid payload type '{payload_type}' for channel {ch}. Must be 'text' or 'file'"}), 400
        except Exception as e:
            return jsonify({"error": f"Failed to process channel payloads: {str(e)}"}), 400

        try:
            encoded_name, encoded_bytes = encode_multi_channel(
                image_bytes,
                channel_payloads,
                filename=filename,
                output_format=output_format,
            )
            output_mime = sniff_image_mime(encoded_bytes)
            return jsonify(
                {"filename": encoded_name, "data_url": as_data_url(encoded_bytes, mime=output_mime)}
            )
        except ValueError as exc:
            return jsonify({"error": f"Encoding failed: {str(exc)}"}), 400
        except Exception as exc:
            return jsonify({"error": f"Unexpected error during encoding: {str(exc)}"}), 500

    # Simple mode
    if mode not in {"text", "zlib"}:
        return jsonify({"error": f"Invalid mode '{mode}'. Must be 'text' or 'zlib'"}), 400

    if mode == "text":
        text = payload_text or ""
        if not text:
            return jsonify({"error": "Text payload is required for text mode"}), 400
        file_data = None
    else:  # zlib mode
        if not payload_file:
            return jsonify({"error": "File upload is required for zlib mode"}), 400
        try:
            file_data = payload_file.read()
            if not file_data:
                return jsonify({"error": "Payload file is empty"}), 400
        except Exception as e:
            return jsonify({"error": f"Failed to read payload file: {str(e)}"}), 400
        text = None

    try:
        encoded_name, encoded_bytes = encode_payload(
            image_bytes,
            filename=filename,
            mode=mode,
            plane=plane,
            text=text,
            file_data=file_data,
            output_format=output_format,
            lossy_output=output_format != "jpeg",
        )
        output_mime = sniff_image_mime(encoded_bytes)
        return jsonify(
            {
                "filename": encoded_name,
                "data_url": as_data_url(encoded_bytes, mime=output_mime),
            }
        )
    except ValueError as exc:
        return jsonify({"error": f"Encoding failed: {str(exc)}"}), 400
    except Exception as exc:
        return jsonify({"error": f"Unexpected error during encoding: {str(exc)}"}), 500


@app.post("/api/decode")
def api_decode():
    image_file = request.files.get("image")
    if image_file is None:
        return jsonify({"error": "Image file is required"}), 400

    if not image_file.filename:
        return jsonify({"error": "Image file must have a filename"}), 400

    try:
        image_bytes = image_file.read()
    except Exception as e:
        return jsonify({"error": f"Failed to read image file: {str(e)}"}), 400

    if not image_bytes:
        return jsonify({"error": "Image file is empty"}), 400

    # Validate file size (8MB limit)
    max_size = 8 * 1024 * 1024  # 8MB
    if len(image_bytes) > max_size:
        return jsonify({"error": f"Image file too large. Maximum size is {max_size // (1024 * 1024)}MB"}), 400

    password = request.form.get("password") or None
    deep_analysis = _form_flag(request.form.get("deep", "false"))
    manual_tools = _form_flag(request.form.get("manual", "false"))
    binwalk_extract = _form_flag(request.form.get("binwalkExtract", "false"))

    try:
        analysis = run_analysis(
            image_bytes,
            image_file.filename or "upload.png",
            password=password,
            deep_analysis=deep_analysis,
            manual_tools=manual_tools,
            binwalk_extract=binwalk_extract,
        )
        return jsonify(analysis)
    except ValueError as exc:
        return jsonify({"error": f"Analysis failed: {str(exc)}"}), 400
    except Exception as exc:
        return jsonify({"error": f"Unexpected error during analysis: {str(exc)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
