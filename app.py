"""Flask entrypoint that exposes encoder/decoder endpoints and serves the UI."""

from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, render_template, request

from engine.decoder import run_analysis
from engine.encoder import (
    as_data_url,
    encode_chroma_payload,
    encode_dct_payload,
    encode_f5_payload,
    encode_lsb_payload,
    encode_multi_channel,
    encode_palette_payload,
    encode_payload,
    encode_png_chunks_payload,
    encode_pvd_payload,
    encode_spread_spectrum_payload,
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
        encode_method = request.form.get("encodeMethod")
        legacy_mode = request.form.get("encodeMode")
        if not encode_method:
            encode_method = "advanced_lsb" if legacy_mode == "advanced" else "simple_lsb"
        encode_method = (encode_method or "simple_lsb").strip().lower()

        payload_text: Optional[str] = request.form.get("text") or None
        payload_mode = (request.form.get("payloadMode") or "text").strip().lower()
        if payload_mode == "text" and request.files.get("payload") and not payload_text:
            payload_mode = "file"
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

    # Advanced LSB per-channel
    if encode_method == "advanced_lsb":
        if not channels_json:
            return jsonify({"error": "Channel payloads are required for advanced_lsb."}), 400
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

    # Simple LSB (legacy-compatible)
    if encode_method == "simple_lsb":
        mode = request.form.get("mode") or ("zlib" if payload_mode == "file" else "text")
        plane = request.form.get("plane", "RGB")
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
                lossy_output=True,
            )
            output_mime = sniff_image_mime(encoded_bytes)
            return jsonify(
                {"filename": encoded_name, "data_url": as_data_url(encoded_bytes, mime=output_mime)}
            )
        except ValueError as exc:
            return jsonify({"error": f"Encoding failed: {str(exc)}"}), 400
        except Exception as exc:
            return jsonify({"error": f"Unexpected error during encoding: {str(exc)}"}), 500

    # Ste.gg-style encoder methods
    payload: Optional[bytes] = None
    if payload_mode == "file":
        if not payload_file:
            return jsonify({"error": "Payload file is required for file mode"}), 400
        try:
            payload = payload_file.read()
        except Exception as exc:
            return jsonify({"error": f"Failed to read payload file: {str(exc)}"}), 400
        if not payload:
            return jsonify({"error": "Payload file is empty"}), 400
    else:
        text = payload_text or ""
        if not text:
            return jsonify({"error": "Text payload is required for this mode"}), 400
        payload = text.encode("utf-8")

    try:
        if encode_method == "lsb":
            channels = request.form.get("lsbChannels", "RGB")
            bits_per_channel = int(request.form.get("lsbBits", "1"))
            encoded_name, encoded_bytes = encode_lsb_payload(
                image_bytes,
                payload,
                channels=channels,
                bits_per_channel=bits_per_channel,
                output_format=output_format,
                filename=filename,
            )
        elif encode_method == "pvd":
            direction = request.form.get("pvdDirection", "horizontal")
            range_kind = request.form.get("pvdRange", "wu-tsai")
            encoded_name, encoded_bytes = encode_pvd_payload(
                image_bytes,
                payload,
                direction=direction,
                range_kind=range_kind,
                output_format=output_format,
                filename=filename,
            )
        elif encode_method == "dct":
            robustness = request.form.get("dctRobustness", "medium")
            block_size = int(request.form.get("dctBlockSize", "8"))
            encoded_name, encoded_bytes = encode_dct_payload(
                image_bytes,
                payload,
                block_size=block_size,
                robustness=robustness,
                output_format="jpeg",
                filename=filename,
            )
        elif encode_method == "f5":
            password = request.form.get("f5Password") or ""
            if not password:
                return jsonify({"error": "F5 requires a password."}), 400
            quality_val = request.form.get("f5Quality", "0.95")
            try:
                quality_float = float(quality_val)
            except ValueError:
                quality_float = 0.95
            quality = int(quality_float * 100) if quality_float <= 1 else int(quality_float)
            encoded_name, encoded_bytes = encode_f5_payload(
                image_bytes,
                payload,
                password=password,
                quality=quality,
                output_format="jpeg",
                filename=filename,
            )
        elif encode_method == "spread_spectrum":
            password = request.form.get("spreadPassword") or ""
            if not password:
                return jsonify({"error": "Spread spectrum requires a password."}), 400
            chip_length = int(request.form.get("spreadFactor", "16"))
            strength = int(request.form.get("spreadStrength", "2"))
            encoded_name, encoded_bytes = encode_spread_spectrum_payload(
                image_bytes,
                payload,
                password=password,
                chip_length=chip_length,
                strength=strength,
                output_format=output_format,
                filename=filename,
            )
        elif encode_method == "palette":
            colors = int(request.form.get("paletteColors", "256"))
            mode = request.form.get("paletteMode", "index")
            encoded_name, encoded_bytes = encode_palette_payload(
                image_bytes,
                payload,
                colors=colors,
                mode=mode,
                output_format="png",
                filename=filename,
            )
        elif encode_method == "chroma":
            color_space = request.form.get("chromaSpace", "ycbcr")
            channel = request.form.get("chromaChannel", "both")
            intensity = int(request.form.get("chromaIntensity", "2"))
            pattern = request.form.get("chromaPattern", "sequential")
            encoded_name, encoded_bytes = encode_chroma_payload(
                image_bytes,
                payload,
                color_space=color_space,
                channel=channel,
                intensity=intensity,
                pattern=pattern,
                output_format=output_format,
                filename=filename,
            )
        elif encode_method == "png_chunks":
            chunk_type = request.form.get("pngChunkType", "tEXt")
            keyword = request.form.get("pngChunkKeyword", "Comment")
            encoded_name, encoded_bytes = encode_png_chunks_payload(
                image_bytes,
                payload,
                chunk_type=chunk_type,
                keyword=keyword,
                output_format="png",
                filename=filename,
            )
        else:
            return jsonify({"error": f"Unknown encode method '{encode_method}'"}), 400

        output_mime = sniff_image_mime(encoded_bytes)
        return jsonify(
            {"filename": encoded_name, "data_url": as_data_url(encoded_bytes, mime=output_mime)}
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
    invisible_unicode = _form_flag(request.form.get("unicodeSweep", "false"))
    unicode_tier1 = _form_flag(request.form.get("unicodeTier1", "false"))
    unicode_separators = _form_flag(request.form.get("unicodeSeparators", "false"))
    unicode_aggressiveness = request.form.get("unicodeAggressiveness") or "balanced"
    decode_option = request.form.get("decodeOption") or None
    spread_enabled = _form_flag(request.form.get("spreadSpectrum", "false"))

    try:
        analysis = run_analysis(
            image_bytes,
            image_file.filename or "upload.png",
            password=password,
            deep_analysis=deep_analysis,
            manual_tools=manual_tools,
            binwalk_extract=binwalk_extract,
            invisible_unicode=invisible_unicode,
            unicode_tier1=unicode_tier1,
            unicode_separators=unicode_separators,
            unicode_aggressiveness=unicode_aggressiveness,
            spread_enabled=spread_enabled,
            decode_option=decode_option,
        )
        return jsonify(analysis)
    except ValueError as exc:
        return jsonify({"error": f"Analysis failed: {str(exc)}"}), 400
    except Exception as exc:
        return jsonify({"error": f"Unexpected error during analysis: {str(exc)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
