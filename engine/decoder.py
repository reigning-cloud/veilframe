"""Wrapper to run the analyzer suite on an uploaded image."""

import base64
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .analyzers import (
    analyze_binwalk,
    analyze_decomposer,
    analyze_exiftool,
    analyze_foremost,
    analyze_outguess,
    analyze_simple_lsb,
    analyze_simple_zlib,
    analyze_steghide,
    analyze_strings,
    analyze_tool_suite,
    analyze_zsteg,
)
from .analyzers.utils import update_data
from .tooling import get_tool_status


def _file_to_data_url(path: Path, mime: str) -> str:
    """Read a file and convert to data URL."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    try:
        data = path.read_bytes()
    except Exception as e:
        raise IOError(f"Failed to read file '{path}': {str(e)}")

    try:
        b64 = base64.b64encode(data).decode()
        return f"data:{mime};base64,{b64}"
    except Exception as e:
        raise ValueError(f"Failed to encode file as base64 data URL: {str(e)}")


def _read_results_file(output_dir: Path) -> Dict[str, Any]:
    """Load results.json if present."""
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    results_path = output_dir / "results.json"
    if not results_path.exists():
        return {}

    try:
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse results.json: {str(e)}")
    except Exception as e:
        raise IOError(f"Failed to read results.json: {str(e)}")


def _collect_artifacts(output_dir: Path) -> Dict[str, List[Dict[str, str]]]:
    """
    Collect generated images and archives as base64 data URLs so the frontend
    can preview and download them without touching the filesystem.
    """
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    images: List[Dict[str, str]] = []
    archives: List[Dict[str, str]] = []

    try:
        for path in sorted(output_dir.rglob("*")):
            if not path.is_file():
                continue

            try:
                if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
                    images.append({"name": path.name, "data_url": _file_to_data_url(path, mime)})
                elif path.suffix.lower() == ".7z":
                    archives.append(
                        {
                            "name": path.name,
                            "data_url": _file_to_data_url(
                                path, "application/x-7z-compressed"
                            ),
                        }
                    )
            except Exception as e:
                # Log the error but continue processing other files
                print(f"Warning: Failed to process artifact file '{path}': {str(e)}")
                continue

        return {"images": images, "archives": archives}
    except Exception as e:
        raise IOError(f"Failed to collect artifacts from {output_dir}: {str(e)}")


def _decode_bits_to_text(bits: np.ndarray) -> str:
    """Decode a flat LSB bitstream into text until a NULL byte."""
    try:
        usable = (bits.size // 8) * 8
        if usable == 0:
            return ""
        bits = bits[:usable]
        packed = np.packbits(bits, bitorder="big")
        if packed.size == 0:
            return ""
        zero_idx = np.where(packed == 0)[0]
        end = int(zero_idx[0]) if zero_idx.size else packed.size
        if end <= 0:
            return ""
        return bytes(packed[:end]).decode("latin-1")
    except Exception as e:
        raise ValueError(f"Failed to decode LSB text: {str(e)}")


def run_analysis(
    image_bytes: bytes,
    filename: str,
    *,
    password: Optional[str] = None,
    deep_analysis: bool = False,
    manual_tools: bool = False,
    binwalk_extract: bool = False,
) -> Dict[str, Any]:
    """
    Execute all analyzers over the provided image bytes and return results plus artifacts.
    """
    if not isinstance(image_bytes, bytes):
        raise TypeError(f"image_bytes must be bytes, got {type(image_bytes).__name__}")

    if not image_bytes:
        raise ValueError("Cannot analyze empty image data")

    if not filename:
        filename = "upload.png"

    try:
        safe_name = Path(filename).name or "upload.png"
    except Exception as e:
        raise ValueError(f"Invalid filename: {str(e)}")

    with TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        image_path = tmp_dir / safe_name
        output_dir = tmp_dir / "analysis"

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise IOError(f"Failed to create output directory: {str(e)}")

        try:
            with open(image_path, "wb") as f:
                f.write(image_bytes)
        except Exception as e:
            raise IOError(f"Failed to write image to temporary file: {str(e)}")

        channel_map = [
            ("red_plane", 0),
            ("green_plane", 1),
            ("blue_plane", 2),
            ("alpha_plane", 3),
        ]
        simple_rgb_text = ""
        channel_texts = {key: "" for key, _ in channel_map}

        try:
            img_local = Image.open(image_path).convert("RGBA")
            arr = np.array(img_local)
        except Exception as e:
            print(f"Warning: Failed to open image for channel decoding: {str(e)}")
            arr = None

        if arr is not None:
            try:
                simple_rgb_text = _decode_bits_to_text((arr[..., :3] & 1).reshape(-1))
            except Exception as e:
                # Non-fatal: continue analysis even if simple RGB decode fails
                print(f"Warning: Simple RGB decode failed: {str(e)}")
                simple_rgb_text = ""

            for key, idx in channel_map:
                try:
                    channel_texts[key] = _decode_bits_to_text(
                        (arr[..., idx] & 1).reshape(-1)
                    )
                except Exception as e:
                    print(f"Warning: {key} decode failed: {str(e)}")
                    channel_texts[key] = ""

        analyzers: List[Tuple[Any, Tuple[Any, ...]]] = [
            (analyze_binwalk, (image_path, output_dir, binwalk_extract)),
            (analyze_decomposer, (image_path, output_dir)),
            (analyze_exiftool, (image_path, output_dir)),
            (analyze_foremost, (image_path, output_dir)),
            (analyze_simple_lsb, (image_path, output_dir)),
            (analyze_simple_zlib, (image_path, output_dir)),
            (analyze_strings, (image_path, output_dir)),
            (analyze_steghide, (image_path, output_dir, password)),
            (analyze_tool_suite, (image_path, output_dir, deep_analysis, manual_tools)),
            (analyze_zsteg, (image_path, output_dir)),
        ]

        if deep_analysis:
            try:
                tools = get_tool_status()
                if tools.get("outguess", {}).get("available"):
                    analyzers.append((analyze_outguess, (image_path, output_dir)))
                else:
                    update_data(
                        output_dir,
                        {
                            "outguess": {
                                "status": "skipped",
                                "reason": "outguess not installed in runtime environment",
                            }
                        },
                    )
            except Exception as e:
                print(f"Warning: Failed to check outguess availability: {str(e)}")

        for analyzer_func, args in analyzers:
            try:
                analyzer_func(*args)
            except Exception as exc:  # pragma: no cover - defensive
                # Errors are already recorded inside analyzer, this keeps the pipeline alive.
                print(f"Analyzer {analyzer_func.__name__} failed: {exc}")

        try:
            results = _read_results_file(output_dir)
        except Exception as e:
            print(f"Warning: Failed to read results file: {str(e)}")
            results = {}

        results = {
            "simple_rgb": {
                "status": "ok" if simple_rgb_text else "empty",
                "output": simple_rgb_text,
            },
            "red_plane": {
                "status": "ok" if channel_texts["red_plane"] else "empty",
                "output": channel_texts["red_plane"],
            },
            "green_plane": {
                "status": "ok" if channel_texts["green_plane"] else "empty",
                "output": channel_texts["green_plane"],
            },
            "blue_plane": {
                "status": "ok" if channel_texts["blue_plane"] else "empty",
                "output": channel_texts["blue_plane"],
            },
            "alpha_plane": {
                "status": "ok" if channel_texts["alpha_plane"] else "empty",
                "output": channel_texts["alpha_plane"],
            },
            **results,
        }

        try:
            artifacts = _collect_artifacts(output_dir)
        except Exception as e:
            print(f"Warning: Failed to collect artifacts: {str(e)}")
            artifacts = {"images": [], "archives": []}

        return {"results": results, "artifacts": artifacts}
