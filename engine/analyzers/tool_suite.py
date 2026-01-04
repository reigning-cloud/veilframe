"""Run a broad CLI tool suite against an uploaded file."""

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from PIL import Image

from .utils import MAX_PENDING_TIME, update_data

MAX_OUTPUT_LINES = 200
MAX_OUTPUT_CHARS = 4000


def _truncate_lines(text: str, max_lines: int = MAX_OUTPUT_LINES) -> list[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) > max_lines:
        extra = len(lines) - max_lines
        lines = lines[:max_lines]
        lines.append(f"... ({extra} more lines truncated)")
    return lines


def _truncate_text(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... (truncated)"


def _detect_mime(input_img: Path) -> str:
    try:
        data = subprocess.run(
            ["file", "--mime-type", "-b", str(input_img)],
            capture_output=True,
            text=True,
            timeout=MAX_PENDING_TIME,
            check=False,
        )
        if data.returncode == 0 and data.stdout:
            return data.stdout.strip()
    except Exception:
        pass

    try:
        img = Image.open(input_img)
        if img.format:
            return f"image/{img.format.lower()}"
    except Exception:
        pass

    return ""


def _record(
    output_dir: Path,
    key: str,
    *,
    status: str,
    output: Optional[object] = None,
    error: Optional[str] = None,
    reason: Optional[str] = None,
) -> None:
    payload: dict[str, object] = {"status": status}
    if output is not None:
        payload["output"] = output
    if error:
        payload["error"] = error
    if reason:
        payload["reason"] = reason
    update_data(output_dir, {key: payload})


def _run_tool(
    output_dir: Path,
    key: str,
    cmd: list[str],
    *,
    cwd: Optional[Path] = None,
    allow_error: bool = False,
    output_mode: str = "lines",
    note: Optional[str] = None,
) -> bool:
    binary = cmd[0]
    if not shutil.which(binary):
        _record(output_dir, key, status="skipped", reason=f"{binary} not installed")
        return False

    try:
        data = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=MAX_PENDING_TIME,
            check=False,
        )
    except subprocess.TimeoutExpired:
        _record(
            output_dir,
            key,
            status="error",
            error=f"{binary} timed out after {MAX_PENDING_TIME} seconds",
        )
        return False
    except Exception as exc:
        _record(output_dir, key, status="error", error=str(exc))
        return False

    combined = "\n".join([s for s in [data.stdout, data.stderr] if s])
    if data.returncode != 0 and not allow_error:
        _record(
            output_dir,
            key,
            status="error",
            error=_truncate_text(combined) or f"{binary} exited with code {data.returncode}",
        )
        return False

    if output_mode == "text":
        output = _truncate_text(combined) if combined else "ok"
        if note:
            output = f"{note}\n{output}" if output else note
    else:
        output = _truncate_lines(combined) if combined else ["ok"]
        if note:
            output = [note, *output]

    _record(output_dir, key, status="ok", output=output)
    return True


def _skip_if(
    output_dir: Path,
    key: str,
    *,
    condition: bool,
    reason: str,
) -> bool:
    if condition:
        return False
    _record(output_dir, key, status="skipped", reason=reason)
    return True


def _list_files(base_dir: Path) -> list[str]:
    files: list[str] = []
    for path in sorted(base_dir.rglob("*")):
        if path.is_file():
            try:
                files.append(str(path.relative_to(base_dir)))
            except Exception:
                files.append(str(path))
    return files


def analyze_tool_suite(
    input_img: Path,
    output_dir: Path,
    deep_analysis: bool = False,
    manual_tools: bool = False,
) -> None:
    """Run a large CLI tool suite against the uploaded file."""
    if not input_img.exists():
        _record(
            output_dir,
            "tool_suite",
            status="error",
            error=f"Input image not found: {input_img}",
        )
        return

    mime = _detect_mime(input_img)
    is_jpeg = mime == "image/jpeg"
    is_png = mime == "image/png"
    is_image = mime.startswith("image/")
    is_pdf = mime == "application/pdf"
    is_audio = mime.startswith("audio/")
    is_video = mime.startswith("video/")
    is_pcap = mime in {"application/vnd.tcpdump.pcap", "application/x-pcap"}

    tool_dir = output_dir / "tool_suite"
    tool_dir.mkdir(parents=True, exist_ok=True)

    if not _skip_if(output_dir, "identify", condition=is_image, reason="Not an image"):
        _run_tool(output_dir, "identify", ["identify", "-verbose", str(input_img)])

    if not _skip_if(output_dir, "convert", condition=is_image, reason="Not an image"):
        _run_tool(
            output_dir,
            "convert",
            ["convert", str(input_img), "-format", "%m %w %h", "info:"],
            output_mode="text",
        )

    if not _skip_if(output_dir, "jpeginfo", condition=is_jpeg, reason="Not a JPEG"):
        _run_tool(output_dir, "jpeginfo", ["jpeginfo", "-c", str(input_img)])

    if not _skip_if(output_dir, "jpegtran", condition=is_jpeg, reason="Not a JPEG"):
        out_file = tool_dir / "jpegtran.jpg"
        _run_tool(
            output_dir,
            "jpegtran",
            ["jpegtran", "-copy", "none", "-optimize", "-outfile", str(out_file), str(input_img)],
        )

    _run_tool(output_dir, "cjpeg", ["cjpeg", "-version"], allow_error=True, output_mode="text")
    _run_tool(output_dir, "djpeg", ["djpeg", "-version"], allow_error=True, output_mode="text")

    if not _skip_if(output_dir, "jpegsnoop", condition=is_jpeg, reason="Not a JPEG"):
        _run_tool(output_dir, "jpegsnoop", ["jpegsnoop", str(input_img)])

    if not _skip_if(output_dir, "jhead", condition=is_jpeg, reason="Not a JPEG"):
        _run_tool(output_dir, "jhead", ["jhead", str(input_img)])

    if not _skip_if(output_dir, "exiv2", condition=is_image, reason="Not an image"):
        _run_tool(output_dir, "exiv2", ["exiv2", "-pa", str(input_img)])

    if not _skip_if(output_dir, "exifprobe", condition=is_image, reason="Not an image"):
        _run_tool(output_dir, "exifprobe", ["exifprobe", str(input_img)])

    if not _skip_if(output_dir, "pngcheck", condition=is_png, reason="Not a PNG"):
        _run_tool(output_dir, "pngcheck", ["pngcheck", "-v", str(input_img)])

    if not _skip_if(output_dir, "optipng", condition=is_png, reason="Not a PNG"):
        _run_tool(output_dir, "optipng", ["optipng", "-simulate", "-quiet", str(input_img)])

    if not _skip_if(output_dir, "pngcrush", condition=is_png, reason="Not a PNG"):
        out_file = tool_dir / "pngcrush.png"
        _run_tool(output_dir, "pngcrush", ["pngcrush", "-q", str(input_img), str(out_file)])

    if not _skip_if(output_dir, "pngtools", condition=is_png, reason="Not a PNG"):
        _run_tool(
            output_dir,
            "pngtools",
            ["pngtools", str(input_img)],
            output_mode="text",
            note="auto mode: pngtools (pngfix fallback uses pngcheck -v)",
        )

    if not _skip_if(output_dir, "stegdetect", condition=is_jpeg, reason="Not a JPEG"):
        _run_tool(output_dir, "stegdetect", ["stegdetect", "-t", "jopi", "-v", str(input_img)])

    if not _skip_if(output_dir, "jsteg", condition=is_jpeg, reason="Not a JPEG"):
        _run_tool(
            output_dir,
            "jsteg",
            ["jsteg", "reveal", str(input_img)],
            allow_error=True,
            output_mode="text",
        )

    if _skip_if(
        output_dir,
        "stegbreak",
        condition=deep_analysis,
        reason="Enable deep analysis to run brute-force tools",
    ):
        pass
    elif not _skip_if(output_dir, "stegbreak", condition=is_jpeg, reason="Not a JPEG"):
        _run_tool(output_dir, "stegbreak", ["stegbreak", "-r", "1", "-n", "1", str(input_img)])

    if _skip_if(
        output_dir,
        "stegseek",
        condition=deep_analysis,
        reason="Enable deep analysis to run brute-force tools",
    ):
        pass
    elif not _skip_if(output_dir, "stegseek", condition=is_jpeg, reason="Not a JPEG"):
        seed_out = tool_dir / "stegseek_seed.out"
        _run_tool(
            output_dir,
            "stegseek",
            ["stegseek", "--seed", str(input_img), str(seed_out)],
            allow_error=True,
            cwd=tool_dir,
        )

    if _skip_if(
        output_dir,
        "stegcracker",
        condition=deep_analysis,
        reason="Enable deep analysis to run brute-force tools",
    ):
        pass
    elif not _skip_if(output_dir, "stegcracker", condition=is_jpeg, reason="Not a JPEG"):
        _run_tool(
            output_dir,
            "stegcracker",
            ["stegcracker", str(input_img)],
            allow_error=True,
        )

    if _skip_if(
        output_dir,
        "fcrackzip",
        condition=deep_analysis,
        reason="Enable deep analysis to run brute-force tools",
    ):
        pass
    else:
        _record(output_dir, "fcrackzip", status="skipped", reason="Requires a zip archive")

    if _skip_if(
        output_dir,
        "bulk_extractor",
        condition=deep_analysis,
        reason="Enable deep analysis to run bulk_extractor",
    ):
        pass
    else:
        bulk_dir = tool_dir / "bulk_extractor"
        bulk_dir.mkdir(parents=True, exist_ok=True)
        if _run_tool(
            output_dir,
            "bulk_extractor",
            ["bulk_extractor", "-q", "-o", str(bulk_dir), str(input_img)],
            cwd=bulk_dir,
        ):
            _record(output_dir, "bulk_extractor", status="ok", output=_list_files(bulk_dir))

    if _skip_if(
        output_dir,
        "scalpel",
        condition=deep_analysis,
        reason="Enable deep analysis to run scalpel",
    ):
        pass
    else:
        scalpel_dir = tool_dir / "scalpel"
        scalpel_dir.mkdir(parents=True, exist_ok=True)
        if _run_tool(
            output_dir,
            "scalpel",
            ["scalpel", "-o", str(scalpel_dir), str(input_img)],
            cwd=scalpel_dir,
            allow_error=True,
        ):
            _record(output_dir, "scalpel", status="ok", output=_list_files(scalpel_dir))

    if _skip_if(
        output_dir,
        "testdisk",
        condition=manual_tools,
        reason="Enable manual tools to run interactive utilities",
    ):
        pass
    else:
        _run_tool(
            output_dir,
            "testdisk",
            ["testdisk", "/version"],
            allow_error=True,
            output_mode="text",
            note="manual mode: testdisk /version",
        )

    if _skip_if(
        output_dir,
        "photorec",
        condition=manual_tools,
        reason="Enable manual tools to run interactive utilities",
    ):
        pass
    else:
        _run_tool(
            output_dir,
            "photorec",
            ["photorec", "/version"],
            allow_error=True,
            output_mode="text",
            note="manual mode: photorec /version",
        )

    if _skip_if(
        output_dir,
        "stegoveritas",
        condition=deep_analysis,
        reason="Enable deep analysis to run stegoveritas",
    ):
        pass
    else:
        veritas_dir = tool_dir / "stegoveritas"
        veritas_dir.mkdir(parents=True, exist_ok=True)
        if _run_tool(
            output_dir,
            "stegoveritas",
            ["stegoveritas", "-meta", "-out", str(veritas_dir), str(input_img)],
            cwd=veritas_dir,
            allow_error=True,
        ):
            _record(output_dir, "stegoveritas", status="ok", output=_list_files(veritas_dir))

    if not _skip_if(output_dir, "zbarimg", condition=is_image, reason="Not an image"):
        _run_tool(
            output_dir,
            "zbarimg",
            ["zbarimg", "--quiet", str(input_img)],
            allow_error=True,
            note="auto mode: zbarimg --quiet (no barcode found is normal)",
        )

    if _skip_if(
        output_dir,
        "qrencode",
        condition=manual_tools,
        reason="Enable manual tools to run encoder utilities",
    ):
        pass
    else:
        qr_out = tool_dir / "qrencode.png"
        qr_text = f"veilframe:{input_img.name}"
        if _run_tool(
            output_dir,
            "qrencode",
            ["qrencode", "-o", str(qr_out), "-t", "PNG", qr_text],
            allow_error=True,
            output_mode="text",
            note=f"manual mode: qrencode -o {qr_out.name} -t PNG \"{qr_text}\"",
        ):
            if qr_out.exists():
                _record(
                    output_dir,
                    "qrencode",
                    status="ok",
                    output=[
                        f"manual mode: qrencode -o {qr_out.name} -t PNG \"{qr_text}\"",
                        qr_out.name,
                    ],
                )
            else:
                _record(
                    output_dir,
                    "qrencode",
                    status="ok",
                    output=[
                        f"manual mode: qrencode -o {qr_out.name} -t PNG \"{qr_text}\"",
                        "no output file created",
                    ],
                )

    if not _skip_if(output_dir, "tesseract", condition=is_image, reason="Not an image"):
        out_base = tool_dir / "tesseract_output"
        if _run_tool(
            output_dir,
            "tesseract",
            ["tesseract", str(input_img), str(out_base)],
            allow_error=True,
        ):
            out_txt = out_base.with_suffix(".txt")
            if out_txt.exists():
                _record(
                    output_dir,
                    "tesseract",
                    status="ok",
                    output=_truncate_text(out_txt.read_text(errors="ignore")),
                )

    if not _skip_if(output_dir, "ffprobe", condition=is_image or is_audio or is_video, reason="Not media"):
        _run_tool(
            output_dir,
            "ffprobe",
            ["ffprobe", "-hide_banner", "-show_format", "-show_streams", str(input_img)],
            allow_error=True,
        )

    if _skip_if(
        output_dir,
        "ffmpeg",
        condition=deep_analysis,
        reason="Enable deep analysis to run ffmpeg",
    ):
        pass
    else:
        _run_tool(
            output_dir,
            "ffmpeg",
            ["ffmpeg", "-hide_banner", "-i", str(input_img), "-f", "null", "-"],
            allow_error=True,
        )

    if not _skip_if(output_dir, "mediainfo", condition=is_image or is_audio or is_video, reason="Not media"):
        _run_tool(output_dir, "mediainfo", ["mediainfo", str(input_img)])

    if not _skip_if(output_dir, "sox", condition=is_audio, reason="Not audio"):
        _run_tool(output_dir, "sox", ["sox", "--i", str(input_img)])

    if not _skip_if(output_dir, "pdfinfo", condition=is_pdf, reason="Not a PDF"):
        _run_tool(output_dir, "pdfinfo", ["pdfinfo", str(input_img)])

    if not _skip_if(output_dir, "pdftotext", condition=is_pdf, reason="Not a PDF"):
        out_file = tool_dir / "pdftotext.txt"
        if _run_tool(output_dir, "pdftotext", ["pdftotext", str(input_img), str(out_file)]):
            if out_file.exists():
                _record(
                    output_dir,
                    "pdftotext",
                    status="ok",
                    output=_truncate_text(out_file.read_text(errors="ignore")),
                )

    if not _skip_if(output_dir, "pdfimages", condition=is_pdf, reason="Not a PDF"):
        out_prefix = tool_dir / "pdfimages"
        if _run_tool(output_dir, "pdfimages", ["pdfimages", str(input_img), str(out_prefix)]):
            matches = sorted(path.name for path in tool_dir.glob("pdfimages*") if path.is_file())
            _record(output_dir, "pdfimages", status="ok", output=matches or ["no outputs found"])

    if not _skip_if(output_dir, "qpdf", condition=is_pdf, reason="Not a PDF"):
        _run_tool(output_dir, "qpdf", ["qpdf", "--show-npages", str(input_img)])

    _run_tool(output_dir, "radare2", ["radare2", "-q", "-c", "iI;izzq", str(input_img)], allow_error=True)
    _run_tool(output_dir, "rizin", ["rizin", "-q", "-c", "iI;izzq", str(input_img)], allow_error=True)
    _run_tool(output_dir, "hexyl", ["hexyl", "-n", "256", str(input_img)], allow_error=True)
    if _skip_if(
        output_dir,
        "bvi",
        condition=manual_tools,
        reason="Enable manual tools to run interactive utilities",
    ):
        pass
    else:
        _run_tool(
            output_dir,
            "bvi",
            ["bvi", "-v"],
            allow_error=True,
            output_mode="text",
            note="manual mode: bvi -v",
        )
    _run_tool(output_dir, "xxd", ["xxd", "-l", "256", str(input_img)], allow_error=True)
    _run_tool(output_dir, "rg", ["rg", "-a", "-n", "-m", "3", "-e", "flag", "-e", "ctf", "-e", "steg", str(input_img)], allow_error=True)

    if not _skip_if(output_dir, "tshark", condition=is_pcap, reason="Not a pcap"):
        _run_tool(output_dir, "tshark", ["tshark", "-r", str(input_img), "-c", "5"], allow_error=True)

    if _skip_if(
        output_dir,
        "wireshark",
        condition=manual_tools,
        reason="Enable manual tools to run GUI utilities",
    ):
        pass
    else:
        _run_tool(
            output_dir,
            "wireshark",
            ["wireshark", "--version"],
            allow_error=True,
            output_mode="text",
            note="manual mode: wireshark --version",
        )
    is_disk_candidate = not (is_image or is_audio or is_video or is_pdf)
    if _skip_if(output_dir, "sleuthkit", condition=is_disk_candidate, reason="Not a disk image"):
        pass
    else:
        _run_tool(
            output_dir,
            "sleuthkit",
            ["mmls", str(input_img)],
            allow_error=False,
            note="auto mode: mmls",
        )
    _record(output_dir, "volatility", status="skipped", reason="Requires a memory image")
    if _skip_if(
        output_dir,
        "stegsolve",
        condition=manual_tools,
        reason="Enable manual tools to run GUI utilities",
    ):
        pass
    else:
        _run_tool(
            output_dir,
            "stegsolve",
            ["stegsolve", "--help"],
            allow_error=True,
            output_mode="text",
            note="manual mode: stegsolve --help",
        )
    _run_tool(
        output_dir,
        "openstego",
        ["openstego", "algorithms"],
        allow_error=True,
        output_mode="text",
        note="auto mode: openstego algorithms",
    )
