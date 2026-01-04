"""Shared tooling availability helpers."""

import shutil
from typing import Dict, List, Union

ToolCmd = Union[str, List[str]]

TOOLS: Dict[str, ToolCmd] = {
    "binwalk": "binwalk",
    "foremost": "foremost",
    "exiftool": "exiftool",
    "steghide": "steghide",
    "outguess": "outguess",
    "zsteg": "zsteg",
    "strings": "strings",
    "identify": "identify",
    "convert": "convert",
    "jpegtran": "jpegtran",
    "cjpeg": "cjpeg",
    "djpeg": "djpeg",
    "jpeginfo": "jpeginfo",
    "jpegsnoop": "jpegsnoop",
    "jhead": "jhead",
    "exiv2": "exiv2",
    "exifprobe": "exifprobe",
    "pngcheck": "pngcheck",
    "optipng": "optipng",
    "pngcrush": "pngcrush",
    "pngtools": "pngtools",
    "stegdetect": "stegdetect",
    "jsteg": "jsteg",
    "stegbreak": "stegbreak",
    "stegseek": "stegseek",
    "stegcracker": "stegcracker",
    "fcrackzip": "fcrackzip",
    "bulk_extractor": ["bulk_extractor", "bulk-extractor"],
    "stegoveritas": "stegoveritas",
    "scalpel": "scalpel",
    "testdisk": "testdisk",
    "photorec": "photorec",
    "zbarimg": "zbarimg",
    "qrencode": "qrencode",
    "tesseract": "tesseract",
    "ffmpeg": "ffmpeg",
    "ffprobe": "ffprobe",
    "mediainfo": "mediainfo",
    "sox": "sox",
    "pdfinfo": "pdfinfo",
    "pdftotext": "pdftotext",
    "pdfimages": "pdfimages",
    "qpdf": "qpdf",
    "radare2": ["radare2", "r2"],
    "rizin": "rizin",
    "hexyl": "hexyl",
    "bvi": "bvi",
    "rg": "rg",
    "xxd": "xxd",
    "tshark": "tshark",
    "wireshark": "wireshark",
    "sleuthkit": "mmls",
    "volatility": ["volatility", "volatility3", "vol"],
    "stegsolve": "stegsolve",
    "openstego": "openstego",
    "7z": "7z",
    "file": "file",
    "unzip": "unzip",
    "tar": "tar",
    "xz": "xz",
    "gzip": "gzip",
    "bzip2": "bzip2",
    "unsquashfs": "unsquashfs",
}

TOOL_MODES: Dict[str, str] = {
    "outguess": "deep",
    "stegbreak": "deep",
    "stegseek": "deep",
    "stegcracker": "deep",
    "fcrackzip": "deep",
    "bulk_extractor": "deep",
    "scalpel": "deep",
    "stegoveritas": "deep",
    "ffmpeg": "deep",
    "testdisk": "manual",
    "photorec": "manual",
    "qrencode": "manual",
    "bvi": "manual",
    "wireshark": "manual",
    "stegsolve": "manual",
}


def get_tool_status() -> Dict[str, Dict[str, str]]:
    """
    Return mapping of tool -> {available: bool, path: str|None}.
    """
    status: Dict[str, Dict[str, str]] = {}
    for name, cmd in TOOLS.items():
        cmds = cmd if isinstance(cmd, list) else [cmd]
        path = ""
        for candidate in cmds:
            found = shutil.which(candidate)
            if found:
                path = found
                break
        status[name] = {
            "available": bool(path),
            "path": path or "",
            "mode": TOOL_MODES.get(name, "auto"),
        }
    return status
