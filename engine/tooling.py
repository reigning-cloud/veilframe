"""Shared tooling availability helpers."""

import shutil
from typing import Dict

TOOLS: Dict[str, str] = {
    "binwalk": "binwalk",
    "foremost": "foremost",
    "exiftool": "exiftool",
    "steghide": "steghide",
    "outguess": "outguess",
    "zsteg": "zsteg",
    "strings": "strings",
    "7z": "7z",
    "file": "file",
    "unzip": "unzip",
    "tar": "tar",
    "xz": "xz",
    "gzip": "gzip",
    "bzip2": "bzip2",
    "unsquashfs": "unsquashfs",
}


def get_tool_status() -> Dict[str, Dict[str, str]]:
    """
    Return mapping of tool -> {available: bool, path: str|None}.
    """
    status: Dict[str, Dict[str, str]] = {}
    for name, cmd in TOOLS.items():
        path = shutil.which(cmd)
        status[name] = {
            "available": bool(path),
            "path": path or "",
        }
    return status
