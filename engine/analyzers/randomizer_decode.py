"""Heuristic decoder for randomizer-style text transforms (P4RS3LT0NGV3 inspired)."""

from __future__ import annotations

import base64
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .utils import update_data

MAX_INPUT_CHARS = 4000
MAX_TRANSFORMS = 120
MIN_TOKEN_LEN = 4

SUSPECT_RE = re.compile(r"(flag\\{|ctf|steg|\\{[^}]{4,}\\})", re.IGNORECASE)

MORSE_MAP = {
    ".-": "A",
    "-...": "B",
    "-.-.": "C",
    "-..": "D",
    ".": "E",
    "..-.": "F",
    "--.": "G",
    "....": "H",
    "..": "I",
    ".---": "J",
    "-.-": "K",
    ".-..": "L",
    "--": "M",
    "-.": "N",
    "---": "O",
    ".--.": "P",
    "--.-": "Q",
    ".-.": "R",
    "...": "S",
    "-": "T",
    "..-": "U",
    "...-": "V",
    ".--": "W",
    "-..-": "X",
    "-.--": "Y",
    "--..": "Z",
    "-----": "0",
    ".----": "1",
    "..---": "2",
    "...--": "3",
    "....-": "4",
    ".....": "5",
    "-....": "6",
    "--...": "7",
    "---..": "8",
    "----.": "9",
}

BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
BASE62_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _strip_invisible(text: str) -> str:
    return "".join(
        ch for ch in text if unicodedata.category(ch) not in {"Cf", "Cc"}
    )


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text


def _score_text(text: str) -> float:
    if not text:
        return 0.0
    length = len(text)
    if length == 0:
        return 0.0
    letters = sum(ch.isalpha() for ch in text)
    spaces = sum(ch.isspace() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    vowels = sum(ch.lower() in "aeiou" for ch in text if ch.isalpha())
    printable = sum(32 <= ord(ch) < 127 for ch in text)
    score = (letters / length) + (vowels / max(1, letters)) * 0.3
    score += (spaces / length) * 0.2 + (printable / length) * 0.1
    score -= (digits / length) * 0.05
    return max(0.0, min(1.0, score))


def _rot13(text: str) -> str:
    out = []
    for ch in text:
        if "a" <= ch <= "z":
            out.append(chr((ord(ch) - 97 + 13) % 26 + 97))
        elif "A" <= ch <= "Z":
            out.append(chr((ord(ch) - 65 + 13) % 26 + 65))
        else:
            out.append(ch)
    return "".join(out)


def _atbash(text: str) -> str:
    out = []
    for ch in text:
        if "a" <= ch <= "z":
            out.append(chr(122 - (ord(ch) - 97)))
        elif "A" <= ch <= "Z":
            out.append(chr(90 - (ord(ch) - 65)))
        else:
            out.append(ch)
    return "".join(out)


def _caesar(text: str, shift: int) -> str:
    out = []
    for ch in text:
        if "a" <= ch <= "z":
            out.append(chr((ord(ch) - 97 + shift) % 26 + 97))
        elif "A" <= ch <= "Z":
            out.append(chr((ord(ch) - 65 + shift) % 26 + 65))
        else:
            out.append(ch)
    return "".join(out)


def _try_base64(text: str) -> Optional[str]:
    if len(text) < 8 or len(text) % 4 != 0:
        return None
    if not re.fullmatch(r"[A-Za-z0-9+/=]+", text):
        return None
    try:
        decoded = base64.b64decode(text, validate=True)
    except Exception:
        return None
    try:
        return decoded.decode("utf-8", errors="strict")
    except Exception:
        return None


def _try_base64url(text: str) -> Optional[str]:
    if len(text) < 8:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_-]+", text):
        return None
    pad = "=" * ((4 - len(text) % 4) % 4)
    try:
        decoded = base64.urlsafe_b64decode(text + pad)
    except Exception:
        return None
    try:
        return decoded.decode("utf-8", errors="strict")
    except Exception:
        return None


def _try_base32(text: str) -> Optional[str]:
    if len(text) < 8:
        return None
    if not re.fullmatch(r"[A-Z2-7=]+", text.upper()):
        return None
    try:
        decoded = base64.b32decode(text.upper(), casefold=True)
    except Exception:
        return None
    try:
        return decoded.decode("utf-8", errors="strict")
    except Exception:
        return None


def _try_hex(text: str) -> Optional[str]:
    if len(text) < 8 or len(text) % 2 != 0:
        return None
    if not re.fullmatch(r"[0-9A-Fa-f]+", text):
        return None
    try:
        decoded = bytes.fromhex(text)
    except Exception:
        return None
    try:
        return decoded.decode("utf-8", errors="strict")
    except Exception:
        return None


def _try_binary(text: str) -> Optional[str]:
    bits = text.replace(" ", "")
    if len(bits) < 8 or len(bits) % 8 != 0:
        return None
    if not re.fullmatch(r"[01]+", bits):
        return None
    try:
        decoded = bytes(int(bits[i : i + 8], 2) for i in range(0, len(bits), 8))
    except Exception:
        return None
    try:
        return decoded.decode("utf-8", errors="strict")
    except Exception:
        return None


def _try_morse(text: str) -> Optional[str]:
    for ch in text:
        if ch not in {".", "-", "/", " "}:
            return None
    out = []
    for token in text.strip().split(" "):
        if token == "/":
            out.append(" ")
            continue
        letter = MORSE_MAP.get(token)
        if not letter:
            return None
        out.append(letter)
    return "".join(out)


def _try_base58(text: str) -> Optional[str]:
    if len(text) < 8:
        return None
    if any(ch not in BASE58_ALPHABET for ch in text):
        return None
    num = 0
    for ch in text:
        num = num * 58 + BASE58_ALPHABET.index(ch)
    raw = num.to_bytes((num.bit_length() + 7) // 8, "big") if num else b""
    pad = len(text) - len(text.lstrip("1"))
    raw = b"\x00" * pad + raw
    try:
        return raw.decode("utf-8", errors="strict")
    except Exception:
        return None


def _try_base62(text: str) -> Optional[str]:
    if len(text) < 8:
        return None
    if any(ch not in BASE62_ALPHABET for ch in text):
        return None
    num = 0
    for ch in text:
        num = num * 62 + BASE62_ALPHABET.index(ch)
    raw = num.to_bytes((num.bit_length() + 7) // 8, "big") if num else b""
    try:
        return raw.decode("utf-8", errors="strict")
    except Exception:
        return None


def _try_leetspeak(text: str) -> Optional[str]:
    if not re.search(r"[0-9]", text):
        return None
    variants = [
        {"4": "a", "3": "e", "1": "i", "0": "o", "5": "s", "7": "t"},
        {"4": "a", "3": "e", "1": "l", "0": "o", "5": "s", "7": "t"},
    ]
    best = None
    best_score = 0.0
    for mapping in variants:
        out = "".join(mapping.get(ch, ch) for ch in text)
        score = _score_text(out)
        if score > best_score:
            best_score = score
            best = out
    return best


def _decode_word(word: str) -> Tuple[str, str, float]:
    candidates: List[Tuple[str, str]] = [("raw", word)]
    norm = _normalize_text(word)
    if norm != word:
        candidates.append(("nfkc", norm))
    if word.isalpha() and len(word) >= MIN_TOKEN_LEN:
        candidates.append(("rot13", _rot13(word)))
        candidates.append(("atbash", _atbash(word)))
        for shift in range(1, 26):
            candidates.append((f"caesar_{shift}", _caesar(word, shift)))
    if len(word) >= MIN_TOKEN_LEN:
        candidates.append(("reverse", word[::-1]))

    for name, func in [
        ("base64", _try_base64),
        ("base64url", _try_base64url),
        ("base32", _try_base32),
        ("hex", _try_hex),
        ("binary", _try_binary),
        ("morse", _try_morse),
        ("base58", _try_base58),
        ("base62", _try_base62),
        ("leetspeak", _try_leetspeak),
    ]:
        decoded = func(word)
        if decoded:
            candidates.append((name, decoded))

    best_name = "raw"
    best_text = word
    best_score = _score_text(word)
    for name, cand in candidates:
        score = _score_text(cand)
        if score > best_score:
            best_name = name
            best_text = cand
            best_score = score
    return best_name, best_text, best_score


def _decode_randomizer_text(text: str) -> Tuple[str, List[Dict[str, str]], float, float]:
    text = _strip_invisible(text)
    text = _normalize_text(text)
    text = text.strip()
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]

    raw_score = _score_text(text)
    parts = re.split(r"(\\s+)", text)
    decoded_parts: List[str] = []
    transforms: List[Dict[str, str]] = []

    for part in parts:
        if not part:
            continue
        if part.isspace():
            decoded_parts.append(part)
            continue
        subparts = re.split(r"([\\W_]+)", part)
        decoded_subparts: List[str] = []
        for sub in subparts:
            if not sub:
                continue
            if re.fullmatch(r"[\\W_]+", sub):
                decoded_subparts.append(sub)
                continue
            name, decoded, _ = _decode_word(sub)
            decoded_subparts.append(decoded)
            if name != "raw" and decoded != sub and len(transforms) < MAX_TRANSFORMS:
                transforms.append({"token": sub, "transform": name, "decoded": decoded})
        decoded_parts.append("".join(decoded_subparts))

    decoded_text = "".join(decoded_parts)
    decoded_score = _score_text(decoded_text)
    return decoded_text, transforms, raw_score, decoded_score


def _printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    printable = sum(1 for ch in text if ch.isprintable() or ch in {"\n", "\r", "\t"})
    return printable / max(1, len(text))


def _is_candidate(text: str) -> bool:
    if not text:
        return False
    text = text.strip()
    if len(text) < MIN_TOKEN_LEN:
        return False
    if _printable_ratio(text) < 0.5:
        return False
    return True


def _extract_text_candidates(payload: Dict[str, Any]) -> List[str]:
    candidates: List[str] = []
    if not payload or not isinstance(payload, dict):
        return candidates

    output = payload.get("output")
    if isinstance(output, str) and _is_candidate(output):
        candidates.append(output)
    elif isinstance(output, list):
        for item in output:
            if isinstance(item, str) and _is_candidate(item):
                candidates.append(item)

    decoded_text = payload.get("decoded_text")
    if isinstance(decoded_text, dict):
        for val in decoded_text.values():
            if isinstance(val, str) and _is_candidate(val):
                candidates.append(val)

    details = payload.get("details")
    if isinstance(details, dict):
        text_channels = details.get("text_channels")
        if isinstance(text_channels, dict):
            for channel_data in text_channels.values():
                if isinstance(channel_data, dict):
                    val = channel_data.get("text_preview")
                    if isinstance(val, str) and _is_candidate(val):
                        candidates.append(val)

        for key in ("preview", "text", "pref_text", "text_preview", "zlib_preview"):
            val = details.get(key)
            if isinstance(val, str) and _is_candidate(val):
                candidates.append(val)

        best = details.get("best")
        if isinstance(best, dict):
            for key in ("pref_text", "text_preview", "zlib_preview", "text"):
                val = best.get(key)
                if isinstance(val, str) and _is_candidate(val):
                    candidates.append(val)

        for cand in details.get("candidates", []) if isinstance(details.get("candidates"), list) else []:
            if not isinstance(cand, dict):
                continue
            for key in ("pref_text", "text_preview", "zlib_preview", "text"):
                val = cand.get(key)
                if isinstance(val, str) and _is_candidate(val):
                    candidates.append(val)

    return candidates


def _collect_sources(results: Dict[str, Any]) -> List[Tuple[str, str]]:
    sources: List[Tuple[str, str]] = []
    seen: set[str] = set()
    source_keys = [
        "advanced_lsb",
        "lsb",
        "pvd",
        "dct",
        "f5",
        "chroma",
        "simple_lsb",
        "simple_rgb",
        "red_plane",
        "green_plane",
        "blue_plane",
        "alpha_plane",
        "strings",
        "stegg",
    ]
    for key in source_keys:
        payload = results.get(key)
        for candidate in _extract_text_candidates(payload):
            cleaned = candidate.strip()
            if cleaned in seen:
                continue
            seen.add(cleaned)
            sources.append((key, cleaned))
            if len(sources) >= 12:
                return sources
    return sources


def analyze_randomizer_decode(input_img: Path, output_dir: Path) -> None:
    """Attempt to reverse randomizer-style transforms on decoded text sources."""
    results_path = output_dir / "results.json"
    if not results_path.exists():
        update_data(
            output_dir,
            {
                "randomizer_decode": {
                    "status": "skipped",
                    "reason": "No results to scan yet.",
                }
            },
        )
        return

    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
    except Exception as exc:
        update_data(
            output_dir,
            {
                "randomizer_decode": {
                    "status": "error",
                    "error": f"Failed to read results.json: {exc}",
                }
            },
        )
        return

    sources = _collect_sources(results)
    if not sources:
        update_data(
            output_dir,
            {
                "randomizer_decode": {
                    "status": "no_signal",
                    "summary": "No candidate text sources for randomizer decoding.",
                }
            },
        )
        return

    best: Optional[Dict[str, Any]] = None
    suspect_samples: List[Dict[str, str]] = []
    for source, text in sources:
        decoded, transforms, raw_score, decoded_score = _decode_randomizer_text(text)
        delta = decoded_score - raw_score
        if _is_candidate(text) and raw_score < 0.35:
            suspect_samples.append({"source": source, "preview": text[:180]})
        payload = {
            "source": source,
            "raw_score": round(raw_score, 3),
            "decoded_score": round(decoded_score, 3),
            "delta": round(delta, 3),
            "decoded_preview": decoded[:240],
            "transforms": transforms[:12],
            "raw_preview": text[:240],
        }
        if best is None or (delta, decoded_score) > (best["delta"], best["decoded_score"]):
            best = payload

    if not best:
        update_data(
            output_dir,
            {
                "randomizer_decode": {
                    "status": "no_signal",
                    "summary": "Randomizer decode produced no candidates.",
                }
            },
        )
        return

    has_flag = bool(SUSPECT_RE.search(best.get("decoded_preview", "")))
    status = "ok" if has_flag or best["delta"] >= 0.15 else "no_signal"
    summary = "Randomizer decode improved readability."
    if has_flag:
        summary = "Potential flag pattern detected after randomizer decode."
    elif status == "no_signal":
        summary = "No clear plaintext; randomizer decode did not significantly improve."

    update_data(
        output_dir,
        {
            "randomizer_decode": {
                "status": status,
                "summary": summary,
                "details": {
                    "best_source": best["source"],
                    "raw_score": best["raw_score"],
                    "decoded_score": best["decoded_score"],
                    "delta": best["delta"],
                    "decoded_preview": best["decoded_preview"],
                    "raw_preview": best["raw_preview"],
                    "transforms": best["transforms"],
                    "suspect_samples": suspect_samples[:5],
                },
            }
        },
    )
