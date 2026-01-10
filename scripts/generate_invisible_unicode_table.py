#!/usr/bin/env python3
"""Generate invisible Unicode reference tables (JSON + TXT)."""
from __future__ import annotations

import json
import unicodedata as ud
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "references" / "invisible_unicode_table.json"
OUT_TXT = ROOT / "references" / "invisible_unicode_table.txt"

ZERO_WIDTH = {0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF}
BIDI_CONTROLS = {
    0x061C,
    0x200E,
    0x200F,
    0x202A,
    0x202B,
    0x202C,
    0x202D,
    0x202E,
    0x2066,
    0x2067,
    0x2068,
    0x2069,
}
INVISIBLE_OPERATORS = {0x2060, 0x2061, 0x2062, 0x2063, 0x2064}
TAG_CHAR_RANGE = range(0xE0000, 0xE0080)
VARIATION_SELECTORS = list(range(0xFE00, 0xFE10)) + list(range(0xE0100, 0xE01F0))
MONGOLIAN_FREE_VS = {0x180B, 0x180C, 0x180D}
HANGUL_FILLERS = {0x115F, 0x1160, 0x3164, 0xFFA0}
EXTRA_CODEPOINTS = {0x034F, 0x00AD} | HANGUL_FILLERS

TIER0_EXPLICIT = (
    ZERO_WIDTH
    | INVISIBLE_OPERATORS
    | BIDI_CONTROLS
    | set(TAG_CHAR_RANGE)
    | EXTRA_CODEPOINTS
)

SPECIAL_NOTES: Dict[int, str] = {
    0x00AD: "soft hyphen (renders only when hyphenation is applied)",
    0x034F: "combining grapheme joiner",
    0x061C: "arabic letter mark",
    0x180E: "deprecated mongolian vowel separator",
    0x200B: "zero width space",
    0x200C: "zero width non-joiner",
    0x200D: "zero width joiner",
    0x200E: "left-to-right mark",
    0x200F: "right-to-left mark",
    0x2060: "word joiner",
    0x2061: "invisible function application",
    0x2062: "invisible times",
    0x2063: "invisible separator",
    0x2064: "invisible plus",
    0xFEFF: "zero width no-break space / BOM",
}


def _tier_for(codepoint: int, category: str) -> str | None:
    if codepoint in VARIATION_SELECTORS:
        return "tier2"
    if codepoint in TIER0_EXPLICIT:
        return "tier0"
    if category in {"Cf", "Cc"}:
        return "tier1"
    if category in {"Zs", "Zl", "Zp"}:
        return "separators"
    return None


def _tags_for(codepoint: int, category: str) -> List[str]:
    tags: List[str] = []
    if codepoint in ZERO_WIDTH:
        tags.append("zero_width")
    if codepoint in BIDI_CONTROLS:
        tags.append("bidi")
    if codepoint in INVISIBLE_OPERATORS:
        tags.append("invisible_operator")
    if codepoint in TAG_CHAR_RANGE:
        tags.append("tag_character")
    if codepoint in VARIATION_SELECTORS:
        tags.append("variation_selector")
        if codepoint >= 0xE0100:
            tags.append("variation_selector_supplement")
    if codepoint in MONGOLIAN_FREE_VS:
        tags.append("mongolian_variation_selector")
    if codepoint == 0x00AD:
        tags.append("soft_hyphen")
    if codepoint == 0x034F:
        tags.append("combining_grapheme_joiner")
    if codepoint == 0xFEFF:
        tags.append("bom")
    if codepoint in HANGUL_FILLERS:
        tags.append("hangul_filler")
    if category == "Cf":
        tags.append("format")
    if category == "Cc":
        tags.append("control")
    if category in {"Zs", "Zl", "Zp"}:
        tags.append("separator")
    if ud.combining(chr(codepoint)) > 0:
        tags.append("combining")
    return tags


def _notes_for(codepoint: int, category: str) -> str:
    notes: List[str] = []
    if codepoint in SPECIAL_NOTES:
        notes.append(SPECIAL_NOTES[codepoint])
    if codepoint in VARIATION_SELECTORS:
        notes.append("variation selector")
    if codepoint in TAG_CHAR_RANGE:
        notes.append("tag character")
    if codepoint in MONGOLIAN_FREE_VS:
        notes.append("mongolian free variation selector")
    if category == "Cc":
        notes.append("control")
    if category == "Cf":
        notes.append("format")
    if category in {"Zs", "Zl", "Zp"}:
        notes.append("separator/whitespace")
    if codepoint in HANGUL_FILLERS:
        notes.append("hangul filler")
    return "; ".join(dict.fromkeys(notes)) if notes else ""


def _collect_codepoints() -> Set[int]:
    codepoints: Set[int] = set()
    for cp in range(0x110000):
        ch = chr(cp)
        category = ud.category(ch)
        if category in {"Cc", "Cf", "Zs", "Zl", "Zp"}:
            codepoints.add(cp)
    codepoints.update(EXTRA_CODEPOINTS)
    codepoints.update(TAG_CHAR_RANGE)
    codepoints.update(VARIATION_SELECTORS)
    codepoints.update(MONGOLIAN_FREE_VS)
    return codepoints


def build_entries() -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    for cp in sorted(_collect_codepoints()):
        ch = chr(cp)
        category = ud.category(ch)
        tier = _tier_for(cp, category)
        if tier is None:
            continue
        try:
            name = ud.name(ch)
        except ValueError:
            name = "UNNAMED"
        entry = {
            "codepoint": cp,
            "code": f"U+{cp:04X}",
            "char": ch,
            "category": category,
            "name": name,
            "tier": tier,
            "tags": sorted(_tags_for(cp, category)),
            "bidi_class": ud.bidirectional(ch),
            "combining_class": ud.combining(ch),
            "notes": _notes_for(cp, category),
        }
        entries.append(entry)
    return entries


def write_json(entries: List[Dict[str, object]]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "unicode_version": ud.unidata_version,
        "entries": entries,
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


def write_txt(entries: List[Dict[str, object]]) -> None:
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Invisible Unicode Reference Table",
        "",
        "Scope: Unicode characters that are typically non-printing or zero-width.",
        "Includes categories Cc/Cf and separators Zs/Zl/Zp, plus explicit invisible ranges",
        "(variation selectors, tag characters, Mongolian free variation selectors, combining",
        "grapheme joiner, and Hangul fillers).",
        "",
        "Columns:",
        "- Code: Unicode code point",
        "- Char: literal character (may render as blank)",
        "- Category: Unicode general category",
        "- Name: Unicode name",
        "- Tier: tier0 | tier1 | tier2 | separators",
        "- Tags: classifier tags",
        "- Bidi: bidi class",
        "- Combine: combining class",
        "- Notes: additional context",
        "",
        f"Total entries: {len(entries)}",
        "",
        "Code\tChar\tCategory\tName\tTier\tTags\tBidi\tCombine\tNotes",
    ]
    with open(OUT_TXT, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(f"{line}\n")
        for entry in entries:
            tags = ",".join(entry["tags"])
            row = (
                f"{entry['code']}\t{entry['char']}\t{entry['category']}\t"
                f"{entry['name']}\t{entry['tier']}\t{tags}\t"
                f"{entry['bidi_class']}\t{entry['combining_class']}\t{entry['notes']}"
            )
            f.write(f"{row}\n")


def main() -> None:
    entries = build_entries()
    write_json(entries)
    write_txt(entries)


if __name__ == "__main__":
    main()
