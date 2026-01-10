"""Invisible Unicode scan across raw bytes and analyzer outputs."""

from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .utils import update_data

TABLE_JSON_PATH = (
    Path(__file__).resolve().parents[2] / "references" / "invisible_unicode_table.json"
)
TABLE_TXT_PATH = "references/invisible_unicode_table.txt"
DOC_PATH = "docs/invisible_unicode.md"

MAX_SOURCE_TEXT = 200_000
MAX_MATCHES = 40
MAX_SAMPLES = 5
MAX_SAMPLE_LEN = 200
CONTEXT_WINDOW = 48

AGGRESSIVENESS_PRESETS = {
    "low": {
        "replacement_ratio_max": 0.01,
        "printable_ratio_min": 0.72,
        "utf16_null_ratio": 0.35,
        "utf16_null_dominance": 0.65,
        "utf32_null_ratio": 0.6,
    },
    "balanced": {
        "replacement_ratio_max": 0.02,
        "printable_ratio_min": 0.6,
        "utf16_null_ratio": 0.25,
        "utf16_null_dominance": 0.6,
        "utf32_null_ratio": 0.55,
    },
    "high": {
        "replacement_ratio_max": 0.05,
        "printable_ratio_min": 0.45,
        "utf16_null_ratio": 0.18,
        "utf16_null_dominance": 0.55,
        "utf32_null_ratio": 0.45,
    },
}

REPLACEMENT_CHAR = "\ufffd"


def _normalize_aggressiveness(value: Optional[str]) -> str:
    if not value:
        return "balanced"
    candidate = str(value).strip().lower()
    return candidate if candidate in AGGRESSIVENESS_PRESETS else "balanced"


@lru_cache(maxsize=1)
def _load_table() -> Dict[int, Dict[str, object]]:
    if not TABLE_JSON_PATH.exists():
        raise FileNotFoundError(
            f"Missing invisible Unicode table: {TABLE_JSON_PATH}. "
            "Run scripts/generate_invisible_unicode_table.py"
        )
    data = json.loads(TABLE_JSON_PATH.read_text(encoding="utf-8"))
    entries = data.get("entries", [])
    table: Dict[int, Dict[str, object]] = {}
    for entry in entries:
        cp = entry.get("codepoint")
        if cp is None:
            continue
        table[int(cp)] = entry
    return table


@lru_cache(maxsize=1)
def _tier_sets() -> Dict[str, set[int]]:
    table = _load_table()
    tiers: Dict[str, set[int]] = {
        "tier0": set(),
        "tier1": set(),
        "tier2": set(),
        "separators": set(),
    }
    for cp, entry in table.items():
        tier = entry.get("tier")
        if tier in tiers:
            tiers[tier].add(cp)
    return tiers


@lru_cache(maxsize=16)
def _utf8_sequences_for_tiers(tiers: Tuple[str, ...]) -> List[Tuple[int, bytes]]:
    table = _load_table()
    sequences: List[Tuple[int, bytes]] = []
    for cp, entry in table.items():
        if entry.get("tier") in tiers:
            sequences.append((cp, chr(cp).encode("utf-8")))
    return sequences


def _iter_strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)


def _collect_text(value: object, max_chars: int = MAX_SOURCE_TEXT) -> str:
    chunks: List[str] = []
    total = 0
    for text in _iter_strings(value):
        if not text:
            continue
        if total >= max_chars:
            break
        remaining = max_chars - total
        snippet = text[:remaining]
        chunks.append(snippet)
        total += len(snippet)
    return "\n".join(chunks)


def _printable_ratio(text: str, table: Dict[int, Dict[str, object]]) -> float:
    if not text:
        return 0.0
    printable = 0
    for ch in text:
        if ch.isprintable() or ch in {"\n", "\r", "\t"} or ord(ch) in table:
            printable += 1
    return printable / max(1, len(text))


def _decode_raw_bytes(
    raw_bytes: bytes,
    encoding: str,
    preset: dict,
    table: Dict[int, Dict[str, object]],
) -> Tuple[Optional[str], dict]:
    text = raw_bytes.decode(encoding, errors="replace")
    replacement_count = text.count(REPLACEMENT_CHAR)
    replacement_ratio = replacement_count / max(1, len(text))
    printable_ratio = _printable_ratio(text, table)
    metrics = {
        "encoding": encoding,
        "length": len(text),
        "replacement_count": replacement_count,
        "replacement_ratio": round(replacement_ratio, 4),
        "printable_ratio": round(printable_ratio, 4),
    }
    if replacement_ratio > preset["replacement_ratio_max"]:
        metrics["reject_reason"] = "high replacement chars"
        return None, metrics
    if printable_ratio < preset["printable_ratio_min"]:
        metrics["reject_reason"] = "low printable ratio"
        return None, metrics
    return text, metrics


def _null_byte_ratio(raw_bytes: bytes) -> float:
    if not raw_bytes:
        return 0.0
    return raw_bytes.count(0) / len(raw_bytes)


def _should_try_utf16(raw_bytes: bytes, preset: dict) -> bool:
    if raw_bytes.startswith(b"\xff\xfe") or raw_bytes.startswith(b"\xfe\xff"):
        return True
    ratio = _null_byte_ratio(raw_bytes)
    if ratio < preset["utf16_null_ratio"]:
        return False
    even_nulls = sum(1 for i in range(0, len(raw_bytes), 2) if raw_bytes[i] == 0)
    odd_nulls = sum(1 for i in range(1, len(raw_bytes), 2) if raw_bytes[i] == 0)
    dominant = max(even_nulls, odd_nulls) / max(1, even_nulls + odd_nulls)
    return dominant >= preset["utf16_null_dominance"]


def _should_try_utf32(raw_bytes: bytes, preset: dict) -> bool:
    if raw_bytes.startswith(b"\xff\xfe\x00\x00") or raw_bytes.startswith(b"\x00\x00\xfe\xff"):
        return True
    return _null_byte_ratio(raw_bytes) >= preset["utf32_null_ratio"]


def _strip_invisible(text: str, table: Dict[int, Dict[str, object]]) -> str:
    return "".join(ch for ch in text if ord(ch) not in table)


def _render_marker(codepoints: List[int], table: Dict[int, Dict[str, object]]) -> str:
    markers: List[str] = []
    for cp in codepoints:
        entry = table.get(cp)
        if not entry:
            continue
        name = entry.get("name", "")
        code = entry.get("code", f"U+{cp:04X}")
        markers.append(f"⟦{code} {name}⟧")
    if len(markers) > 8:
        markers = markers[:6] + ["…"] + markers[-2:]
    return "".join(markers)


def _build_samples(
    text: str,
    runs: List[Tuple[int, int, List[int], bool]],
    table: Dict[int, Dict[str, object]],
) -> List[Dict[str, str]]:
    samples: List[Dict[str, str]] = []
    for start, end, codepoints, has_signal in runs:
        if not has_signal:
            continue
        left = text[max(0, start - CONTEXT_WINDOW) : start]
        right = text[end : end + CONTEXT_WINDOW]
        left = _strip_invisible(left, table)
        right = _strip_invisible(right, table)
        marker = _render_marker(codepoints, table)
        visual = f"{left}{marker}{right}"
        if len(visual) > MAX_SAMPLE_LEN:
            trim = max(0, len(visual) - MAX_SAMPLE_LEN)
            left_trim = min(len(left), trim // 2)
            right_trim = min(len(right), trim - left_trim)
            left = left[left_trim:]
            right = right[: max(0, len(right) - right_trim)]
            visual = f"{left}{marker}{right}"
        samples.append(
            {
                "left_context": left,
                "marker": marker,
                "right_context": right,
                "visualized": visual,
            }
        )
        if len(samples) >= MAX_SAMPLES:
            break
    return samples


def _scan_text(
    text: str,
    source: str,
    table: Dict[int, Dict[str, object]],
    enabled_tiers: set[str],
) -> Tuple[Counter, Dict[str, object], List[Dict[str, str]], List[int]]:
    counts: Counter = Counter()
    counts_by_tier: Counter = Counter()
    counts_by_tag: Counter = Counter()
    patterns: List[Dict[str, str]] = []

    signal_positions: List[int] = []
    signal_codepoints: set[int] = set()
    max_signal_run = 0
    current_signal_run = 0
    vs_pairs = 0
    vs_bases: set[str] = set()

    runs: List[Tuple[int, int, List[int], bool]] = []
    run_start: Optional[int] = None
    run_codepoints: List[int] = []
    run_has_signal = False

    for idx, ch in enumerate(text):
        cp = ord(ch)
        entry = table.get(cp)
        if entry:
            counts[cp] += 1
            tier = entry.get("tier", "")
            counts_by_tier[tier] += 1
            for tag in entry.get("tags", []):
                counts_by_tag[tag] += 1
            if tier in enabled_tiers:
                signal_positions.append(idx)
                signal_codepoints.add(cp)
                current_signal_run += 1
                max_signal_run = max(max_signal_run, current_signal_run)
                run_has_signal = True
            else:
                current_signal_run = 0

            if "variation_selector" in entry.get("tags", []) and idx > 0:
                vs_pairs += 1
                vs_bases.add(text[idx - 1])

            if run_start is None:
                run_start = idx
            run_codepoints.append(cp)
        else:
            current_signal_run = 0
            if run_start is not None:
                runs.append((run_start, idx, run_codepoints, run_has_signal))
            run_start = None
            run_codepoints = []
            run_has_signal = False

    if run_start is not None:
        runs.append((run_start, len(text), run_codepoints, run_has_signal))

    samples = _build_samples(text, runs, table)

    signal_total = sum(counts_by_tier.get(tier, 0) for tier in enabled_tiers)
    distinct_signal = len(signal_codepoints)

    source_info = {
        "source": source,
        "total_chars": len(text),
        "signal_count": int(signal_total),
        "signal_distinct": int(distinct_signal),
        "max_signal_run": int(max_signal_run),
        "vs_pairs": int(vs_pairs),
        "vs_base_distinct": int(len(vs_bases)),
        "counts_by_tier": {k: int(v) for k, v in counts_by_tier.items()},
        "counts_by_tag": {k: int(v) for k, v in counts_by_tag.items()},
        "samples": samples,
    }

    if counts_by_tag.get("zero_width", 0) >= 8 or max_signal_run >= 8:
        patterns.append(
            {
                "label": "zero_width_run",
                "source": source,
                "detail": "zero-width run detected (ZWSP/ZWNJ/ZWJ).",
            }
        )
    if counts_by_tag.get("variation_selector", 0) >= 8:
        patterns.append(
            {
                "label": "variation_selector_stream",
                "source": source,
                "detail": "variation selectors present.",
            }
        )
    if vs_pairs >= 8 and len(vs_bases) <= 4:
        patterns.append(
            {
                "label": "emoji_encoder_pattern",
                "source": source,
                "detail": "variation selectors attached to a small base glyph set.",
            }
        )
    if counts_by_tag.get("tag_character", 0) >= 2:
        patterns.append(
            {
                "label": "tag_character_sequence",
                "source": source,
                "detail": "tag characters present (emoji tag encodings).",
            }
        )
    if counts_by_tag.get("bidi", 0) >= 2:
        patterns.append(
            {
                "label": "bidi_controls",
                "source": source,
                "detail": "bidirectional control characters present.",
            }
        )
    if counts_by_tag.get("invisible_operator", 0) >= 2:
        patterns.append(
            {
                "label": "invisible_operators",
                "source": source,
                "detail": "invisible operator/word-joiner characters present.",
            }
        )

    return counts, source_info, patterns, signal_positions


def _detect_periodic_gap(signal_positions: List[int]) -> Optional[int]:
    if len(signal_positions) < 10:
        return None
    gaps = [signal_positions[i] - signal_positions[i - 1] for i in range(1, len(signal_positions))]
    if not gaps:
        return None
    for candidate in (8, 16, 32):
        matches = sum(1 for gap in gaps if gap == candidate or gap % candidate == 0)
        if matches / len(gaps) >= 0.35:
            return candidate
    return None


def _score_source(
    source_info: Dict[str, object],
    counts: Counter,
    enabled_tiers: set[str],
    decode_metrics: Optional[dict],
) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    score = 0.0

    signal_count = int(source_info.get("signal_count", 0))
    distinct_signal = int(source_info.get("signal_distinct", 0))
    max_run = int(source_info.get("max_signal_run", 0))
    vs_pairs = int(source_info.get("vs_pairs", 0))
    vs_base_distinct = int(source_info.get("vs_base_distinct", 0))
    counts_by_tag = source_info.get("counts_by_tag", {})
    periodic_gap = source_info.get("periodic_gap")

    variation_total = int(counts_by_tag.get("variation_selector", 0))
    zero_width_total = int(counts_by_tag.get("zero_width", 0))
    tag_total = int(counts_by_tag.get("tag_character", 0))
    bidi_total = int(counts_by_tag.get("bidi", 0))
    operator_total = int(counts_by_tag.get("invisible_operator", 0))

    if signal_count >= 8:
        score += 0.25
        reasons.append(f"signal count {signal_count}")
    if signal_count >= 32:
        score += 0.1
        reasons.append("dense signal")
    if distinct_signal in {2, 3, 4} and signal_count >= 16:
        score += 0.2
        reasons.append("restricted alphabet (2-4 symbols)")
    if max_run >= 8:
        score += 0.15
        reasons.append(f"long run length {max_run}")
    if max_run >= 16:
        score += 0.1
    if zero_width_total >= 12:
        score += 0.1
    if variation_total >= 12 and vs_base_distinct <= 4:
        score += 0.2
        reasons.append("variation selectors bound to small base set")
    if vs_pairs >= 8 and vs_base_distinct <= 4:
        score += 0.15
    if tag_total >= 4:
        score += 0.05
    if bidi_total >= 2:
        score += 0.05
    if operator_total >= 2:
        score += 0.05
    if periodic_gap:
        score += 0.1
        reasons.append(f"periodic spacing ~{periodic_gap}")

    vs16_count = counts.get(0xFE0F, 0)
    if variation_total and vs16_count / max(1, variation_total) > 0.85 and vs_base_distinct > 6:
        score *= 0.4
        reasons.append("variation selectors look like normal emoji usage")

    if signal_count < 4:
        score = min(score, 0.2)
        if signal_count > 0:
            reasons.append("very small signal")

    if decode_metrics:
        if decode_metrics.get("reject_reason") == "high replacement chars":
            score *= 0.4
            reasons.append("penalized: decode replacement chars")
        if decode_metrics.get("reject_reason") == "low printable ratio":
            score *= 0.5
            reasons.append("penalized: low printable ratio")

    if not {"tier0", "tier2"}.intersection(enabled_tiers) and signal_count:
        score = min(score, 0.25)
        reasons.append("tier1/separator-only signal")

    return min(1.0, round(score, 3)), reasons


def _scan_raw_utf8_sequences(
    raw_bytes: bytes, tiers: Tuple[str, ...]
) -> Tuple[Counter, List[Tuple[int, int, int]]]:
    sequences = _utf8_sequences_for_tiers(tiers)
    counts: Counter = Counter()
    hits: List[Tuple[int, int, int]] = []
    for cp, seq in sequences:
        start = 0
        while True:
            idx = raw_bytes.find(seq, start)
            if idx == -1:
                break
            counts[cp] += 1
            hits.append((idx, cp, len(seq)))
            start = idx + len(seq)
    hits.sort(key=lambda item: item[0])
    return counts, hits


def _build_raw_samples(
    raw_bytes: bytes, hits: List[Tuple[int, int, int]], table: Dict[int, Dict[str, object]]
) -> List[Dict[str, str]]:
    samples: List[Dict[str, str]] = []
    for offset, cp, length in hits[:MAX_SAMPLES]:
        left_bytes = raw_bytes[max(0, offset - 16) : offset]
        right_bytes = raw_bytes[offset + length : offset + length + 16]
        left = left_bytes.hex(" ")
        right = right_bytes.hex(" ")
        marker = _render_marker([cp], table)
        visual = f"{left}{(' ' if left else '')}{marker}{(' ' if right else '')}{right}"
        samples.append(
            {
                "left_context": left,
                "marker": marker,
                "right_context": right,
                "visualized": visual,
            }
        )
    return samples


def _load_results(output_dir: Path) -> Dict[str, object]:
    results_path = output_dir / "results.json"
    if not results_path.exists():
        return {}
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def analyze_invisible_unicode(
    input_img: Path,
    output_dir: Path,
    enabled: bool = False,
    *,
    tier1: bool = False,
    separators: bool = False,
    aggressiveness: Optional[str] = None,
) -> None:
    """Scan for invisible Unicode across raw bytes and analyzer outputs."""
    if not enabled:
        update_data(
            output_dir,
            {
                "invisible_unicode": {
                    "status": "skipped",
                    "label": "𝐢𝗇𝗏𝐢𝗌𝐢𝖻𝗅𝐞 𝐮𝗇𝐢𝖼𝐨𝖽𝐞 𝗌𝗐𝐞𝐞𝗉 (𝐨𝗉𝐭-𝐢𝗇)",
                    "reason": "Enable Invisible Unicode Sweep to scan raw bytes and outputs",
                }
            },
        )
        return

    if not input_img.exists():
        update_data(
            output_dir,
            {
                "invisible_unicode": {
                    "status": "error",
                    "label": "𝐢𝗇𝗏𝐢𝗌𝐢𝖻𝗅𝐞 𝐮𝗇𝐢𝖼𝐨𝖽𝐞 𝗌𝗐𝐞𝐞𝗉 (𝐨𝗉𝐭-𝐢𝗇)",
                    "error": f"Input image not found: {input_img}",
                }
            },
        )
        return

    try:
        table = _load_table()
    except Exception as exc:
        update_data(
            output_dir,
            {
                "invisible_unicode": {
                    "status": "error",
                    "label": "𝐢𝗇𝗏𝐢𝗌𝐢𝖻𝗅𝐞 𝐮𝗇𝐢𝖼𝐨𝖽𝐞 𝗌𝗐𝐞𝐞𝗉 (𝐨𝗉𝐭-𝐢𝗇)",
                    "error": str(exc),
                }
            },
        )
        return

    raw_bytes = b""
    try:
        raw_bytes = input_img.read_bytes()
    except Exception as exc:
        update_data(
            output_dir,
            {
                "invisible_unicode": {
                    "status": "error",
                    "label": "𝐢𝗇𝗏𝐢𝗌𝐢𝖻𝗅𝐞 𝐮𝗇𝐢𝖼𝐨𝖽𝐞 𝗌𝗐𝐞𝐞𝗉 (𝐨𝗉𝐭-𝐢𝗇)",
                    "error": str(exc),
                }
            },
        )
        return

    enabled_tiers = {"tier0", "tier2"}
    if tier1:
        enabled_tiers.add("tier1")
    if separators:
        enabled_tiers.add("separators")

    preset = AGGRESSIVENESS_PRESETS[_normalize_aggressiveness(aggressiveness)]

    sources: List[Tuple[str, str, Optional[dict]]] = []
    raw_decodes: List[dict] = []

    decoded_text, metrics = _decode_raw_bytes(raw_bytes, "utf-8", preset, table)
    metrics["attempted"] = True
    metrics["accepted"] = decoded_text is not None
    raw_decodes.append(metrics)
    if decoded_text:
        sources.append(("raw:utf8", decoded_text, metrics))

    if _should_try_utf16(raw_bytes, preset):
        for label, enc in [("raw:utf16le", "utf-16-le"), ("raw:utf16be", "utf-16-be")]:
            text, metrics = _decode_raw_bytes(raw_bytes, enc, preset, table)
            metrics["attempted"] = True
            metrics["accepted"] = text is not None
            raw_decodes.append(metrics)
            if text:
                sources.append((label, text, metrics))
    else:
        raw_decodes.append({"encoding": "utf-16", "attempted": False})

    if _should_try_utf32(raw_bytes, preset):
        for label, enc in [("raw:utf32le", "utf-32-le"), ("raw:utf32be", "utf-32-be")]:
            text, metrics = _decode_raw_bytes(raw_bytes, enc, preset, table)
            metrics["attempted"] = True
            metrics["accepted"] = text is not None
            raw_decodes.append(metrics)
            if text:
                sources.append((label, text, metrics))
    else:
        raw_decodes.append({"encoding": "utf-32", "attempted": False})

    results = _load_results(output_dir)
    for key, payload in results.items():
        if key == "invisible_unicode":
            continue
        tool_text = _collect_text(payload)
        if tool_text:
            sources.append((f"results:{key}", tool_text, None))

    raw_signal_tiers = tuple(sorted({"tier0", "tier2"}.intersection(enabled_tiers)))
    raw_counts, raw_hits = _scan_raw_utf8_sequences(raw_bytes, raw_signal_tiers)

    global_counts: Counter = Counter()
    source_summaries: List[Dict[str, object]] = []
    pattern_hits: List[Dict[str, str]] = []
    sources_with_signal: List[str] = []

    for source, text, decode_metrics in sources:
        counts, summary, patterns, signal_positions = _scan_text(
            text, source, table, enabled_tiers
        )
        signal_total = int(summary.get("signal_count", 0))
        max_run = int(summary.get("max_signal_run", 0))
        include_counts = signal_total > 0

        periodic_gap = _detect_periodic_gap(signal_positions)
        if periodic_gap:
            summary["periodic_gap"] = periodic_gap
            patterns.append(
                {
                    "label": "periodic_spacing",
                    "source": source,
                    "detail": f"signal spacing suggests {periodic_gap}-char chunks.",
                }
            )

        confidence, reasons = _score_source(summary, counts, enabled_tiers, decode_metrics)
        summary["confidence"] = confidence
        summary["reasons"] = reasons
        if decode_metrics:
            summary["decode"] = decode_metrics

        if include_counts:
            global_counts.update(counts)
        if signal_total or patterns:
            source_summaries.append(summary)
        if patterns:
            pattern_hits.extend(patterns)
        if signal_total:
            sources_with_signal.append(source)

    has_raw_text_source = any(src.startswith("raw:") for src, _, _ in sources)
    if not has_raw_text_source and raw_hits:
        raw_samples = _build_raw_samples(raw_bytes, raw_hits, table)
        raw_counts_by_tier: Counter = Counter()
        raw_counts_by_tag: Counter = Counter()
        for cp, count in raw_counts.items():
            entry = table.get(cp)
            if not entry:
                continue
            tier = entry.get("tier")
            if tier:
                raw_counts_by_tier[tier] += count
            for tag in entry.get("tags", []):
                raw_counts_by_tag[tag] += count
        source_summaries.append(
            {
                "source": "raw:utf8-bytes",
                "total_chars": len(raw_bytes),
                "signal_count": int(sum(raw_counts.values())),
                "signal_distinct": int(len(raw_counts)),
                "max_signal_run": 0,
                "vs_pairs": 0,
                "vs_base_distinct": 0,
                "counts_by_tier": {k: int(v) for k, v in raw_counts_by_tier.items()},
                "counts_by_tag": {k: int(v) for k, v in raw_counts_by_tag.items()},
                "samples": raw_samples,
                "confidence": 0.2 if raw_counts else 0.0,
                "reasons": ["raw UTF-8 byte sequences detected"],
            }
        )
        global_counts.update(raw_counts)
        sources_with_signal.append("raw:utf8-bytes")

    counts_by_tier: Counter = Counter()
    counts_by_tag: Counter = Counter()
    for cp, count in global_counts.items():
        entry = table.get(cp)
        if not entry:
            continue
        tier = entry.get("tier")
        if tier:
            counts_by_tier[tier] += count
        for tag in entry.get("tags", []):
            counts_by_tag[tag] += count

    matches = [
        {**table[cp], "count": int(count)}
        for cp, count in sorted(global_counts.items(), key=lambda item: item[1], reverse=True)[
            :MAX_MATCHES
        ]
        if cp in table
    ]

    source_summaries.sort(key=lambda item: item.get("source", ""))
    pattern_hits.sort(key=lambda item: (item.get("label", ""), item.get("source", "")))

    confidences = [src.get("confidence", 0.0) for src in source_summaries]
    overall_confidence = round(max(confidences) if confidences else 0.0, 3)
    if len([c for c in confidences if c >= 0.4]) >= 2:
        overall_confidence = min(1.0, round(overall_confidence + 0.05, 3))

    signal_total_enabled = sum(
        int(summary.get("signal_count", 0)) for summary in source_summaries
    )

    sources_with_signal = list(dict.fromkeys(sources_with_signal))

    disabled_notes: List[str] = []
    if not tier1:
        disabled_notes.append("Tier1")
    if not separators:
        disabled_notes.append("Separators")

    if signal_total_enabled == 0:
        status = "empty"
        if disabled_notes:
            summary = (
                "No invisible Unicode detected in enabled tiers "
                f"(off: {', '.join(disabled_notes)})."
            )
        else:
            summary = "No invisible Unicode detected in enabled tiers."
    elif overall_confidence >= 0.45:
        status = "ok"
        summary = (
            f"Signals detected across {len(sources_with_signal)} source(s). "
            f"Tier0: {counts_by_tier.get('tier0', 0)}, "
            f"Tier2: {counts_by_tier.get('tier2', 0)}."
        )
    else:
        status = "no_signal"
        summary = "Invisibles detected, but confidence is low (likely normal formatting)."

    sources_scanned = [src for src, _, _ in sources]
    if raw_signal_tiers:
        sources_scanned.append("raw:utf8-bytes")
    sources_scanned = list(dict.fromkeys(sources_scanned))

    update_data(
        output_dir,
        {
            "invisible_unicode": {
                "status": status,
                "label": "𝐢𝗇𝗏𝐢𝗌𝐢𝖻𝗅𝐞 𝐮𝗇𝐢𝖼𝐨𝖽𝐞 𝗌𝗐𝐞𝐞𝗉 (𝐨𝗉𝐭-𝐢𝗇)",
                "summary": summary,
                "confidence": overall_confidence,
                "details": {
                    "config": {
                        "enabled_tiers": sorted(enabled_tiers),
                        "aggressiveness": _normalize_aggressiveness(aggressiveness),
                    },
                    "bytes_scanned": len(raw_bytes),
                    "sources_scanned": sources_scanned,
                    "sources_with_signal": sources_with_signal,
                    "counts_by_tier": {k: int(v) for k, v in counts_by_tier.items()},
                    "counts_by_tag": {k: int(v) for k, v in counts_by_tag.items()},
                    "matches": matches,
                    "patterns": pattern_hits,
                    "sources": source_summaries,
                    "raw_decodes": raw_decodes,
                    "references": {
                        "table": TABLE_TXT_PATH,
                        "doc": DOC_PATH,
                    },
                },
            }
        },
    )


def _text_ratio(text: str) -> float:
    if not text:
        return 0.0
    printable = sum(1 for ch in text if ch.isprintable() or ch in {"\n", "\r", "\t"})
    return printable / max(1, len(text))


def _bits_to_bytes(bits: List[int]) -> bytes:
    if not bits:
        return b""
    usable = (len(bits) // 8) * 8
    if usable <= 0:
        return b""
    bits = bits[:usable]
    out = bytearray(usable // 8)
    for idx in range(0, usable, 8):
        byte = 0
        for bit in bits[idx : idx + 8]:
            byte = (byte << 1) | int(bit)
        out[idx // 8] = byte
    return bytes(out)


def _decode_bits_payload(bits: List[int]) -> bytes:
    if len(bits) < 8:
        return b""
    if len(bits) >= 16:
        length = int("".join(str(b) for b in bits[:16]), 2)
        remaining = len(bits) - 16
        if 0 < length <= remaining // 8:
            return _bits_to_bytes(bits[16 : 16 + length * 8])
    return _bits_to_bytes(bits)


def _decode_zero_width(text: str) -> List[bytes]:
    zwsp = "\u200b"
    zwnj = "\u200c"
    zwj = "\u200d"
    payloads: List[bytes] = []
    run: List[str] = []
    for ch in text:
        if ch in {zwsp, zwnj, zwj}:
            run.append(ch)
            continue
        if run:
            payloads.extend(_decode_zero_width_run(run))
            run = []
    if run:
        payloads.extend(_decode_zero_width_run(run))
    return payloads


def _decode_zero_width_run(run: List[str]) -> List[bytes]:
    zwsp = "\u200b"
    zwnj = "\u200c"
    zwj = "\u200d"
    segments: List[List[int]] = []
    current: List[int] = []
    for ch in run:
        if ch == zwj:
            if current:
                segments.append(current)
            current = []
            continue
        if ch == zwsp:
            current.append(0)
        elif ch == zwnj:
            current.append(1)
    if current:
        segments.append(current)
    if not segments:
        return []
    payloads: List[bytes] = []
    for bits in segments:
        payload = _decode_bits_payload(bits)
        if payload:
            payloads.append(payload)
    return payloads


def _decode_variation_selectors(text: str) -> List[bytes]:
    vs15 = "\ufe0e"
    vs16 = "\ufe0f"
    bits: List[int] = []
    for ch in text:
        if ch == vs15:
            bits.append(0)
        elif ch == vs16:
            bits.append(1)
    payload = _decode_bits_payload(bits)
    return [payload] if payload else []


def _decode_variation_supplement(text: str) -> List[bytes]:
    payloads: List[bytes] = []
    buffer: List[int] = []
    for ch in text:
        cp = ord(ch)
        if 0xE0100 <= cp <= 0xE01EF:
            buffer.append(cp - 0xE0100)
            continue
        if buffer:
            payloads.append(bytes(buffer))
            buffer = []
    if buffer:
        payloads.append(bytes(buffer))
    return payloads


def _decode_tag_chars(text: str) -> List[bytes]:
    payloads: List[bytes] = []
    buffer: List[int] = []
    for ch in text:
        cp = ord(ch)
        if 0xE0020 <= cp <= 0xE007E:
            buffer.append(cp - 0xE0000)
            continue
        if cp == 0xE007F:
            if buffer:
                payloads.append(bytes(buffer))
                buffer = []
            continue
        if buffer:
            payloads.append(bytes(buffer))
            buffer = []
    if buffer:
        payloads.append(bytes(buffer))
    return payloads


def _collect_decode_sources(
    input_img: Path,
    output_dir: Path,
    table: Dict[int, Dict[str, object]],
    aggressiveness: Optional[str],
) -> List[Tuple[str, str]]:
    preset = AGGRESSIVENESS_PRESETS[_normalize_aggressiveness(aggressiveness)]
    raw_bytes = input_img.read_bytes()
    sources: List[Tuple[str, str]] = []
    decoded_text, _ = _decode_raw_bytes(raw_bytes, "utf-8", preset, table)
    if decoded_text:
        sources.append(("raw:utf8", decoded_text))
    if _should_try_utf16(raw_bytes, preset):
        for label, enc in [("raw:utf16le", "utf-16-le"), ("raw:utf16be", "utf-16-be")]:
            text, _ = _decode_raw_bytes(raw_bytes, enc, preset, table)
            if text:
                sources.append((label, text))
    if _should_try_utf32(raw_bytes, preset):
        for label, enc in [("raw:utf32le", "utf-32-le"), ("raw:utf32be", "utf-32-be")]:
            text, _ = _decode_raw_bytes(raw_bytes, enc, preset, table)
            if text:
                sources.append((label, text))

    results = _load_results(output_dir)
    for key, payload in results.items():
        if key in {"invisible_unicode", "invisible_unicode_decode"}:
            continue
        tool_text = _collect_text(payload)
        if tool_text:
            sources.append((f"results:{key}", tool_text))
    return sources


def analyze_invisible_unicode_decode(
    input_img: Path,
    output_dir: Path,
    enabled: bool = False,
    *,
    aggressiveness: Optional[str] = None,
) -> None:
    if not enabled:
        update_data(
            output_dir,
            {
                "invisible_unicode_decode": {
                    "status": "skipped",
                    "label": "𝐢𝗇𝗏𝐢𝗌𝐢𝖻𝗅𝐞 𝐮𝗇𝐢𝖼𝐨𝖽𝐞 𝖽𝐞𝖼𝐨𝖽𝐞",
                    "summary": "Enable Invisible Unicode Sweep to decode invisible ciphers.",
                    "confidence": 0.0,
                    "details": {},
                    "artifacts": [],
                    "timing_ms": 0,
                }
            },
        )
        return

    try:
        table = _load_table()
    except Exception as exc:
        update_data(
            output_dir,
            {
                "invisible_unicode_decode": {
                    "status": "error",
                    "label": "𝐢𝗇𝗏𝐢𝗌𝐢𝖻𝗅𝐞 𝐮𝗇𝐢𝖼𝐨𝖽𝐞 𝖽𝐞𝖼𝐨𝖽𝐞",
                    "summary": f"Failed to load invisible Unicode table: {exc}",
                    "confidence": 0.0,
                    "details": {},
                    "artifacts": [],
                    "timing_ms": 0,
                }
            },
        )
        return

    sources = _collect_decode_sources(input_img, output_dir, table, aggressiveness)
    if not sources:
        update_data(
            output_dir,
            {
                "invisible_unicode_decode": {
                    "status": "empty",
                    "label": "𝐢𝗇𝗏𝐢𝗌𝐢𝖻𝗅𝐞 𝐮𝗇𝐢𝖼𝐨𝖽𝐞 𝖽𝐞𝖼𝐨𝖽𝐞",
                    "summary": "No text sources available to decode.",
                    "confidence": 0.0,
                    "details": {},
                    "artifacts": [],
                    "timing_ms": 0,
                }
            },
        )
        return

    candidates: List[Dict[str, object]] = []
    seen_texts: set[str] = set()

    for source, text in sources:
        for method, payloads in (
            ("zero_width", _decode_zero_width(text)),
            ("variation_selector_bits", _decode_variation_selectors(text)),
            ("variation_selector_bytes", _decode_variation_supplement(text)),
            ("tag_characters", _decode_tag_chars(text)),
        ):
            for payload in payloads:
                if not payload:
                    continue
                decoded = payload.decode("utf-8", errors="replace").strip()
                ratio = _text_ratio(decoded)
                if ratio < 0.6 and len(decoded) < 6:
                    continue
                if decoded in seen_texts:
                    continue
                seen_texts.add(decoded)
                candidates.append(
                    {
                        "method": method,
                        "source": source,
                        "bytes": len(payload),
                        "text_ratio": round(ratio, 3),
                        "preview": decoded[:240],
                    }
                )

    candidates.sort(key=lambda c: (c["text_ratio"], len(c["preview"])), reverse=True)
    candidates = candidates[:5]

    if not candidates:
        update_data(
            output_dir,
            {
                "invisible_unicode_decode": {
                    "status": "empty",
                    "label": "𝐢𝗇𝗏𝐢𝗌𝐢𝖻𝗅𝐞 𝐮𝗇𝐢𝖼𝐨𝖽𝐞 𝖽𝐞𝖼𝐨𝖽𝐞",
                    "summary": "No invisible Unicode cipher payloads decoded.",
                    "confidence": 0.0,
                    "details": {"sources_scanned": [src for src, _ in sources]},
                    "artifacts": [],
                    "timing_ms": 0,
                }
            },
        )
        return

    best_ratio = max(candidate["text_ratio"] for candidate in candidates)
    summary = f"Decoded {len(candidates)} invisible-cipher payload(s)."
    update_data(
        output_dir,
        {
            "invisible_unicode_decode": {
                "status": "ok",
                "label": "𝐢𝗇𝗏𝐢𝗌𝐢𝖻𝗅𝐞 𝐮𝗇𝐢𝖼𝐨𝖽𝐞 𝖽𝐞𝖼𝐨𝖽𝐞",
                "summary": summary,
                "confidence": round(min(0.95, 0.4 + best_ratio), 3),
                "details": {
                    "sources_scanned": [src for src, _ in sources],
                    "candidates": candidates,
                },
                "artifacts": [
                    {"type": "decoded_text", "data": candidate["preview"]}
                    for candidate in candidates
                ],
                "timing_ms": 0,
            }
        },
    )
