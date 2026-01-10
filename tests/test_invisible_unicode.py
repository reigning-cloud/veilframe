import json
import random

from engine.analyzers.invisible_unicode import analyze_invisible_unicode


def run_scan(tmp_path, raw_bytes, **kwargs):
    input_path = tmp_path / "sample.bin"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    input_path.write_bytes(raw_bytes)
    analyze_invisible_unicode(input_path, output_dir, enabled=True, **kwargs)
    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    return results["invisible_unicode"]


def test_tier0_stream_scores_high(tmp_path):
    zwsp = "\u200b"
    zwnj = "\u200c"
    text = "hello" + (zwsp + zwnj) * 20 + "world"
    result = run_scan(tmp_path, text.encode("utf-8"))

    assert result["status"] == "ok"
    assert result["confidence"] >= 0.6

    counts = result["details"]["counts_by_tier"]
    assert counts.get("tier0", 0) >= 40

    samples = result["details"]["sources"][0]["samples"]
    assert any("⟦U+200B" in sample["visualized"] for sample in samples)


def test_tier2_variation_scores_high(tmp_path):
    vs16 = "\ufe0f"
    text = ("A" + vs16 + "B" + vs16) * 16
    result = run_scan(tmp_path, text.encode("utf-8"))

    assert result["status"] == "ok"
    assert result["confidence"] >= 0.6
    assert result["details"]["counts_by_tier"].get("tier2", 0) >= 16


def test_emoji_vs16_low_confidence(tmp_path):
    text = "hello \u2764\ufe0f \u2600\ufe0f \u26a0\ufe0f"
    result = run_scan(tmp_path, text.encode("utf-8"))

    assert result["status"] != "ok"
    assert result["confidence"] < 0.3
    assert result["details"]["counts_by_tier"].get("tier2", 0) >= 1


def test_random_bytes_gating(tmp_path):
    rng = random.Random(1337)
    raw_bytes = bytes(rng.getrandbits(8) for _ in range(4096))
    result = run_scan(tmp_path, raw_bytes)

    assert result["status"] != "ok"
    assert result["confidence"] < 0.3

    raw_decodes = result["details"]["raw_decodes"]
    utf16 = next((item for item in raw_decodes if item.get("encoding") == "utf-16"), None)
    utf32 = next((item for item in raw_decodes if item.get("encoding") == "utf-32"), None)

    assert utf16 is not None and utf16.get("attempted") is False
    assert utf32 is not None and utf32.get("attempted") is False
