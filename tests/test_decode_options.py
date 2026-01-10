from pathlib import Path

from engine.decode_registry import get_registry


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _assert_schema(result):
    for key in (
        "option_id",
        "label",
        "status",
        "confidence",
        "summary",
        "details",
        "artifacts",
        "timing_ms",
    ):
        assert key in result
    assert isinstance(result["confidence"], float)


def _run_option(option_id, fixture_name, password=None):
    registry = get_registry()
    option = registry[option_id]
    if option_id == "auto_detect":
        return option["analyzer"](
            FIXTURES / fixture_name,
            option_id=option["id"],
            label=option["label"],
            registry=registry,
            password=password,
        )
    params = {"password": password}
    return option["analyzer"](FIXTURES / fixture_name, **option["params"](option, params))


def test_lsb_option():
    result = _run_option("lsb", "lsb.png")
    _assert_schema(result)
    assert result["status"] == "ok"
    assert "LSB_OK" in result["details"].get("preview", "")


def test_pvd_option():
    result = _run_option("pvd", "pvd.png")
    _assert_schema(result)
    assert result["status"] == "ok"
    assert "PVD_OK" in result["details"].get("preview", "")


def test_palette_option():
    result = _run_option("palette", "palette.png")
    _assert_schema(result)
    assert result["status"] == "ok"
    assert "PAL_OK" in result["details"].get("preview", "")


def test_chroma_option():
    result = _run_option("chroma", "chroma.png")
    _assert_schema(result)
    assert result["status"] == "ok"
    assert "CHROMA_OK" in result["details"].get("preview", "")


def test_png_chunks_option():
    result = _run_option("png_chunks", "png_chunks.png")
    _assert_schema(result)
    assert result["status"] == "ok"
    texts = result["details"].get("text", [])
    joined = " ".join(item.get("text", "") for item in texts)
    assert "PNG_CHUNK_OK" in joined


def test_spread_spectrum_option():
    result = _run_option("spread_spectrum", "spread.png", password="veilframe")
    _assert_schema(result)
    assert result["status"] == "ok"
    assert "SPREAD_OK" in result["details"].get("preview", "")


def test_dct_option():
    result = _run_option("dct", "dct.jpg")
    _assert_schema(result)
    assert result["status"] == "ok"
    assert "DCT_OK" in result["details"].get("preview", "")


def test_f5_option():
    result = _run_option("f5", "f5.jpg")
    _assert_schema(result)
    assert result["status"] == "ok"
    assert "F5_OK" in result["details"].get("preview", "")


def test_auto_detect_option():
    result = _run_option("auto_detect", "lsb.png", password="veilframe")
    _assert_schema(result)
    assert result["status"] == "ok"
    candidates = result["details"].get("candidates", [])
    assert candidates
