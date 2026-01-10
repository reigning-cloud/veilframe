from engine.decode_registry import OPTIONS


REQUIRED_IDS = {
    "auto_detect",
    "lsb",
    "pvd",
    "dct",
    "f5",
    "spread_spectrum",
    "palette",
    "chroma",
    "png_chunks",
}

EXPECTED_LABELS = {
    "auto_detect": "Auto Detect",
    "lsb": "LSB (Least Significant Bit)",
    "pvd": "PVD (Pixel Value Differencing)",
    "dct": "DCT (Frequency domain)",
    "f5": "F5 (JPEG domain)",
    "spread_spectrum": "Spread Spectrum (Password-based)",
    "palette": "Palette (Color index encoding)",
    "chroma": "Chroma (Color channel hiding)",
    "png_chunks": "PNG Chunks (Metadata)",
}


def test_registry_contains_required_options():
    assert set(OPTIONS.keys()) == REQUIRED_IDS


def test_registry_labels_match_expected():
    for option_id, option in OPTIONS.items():
        assert option.label == EXPECTED_LABELS[option_id]


def test_registry_analyzers_are_distinct():
    analyzers = [option.analyzer for option in OPTIONS.values()]
    assert len(set(analyzers)) == len(analyzers)
