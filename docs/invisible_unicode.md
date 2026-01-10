# Invisible Unicode Sweep

The Invisible Unicode Sweep scans:
- Raw bytes of the uploaded file (entire file, byte-for-byte)
- Textual outputs from analyzers (results.json)

It focuses on invisible or near-invisible Unicode characters that are commonly used to encode hidden payloads.

## Tiers

Tier0 (default on):
- Canonical invisible cipher characters: ZWSP/ZWNJ/ZWJ/WJ, BOM/ZWNBSP, SHY, CGJ
- Bidi controls and invisible operators (U+2060..U+2064)
- Tag characters (U+E0000..U+E007F)

Tier2 (default on):
- Variation selectors (U+FE00..U+FE0F and U+E0100..U+E01EF)

Tier1 (opt-in):
- Broad Cc/Cf categories beyond Tier0 (format/control characters)

Separators (opt-in):
- Zs/Zl/Zp (whitespace and separators; can add noise)

## Raw Scan Aggressiveness

Raw decoding is gated to avoid false positives:
- low: strict decoding (few replacement chars, high printable ratio)
- balanced: default
- high: permissive decoding (use with caution)

UTF-16/UTF-32 decoding is only attempted when BOMs or NUL-byte patterns suggest it.

## Output Fields

- counts_by_tier: total counts per tier
- counts_by_tag: total counts per tag (zero_width, variation_selector, bidi, etc.)
- samples: bounded context windows with markers like `⟦U+200B ZERO WIDTH SPACE⟧`
- confidence + reasons: summarized evidence, penalized for noisy decodes

## References (external)

- https://github.com/paulgb/emoji-encoder
- https://elder-plinius.github.io/P4RS3LT0NGV3/
