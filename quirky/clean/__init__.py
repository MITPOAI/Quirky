"""
Quirky clean layer — the metadata half of the "clean" stage.

Zero new heavy dependencies: piexif (pure Python, MIT) + Pillow, both already
core dependencies. No cryptographic provenance here (that's C2PA territory — a
deliberate future decision documented in docs/QUIRKY_LAYERS_BLUEPRINT.md); this
is a plain scrub that closes the loop on what `prompt_leak_score` already detects.
"""

from quirky.clean.metadata import scan_metadata, clean_metadata, LEAK_KEYS

__all__ = ["scan_metadata", "clean_metadata", "LEAK_KEYS"]
