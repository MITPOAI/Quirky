"""
Quirky diagnosis layer.

Turns the passive scalar detector into an *explainable*, *prescriptive* surface:

    maps        -> per-pixel "tell maps" (where the image is too clean / too
                   symmetric / spectrally wrong) + a composite Slop X-ray heatmap.
    report      -> a structured list of named defects with severity, location and a
                   concrete recommended fix ("fix cards" the UI can accept/reject).
    fingerprint -> weight-free guess of *which generator* likely made an image, plus
                   the specific inverse corrections that generator's signature calls for.
    fidelity    -> SSIM + a minimal-edit closed loop that stops as soon as the target
                   is met, so the image is never over-cooked.
    oracle      -> a pluggable *external* detector interface so before/after can be
                   scored against something other than Quirky's own metric.
    remap       -> "map it again": re-diagnose an edited image against the original,
                   reporting resolved/remaining/newly-introduced defects, a red/green
                   delta heatmap, and a multi-round convergence loop.

Everything here is CPU-only classical signal processing unless an optional extra is
explicitly installed. No network, no weights in the core.
"""

from quirky.diagnose.maps import (
    tell_maps,
    composite_slop_map,
    render_heatmap_overlay,
    heatmap_png_bytes,
)
from quirky.diagnose.report import diagnose_image, DEFECT_CATALOG
from quirky.diagnose.fingerprint import fingerprint_image
from quirky.diagnose.fidelity import ssim, humanize_locked
from quirky.diagnose.oracle import (
    DetectorOracle,
    EnsembleHeuristicOracle,
    OracleUnavailable,
    get_oracle,
    audit,
)
from quirky.diagnose.remap import remap_image, remap_loop

__all__ = [
    "tell_maps",
    "composite_slop_map",
    "render_heatmap_overlay",
    "heatmap_png_bytes",
    "diagnose_image",
    "DEFECT_CATALOG",
    "fingerprint_image",
    "ssim",
    "humanize_locked",
    "DetectorOracle",
    "EnsembleHeuristicOracle",
    "OracleUnavailable",
    "get_oracle",
    "audit",
    "remap_image",
    "remap_loop",
]
