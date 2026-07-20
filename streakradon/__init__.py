"""streakradon — FRT-based trailed-source detection for solar-system fast movers.

Architecture: "FRT finds, Veres fit measures."
  1. pyradon (vendored, pinned) Fast Radon Transform locates streak candidates on
     pre-whitened difference tiles (permissive threshold, physical min_length).
  2. Each candidate is re-measured with the validated Veres line(x)Gaussian forward
     fit (trail_fit) and scored with our own integrated matched-filter SNR (mf_snr).
  3. Real/bogus vetting (rb) + cross-exposure repetition test.
  4. Output: hldet_colformat01 CSV for heliolinx make_trailed_tracklets.

Provenance: detection/measurement modules lifted from the validated ztf_streak
pipeline (/astro/store/shire/rstrau/ztf_streak, 2026-06); pyradon benchmark and
patches documented 2026-07 (min_length trap, pre-whitening).
"""
import os
import sys

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYRADON_DIR = os.path.join(PKG_ROOT, "vendor", "pyradon")


def import_pyradon():
    """Make the vendored pyradon importable; return its Finder class."""
    if PYRADON_DIR not in sys.path:
        sys.path.insert(0, PYRADON_DIR)
    from src.finder import Finder  # noqa: E402
    return Finder
