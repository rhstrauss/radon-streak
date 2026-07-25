# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ryder H. Strauss
"""streakradon — FRT-based trailed-source detection for solar-system fast movers.

Architecture: "FRT finds, Veres fit measures."
  1. Our own clean-room Fast Radon Transform (streakradon.frt) locates streak
     candidates on pre-whitened difference tiles (permissive threshold, physical
     min_length; multi-length foldings built in).
  2. Each candidate is re-measured with the validated Veres line(x)Gaussian
     forward fit (trail_fit) and scored with our integrated matched-filter SNR
     (mf_snr).
  3. Real/bogus vetting (rb) + cross-exposure repetition test.
  4. Output: hldet_colformat01 CSV for heliolinx make_trailed_tracklets.

The FRT (streakradon/frt.py) is an INDEPENDENT clean-room implementation of the
recursive-doubling discrete Radon transform, written from the published
algorithm (Gotz-Druckmuller 1996; Brady 1998; Nir-Ofek 2018) and not derived
from any existing source. See LICENSING.md / NOTICE.

Provenance: detection/measurement modules lifted from the validated ztf_streak
pipeline (/astro/store/shire/rstrau/ztf_streak, 2026-06).
"""
