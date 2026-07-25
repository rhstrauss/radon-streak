#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ryder H. Strauss
"""Survey-agnostic per-exposure detection chain:

  FRT candidates (pre-whitened) -> MF grid refinement -> prefit RB gate ->
  Veres fit -> RB vetting -> dipole-flank cut.

Consumed by bin/run_g96_sequence.py, bin/measure_efficiency.py, bin/run_ztf.py.
"""
import time

import numpy as np

from . import rb
from .frt_driver import detect_streaks
from .mf_snr import refine_candidate
from .trail_fit import fit_trail
from .roughfit import rough_fit


def near_bad_col(cand, mask, reach=12):
    """Candidate adjacent to masked bleed/saturation structure? (enables the
    conditional detector-axis cut on g96)."""
    x, y = int(round(cand["x"])), int(round(cand["y"]))
    sl = (slice(max(0, y - reach), y + reach + 1),
          slice(max(0, x - reach), x + reach + 1))
    return bool((mask[sl] > 0).any())


def rb_config(cfg):
    det = cfg.get("detect", {})
    vet = cfg.get("vet", {})
    return dict(
        mf_snr_min=det.get("mf_snr_min", 6.0),
        len_min_arcsec=vet.get("len_min_arcsec", 6.0),
        len_max_arcsec=vet.get("len_max_arcsec", 300.0),
        chi2r_lo=vet.get("chi2r_lo", 0.3), chi2r_hi=vet.get("chi2r_hi", 3.0),
        sig_along_max_arcsec=vet.get("sig_along_max_arcsec", 4.0),
        edge_px=vet.get("edge_px", 30),
        repeat_rad_arcsec=vet.get("repeat_rad_arcsec", 5.0),
        repeat_pa_deg=vet.get("repeat_pa_deg", 10.0),
        axis_blanket=vet.get("axis_blanket", None),
    )


def process_exposure(e, cfg, survey="g96", verbose=True, fast=None):
    """Returns list of surviving fit dicts (each with mf_snr + cand info).

    fast : survey-scale fast path -- skip the per-candidate Veres LSQ (~917 ms
        each, ~35% of pipeline runtime) and report the FRT+MF rough trail with
        approximate cross/along-track errors, which is already well inside
        heliolinc's tracklet gates (measured: PA 0.47 deg, length 1.9% vs
        truth). Run the Veres fit as a POST-LINKING refinement instead.
        None -> cfg['detect']['fast'] (default False = legacy behaviour).
    """
    det = cfg.get("detect", {})
    rb_cfg = rb_config(cfg)
    if fast is None:
        fast = bool(det.get("fast", False))
    t0 = time.time()
    cands = detect_streaks(e.white, e.psf_sigma_px,
                           tile=det.get("tile", 1024),
                           overlap=det.get("overlap", 128),
                           min_length=det.get("min_length", 8),
                           threshold=det.get("frt_threshold", 5.0),
                           fast_suppress=det.get("fast_suppress", True),
                           backend=det.get("backend", "native"))
    t_frt = time.time() - t0
    fits_out, reasons = [], {}
    for c in cands:
        c["near_bad_col"] = near_bad_col(c, e.mask)
        ref = refine_candidate(e.white, c["x"], c["y"], c["pa_rad"], e.psf_sigma_px)
        c.update(snr=ref["snr"], pa_px_deg=np.degrees(ref["pa_rad"]))
        ok, why = rb.prefit_pass(c, rb_cfg, survey=survey)
        if not ok:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        if fast:
            fit = rough_fit(e.diff, e.mask, e.wcs, ref, e.psf_sigma_px, e.magzp,
                            pixscale_arcsec=getattr(e, "pixscale_arcsec", None))
        else:
            fit = fit_trail(e.diff, e.mask, e.wcs, ref["x"], ref["y"], e.psf_sigma_px,
                            e.magzp, theta0=ref["pa_rad"], h0=ref["L"] / 2.0)
        ok, why = rb.passes_rb(fit, c, imshape=e.diff.shape, cfg=rb_cfg, survey=survey)
        if not ok:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        fr = rb.flank_ratio(e.diff, fit, e.psf_sigma_px, e.mask)
        if fr < rb.DEFAULTS["flank_ratio_max"]:
            reasons["dipole_flank"] = reasons.get("dipole_flank", 0) + 1
            continue
        fit["mf_snr"] = ref["snr"]
        fit["cand"] = {k: c[k] for k in ("x", "y", "pa_rad", "L_frt", "snr_frt")}
        fits_out.append(fit)
    if verbose:
        print(f"  exp{e.image_index}: {len(cands)} FRT cands ({t_frt:.0f}s), "
              f"{len(fits_out)} pass fit+RB; rejects: {reasons}")
    return fits_out
