#!/usr/bin/env python
"""Diagnose the Phase-1 A/B: why do legacy and fast detection lists differ, and
where does the remaining fast-path runtime go?"""
import os, sys, time, glob
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from streakradon.adapters.g96 import prepare_sequence            # noqa: E402
from streakradon.pipeline import process_exposure, rb_config     # noqa: E402
from streakradon.frt_driver import detect_streaks                # noqa: E402
from streakradon.mf_snr import refine_candidate                  # noqa: E402
from streakradon.trail_fit import fit_trail                      # noqa: E402
from streakradon.roughfit import rough_fit                       # noqa: E402
from streakradon import rb                                       # noqa: E402
from streakradon.fastmask import (install_fast_suppression,      # noqa: E402
                                  uninstall_fast_suppression)

FRAMES = ("/astro/store/shire/rstrau/css_pds_pilot/data/N25057_20240501/"
          "G96_20240501_2B_N25057_01_000?.arch.fz")


def sep(a, b):
    return np.hypot((a["ra"] - b["ra"]) * np.cos(np.radians(a["dec"])),
                    a["dec"] - b["dec"]) * 3600.0


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "g96.yaml")))
    exps = prepare_sequence(sorted(glob.glob(FRAMES)), cfg, verbose=False)
    e = exps[0]

    # --- stage timing of the FAST path on one exposure ---
    install_fast_suppression()
    det = cfg["detect"]
    t0 = time.time()
    cands = detect_streaks(e.white, e.psf_sigma_px, tile=det["tile"],
                           overlap=det["overlap"], min_length=det["min_length"],
                           threshold=det["frt_threshold"], fast_suppress=True)
    t_det = time.time() - t0
    rbc = rb_config(cfg)
    t_ref = t_rough = t_rb = 0.0
    kept = 0
    for c in cands:
        t = time.time()
        ref = refine_candidate(e.white, c["x"], c["y"], c["pa_rad"], e.psf_sigma_px)
        t_ref += time.time() - t
        c.update(snr=ref["snr"], pa_px_deg=np.degrees(ref["pa_rad"]),
                 near_bad_col=False)
        ok, _ = rb.prefit_pass(c, rbc, survey="g96")
        if not ok:
            continue
        t = time.time()
        f = rough_fit(e.diff, e.mask, e.wcs, ref, e.psf_sigma_px, e.magzp)
        t_rough += time.time() - t
        t = time.time()
        ok, _ = rb.passes_rb(f, c, imshape=e.diff.shape, cfg=rbc, survey="g96")
        if ok:
            rb.flank_ratio(e.diff, f, e.psf_sigma_px, e.mask)
            kept += 1
        t_rb += time.time() - t
    print("FAST-PATH stage timing, exposure 0:")
    print(f"  detect_streaks (FRT+suppress): {t_det:7.1f}s   ({len(cands)} candidates)")
    print(f"  refine_candidate total       : {t_ref:7.1f}s   ({t_ref/max(len(cands),1)*1000:.0f} ms/cand)")
    print(f"  rough_fit total              : {t_rough:7.1f}s")
    print(f"  rb vetting + flank           : {t_rb:7.1f}s")
    print(f"  TOTAL                        : {t_det+t_ref+t_rough+t_rb:7.1f}s  -> {kept} kept")

    # --- detection-list comparison at several tolerances ---
    print("\nDETECTION LIST comparison (exposure 0):")
    uninstall_fast_suppression()
    cl = dict(cfg); cl["detect"] = dict(det); cl["detect"]["fast_suppress"] = False
    L = process_exposure(e, cl, verbose=False, fast=False)
    install_fast_suppression()
    cf = dict(cfg); cf["detect"] = dict(det); cf["detect"]["fast_suppress"] = True
    F = process_exposure(e, cf, verbose=False, fast=True)
    print(f"  legacy kept {len(L)}   fast kept {len(F)}")
    for tol in (1, 2, 3, 5, 10, 30):
        m = sum(1 for a in L if any(sep(a, b) <= tol for b in F))
        print(f"    tol {tol:3d}\": {m}/{len(L)} legacy detections have a fast counterpart")
    print("\n  legacy-only detections (no fast counterpart within 30\"):")
    for a in L:
        if not any(sep(a, b) <= 30 for b in F):
            print(f"    RA {a['ra']:.4f} Dec {a['dec']:+.4f} L {a['trail_len']:6.1f}\" "
                  f"snr {a.get('mf_snr', 0):5.1f} chi2r {a['chi2r']:.2f}")
    print("\n  fast-only detections (no legacy counterpart within 30\"):")
    for b in F:
        if not any(sep(a, b) <= 30 for a in L):
            print(f"    RA {b['ra']:.4f} Dec {b['dec']:+.4f} L {b['trail_len']:6.1f}\" "
                  f"snr {b.get('mf_snr', 0):5.1f}")


if __name__ == "__main__":
    main()
