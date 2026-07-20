#!/usr/bin/env python
"""P7: run streak_radon on real ATLAS difference stamps of the known NEO 2024 KV
and compare our measured trail LENGTH and PA against ATLAS's own trailed-source
catalog (kv_matches.csv = the trailmain truth).

The ATLAS stamp WCS is broken for absolute astrometry (TPV+SIP conflict, CRPIX
thousands of px outside the 400x400 stamp), so we work in PIXEL space: each
stamp is centered on KV's catalogued position, so KV is the trail nearest the
stamp center. We measure its length (px -> arcsec via the local CD scale) and
pixel-frame PA, and compare against the catalog traillen/celpa. ATLAS uses a
different trail estimator, so exact agreement isn't expected -- metric
consistency (length to ~few arcsec, PA to ~10 deg mod 180) is the test.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from streakradon.adapters.atlas import prepare_stamp        # noqa: E402
from streakradon.frt_driver import detect_streaks           # noqa: E402
from streakradon.mf_snr import refine_candidate             # noqa: E402
from streakradon.trail_fit import fit_trail                 # noqa: E402
from streakradon import rb                                  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATL = os.path.join(ROOT, "work", "atlas")


def load_truth():
    rows = {}
    for ln in open(os.path.join(ATL, "kv_matches.csv")).read().strip().split("\n")[1:]:
        f = ln.split(",")
        # ID like 04a60507o0350o00291 -> exposure key 04a60507o0350o
        expkey = f[10][:14]
        rows.setdefault(expkey, []).append(dict(
            mjd=float(f[0]), ra=float(f[1]), dec=float(f[2]), mag=float(f[3]),
            traillen=float(f[4]), celpa=float(f[5]), obscode=f[12]))
    return rows


def main():
    truth = load_truth()
    diffs = sorted(glob.glob(os.path.join(ATL, "stamps", "*", "diff", "*_diff.fits")))
    print(f"{len(diffs)} ATLAS difference stamps; {sum(len(v) for v in truth.values())} "
          f"catalogued 2024 KV detections")
    results = []
    seen_expkeys = set()
    for path in diffs:
        expkey = os.path.basename(path).replace("_diff.fits", "")[:14]
        if expkey in seen_expkeys:
            continue  # one stamp per exposure (dirs duplicate the same frame)
        seen_expkeys.add(expkey)
        tlist = truth.get(expkey, [])
        if not tlist:
            continue
        t = tlist[0]  # KV detection catalogued in this exposure
        e = prepare_stamp(path)
        cx0, cy0 = e.diff.shape[1] / 2.0, e.diff.shape[0] / 2.0
        cands = detect_streaks(e.white, e.psf_sigma_px, tile=400, overlap=0,
                               min_length=6, threshold=5.0)
        # KV is nearest the stamp center (stamp centered on catalogued position)
        fits_out = []
        for c in cands:
            ref = refine_candidate(e.white, c["x"], c["y"], c["pa_rad"], e.psf_sigma_px)
            if ref["snr"] < 6.0:
                continue
            fit = fit_trail(e.diff, e.mask, e.wcs, ref["x"], ref["y"], e.psf_sigma_px,
                            e.magzp, theta0=ref["pa_rad"], h0=ref["L"] / 2.0)
            if fit is None or not (3.0 <= fit["trail_len"] <= 200.0):
                continue
            fit["mf_snr"] = ref["snr"]
            fit["dcen_px"] = np.hypot(fit["x"] - cx0, fit["y"] - cy0)
            fits_out.append(fit)
        near = [f for f in fits_out if f["dcen_px"] < 60]  # within ~112" of center
        if near:
            best = min(near, key=lambda f: f["dcen_px"])
            dlen = best["trail_len"] - t["traillen"]
            dpa = abs((best["trail_PA"] - t["celpa"]) % 180.0)
            dpa = min(dpa, 180.0 - dpa)
            results.append(dict(expkey=expkey, obscode=t["obscode"],
                                dcen_px=best["dcen_px"],
                                our_len=best["trail_len"], cat_len=t["traillen"],
                                dlen=dlen, our_pa=best["trail_PA"], cat_pa=t["celpa"],
                                dpa=dpa, snr=best["mf_snr"]))
            print(f"{expkey} [{t['obscode']}]: KV @ {best['dcen_px']:.0f}px from center | "
                  f"len {best['trail_len']:.1f} vs cat {t['traillen']:.1f}\" (d{dlen:+.1f}) | "
                  f"PA(mod180) d{dpa:.1f} deg | snr {best['mf_snr']:.0f}")
        else:
            print(f"{expkey} [{t['obscode']}]: KV NOT found near center "
                  f"({len(fits_out)} trails in stamp, nearest "
                  f"{min((f['dcen_px'] for f in fits_out), default=-1):.0f}px)")
    n_match = len(results)
    n_exp = len({os.path.basename(p).replace('_diff.fits', '')[:14] for p in diffs
                 if truth.get(os.path.basename(p).replace('_diff.fits', '')[:14])})
    if results:
        dl = np.array([r["dlen"] for r in results])
        dp = np.array([r["dpa"] for r in results])
        print(f"\n=== ATLAS 2024 KV comparison: {n_match}/{n_exp} exposures recovered ===")
        print(f"length:   median |dlen| {np.median(np.abs(dl)):.2f}\" "
              f"(catalog trails ~10\"; our mean {np.mean([r['our_len'] for r in results]):.1f}\" "
              f"vs cat {np.mean([r['cat_len'] for r in results]):.1f}\")")
        print(f"PA:       median |dPA| {np.median(dp):.2f} deg (mod 180)")
    json.dump(results, open(os.path.join(ATL, "comparison.json"), "w"), indent=1)
    print(f"wrote {os.path.join(ATL, 'comparison.json')}")


if __name__ == "__main__":
    main()
