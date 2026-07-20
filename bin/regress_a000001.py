#!/usr/bin/env python
"""P1 REGRESSION GATE — the pipeline must blindly rediscover a000001 in both
ZTF exposures (ground truth from the validated 2026-06 reverify fit).

Chain under test: whiten -> FRT (min_length=8) -> dedup -> refine_length ->
mf_snr -> Veres fit. Runs on a 512x512 cutout around the truth position (the
2026-07 benchmark protocol); full-quad blind runs happen in P6.

Pass criteria (per exposure):
  * FRT produces a candidate whose line passes within 15 px of the truth center
  * our integrated MF SNR at the refined candidate >= 15 (exp1) / >= 10 (exp2)
  * Veres fit: center within 3" of truth, trail_len within +-10" of truth,
    PA within 5 deg (mod 180)

Exit 0 = PASS (both exposures), 1 = FAIL.
"""
import os
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from streakradon.frt_driver import detect_streaks            # noqa: E402
from streakradon.mf_snr import refine_candidate              # noqa: E402
from streakradon.trail_fit import fit_trail                  # noqa: E402
from streakradon.varmap import whiten                        # noqa: E402

DATA = "/astro/store/shire/rstrau/ari_a000001_check/reverify_2026-06/ztf_search"
TRUTH = {
    "exp1": dict(ra=154.9703050, dec=27.6108747, L=66.759, pa=350.126, snr_min=15.0),
    "exp2": dict(ra=154.8539060, dec=28.2171955, L=71.477, pa=350.596, snr_min=10.0),
}
CUT = 256  # half-width of the test cutout


def open2d(path):
    for h in fits.open(path):
        if h.data is not None and getattr(h.data, "ndim", 0) == 2:
            return np.asarray(h.data, float), h.header
    raise ValueError(path)


def run_exposure(tag):
    t = TRUTH[tag]
    diff, hdr = open2d(f"{DATA}/ztf_{tag}_diffimg.fits.fz")
    mask, _ = open2d(f"{DATA}/ztf_{tag}_mskimg.fits")
    wcs = WCS(hdr)
    magzp = hdr.get("MAGZP", 26.0)
    seeing_px = hdr.get("SEEING", 2.0)  # ZTF header SEEING is in px (FWHM)
    sigma = seeing_px / 2.355
    pixscale = np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600.0

    tx, ty = wcs.world_to_pixel_values(t["ra"], t["dec"])
    tx, ty = float(tx), float(ty)
    x0 = int(np.clip(tx - CUT, 0, diff.shape[1] - 2 * CUT))
    y0 = int(np.clip(ty - CUT, 0, diff.shape[0] - 2 * CUT))
    sub = diff[y0:y0 + 2 * CUT, x0:x0 + 2 * CUT]
    smask = mask[y0:y0 + 2 * CUT, x0:x0 + 2 * CUT]

    white, var, bg = whiten(sub, (smask != 0).astype(np.uint8), grid=64)
    cands = detect_streaks(white, sigma, tile=512, overlap=0,
                           min_length=8, threshold=5.0)
    print(f"[{tag}] {len(cands)} FRT candidates in {2*CUT}x{2*CUT} cutout")

    # nearest candidate line to the truth center (cutout coords)
    ttx, tty = tx - x0, ty - y0
    best, bestd = None, 1e9
    for c in cands:
        dx, dy = np.cos(c["pa_rad"]), np.sin(c["pa_rad"])
        px, py = ttx - c["x"], tty - c["y"]
        perp = abs(-dy * px + dx * py)
        along = abs(dx * px + dy * py) - 0.5 * c["L_frt"]
        d = perp + max(along, 0)
        if d < bestd:
            best, bestd = c, d
    if best is None or bestd > 15.0:
        print(f"[{tag}] FAIL: no FRT candidate near truth (best dist {bestd:.1f} px)")
        return False
    print(f"[{tag}] candidate @ ({best['x']:.1f},{best['y']:.1f}) dist {bestd:.1f} px, "
          f"L_frt {best['L_frt']:.0f} px, snr_frt {best['snr_frt']:.1f}, "
          f"pa {np.degrees(best['pa_rad']):.1f} deg")

    ref = refine_candidate(white, best["x"], best["y"], best["pa_rad"], sigma)
    L_est, cx, cy, snr = ref["L"], ref["x"], ref["y"], ref["snr"]
    print(f"[{tag}] refined L {L_est:.0f} px @ ({cx:.1f},{cy:.1f}) "
          f"pa {np.degrees(ref['pa_rad']):.1f}; MF SNR {snr:.1f} (gate >= {t['snr_min']})")
    if snr < t["snr_min"]:
        print(f"[{tag}] FAIL: MF SNR below gate")
        return False

    # Veres fit on the RAW diff cutout (unwhitened), global-frame WCS
    class SubWCS:
        """Shift-aware wrapper: fit in cutout coords, world ops via parent WCS."""
        def __init__(self, wcs, x0, y0):
            self.w, self.x0, self.y0 = wcs, x0, y0
            self.pixel_scale_matrix = wcs.pixel_scale_matrix

        def pixel_to_world(self, x, y):
            return self.w.pixel_to_world(x + self.x0, y + self.y0)

    fit = fit_trail(sub, smask, SubWCS(wcs, x0, y0), cx, cy, sigma, magzp,
                    theta0=ref["pa_rad"], h0=L_est / 2.0)
    if fit is None:
        print(f"[{tag}] FAIL: Veres fit failed")
        return False
    dra = (fit["ra"] - t["ra"]) * np.cos(np.radians(t["dec"])) * 3600.0
    ddec = (fit["dec"] - t["dec"]) * 3600.0
    dcen = np.hypot(dra, ddec)
    dL = fit["trail_len"] - t["L"]
    dpa = abs((fit["trail_PA"] - t["pa"]) % 180.0)
    dpa = min(dpa, 180.0 - dpa)
    print(f"[{tag}] fit: center off {dcen:.2f}\" (gate 3\"), len {fit['trail_len']:.1f}\" "
          f"(truth {t['L']:.1f}, dL {dL:+.1f}\", gate +-10\"), dPA {dpa:.2f} deg (gate 5), "
          f"mag {fit['mag']:.2f}, chi2r {fit['chi2r']:.2f}")
    ok = dcen <= 3.0 and abs(dL) <= 10.0 and dpa <= 5.0
    print(f"[{tag}] {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    results = {tag: run_exposure(tag) for tag in ("exp1", "exp2")}
    print("\n=== REGRESSION GATE:", "PASS" if all(results.values()) else "FAIL",
          results, "===")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
