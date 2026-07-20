#!/usr/bin/env python
"""Single image-only mosaic (no text, no overlays) of the best real streaks:
  - the 2 best Catalina G96 streaks,
  - a000001 in both ZTF exposures,
  - the known NEO 2024 KV on ATLAS (difference + reduced).
Plus a second version that also includes VALIDATED FAKES (synthetic trails
injected into the real G96 frames and recovered by the pipeline).

Each panel is a bare difference-image cutout centred on the streak, robustly
stretched; no titles, ticks, or captions.
"""
import glob
import json
import os
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pickle
from streakradon.adapters.atlas import prepare_stamp, _atlas_wcs   # noqa: E402
from streakradon.frt_driver import detect_streaks                 # noqa: E402
from streakradon.mf_snr import refine_candidate                   # noqa: E402
from streakradon.trail_fit import fit_trail                       # noqa: E402
from streakradon.inject import inject_trail                       # noqa: E402

ZDIR = "/astro/store/shire/rstrau/ari_a000001_check/reverify_2026-06/ztf_search"
ATL = os.path.join(ROOT, "work", "atlas")
A0 = {"exp1": (154.9703050, 27.6108747), "exp2": (154.8539060, 28.2171955)}


def open2d(p):
    for h in fits.open(p):
        if h.data is not None and getattr(h.data, "ndim", 0) == 2:
            return np.asarray(h.data, float), h.header
    raise ValueError(p)


def cutout(diff, x, y, h_px):
    hw = int(max(42, 1.25 * max(h_px, 8) + 22))
    xi, yi = int(round(x)), int(round(y))
    y0, y1 = max(0, yi - hw), min(diff.shape[0], yi + hw)
    x0, x1 = max(0, xi - hw), min(diff.shape[1], xi + hw)
    return diff[y0:y1, x0:x1]


def g96_panels():
    """The 2 genuine exp3 detections are the SAME physical streak (co-linear:
    PA agree to 0.4 deg, perpendicular offset 0.6", and the 273" detection's
    extent contains the 49" one). Merge them into the single JOINT trail (union
    of the two extents along the common axis) and return one cutout of the full
    streak."""
    exps = pickle.load(open(f"{ROOT}/work/g96_exps.pkl", "rb"))
    fbe = json.load(open(f"{ROOT}/work/g96_fits.json"))
    f1 = min(fbe[3], key=lambda g: np.hypot(g["x"] - 2448, g["y"] - 2085))
    f2 = min(fbe[3], key=lambda g: np.hypot(g["x"] - 2398, g["y"] - 2067))
    # common axis = the longer detection's; project both endpoints onto it
    th = f2["theta_px"]; ct, st = np.cos(th), np.sin(th)
    ox, oy = f2["x"], f2["y"]
    ends = []
    for f in (f1, f2):
        for sgn in (-1, 1):
            ex, ey = f["x"] + sgn * f["h_px"] * np.cos(f["theta_px"]), \
                     f["y"] + sgn * f["h_px"] * np.sin(f["theta_px"])
            ends.append((ex - ox) * ct + (ey - oy) * st)
    lo, hi = min(ends), max(ends)
    h_joint = (hi - lo) / 2.0
    cx, cy = ox + (lo + hi) / 2.0 * ct, oy + (lo + hi) / 2.0 * st
    return [cutout(exps[3].diff, cx, cy, h_joint)]


def ztf_panels():
    out = []
    for tag in ("exp1", "exp2"):
        diff, _ = open2d(f"{ZDIR}/ztf_{tag}_diffimg.fits.fz")
        ar, ad = A0[tag]
        fits_ = json.load(open(f"{ROOT}/work/ztf_{tag}_fits.json"))
        # a000001 = detection nearest truth
        f = min(fits_, key=lambda g: np.hypot((g["ra"] - ar) * np.cos(np.radians(ad)),
                                              g["dec"] - ad))
        out.append(cutout(diff, f["x"], f["y"], f["h_px"]))
    return out


def atlas_panels():
    # the clean KV recovery (blind detect on the diff stamps; keep the catalog-
    # consistent one), shown in both diff and reduced images
    truth = {}
    for ln in open(f"{ATL}/kv_matches.csv").read().strip().split("\n")[1:]:
        c = ln.split(","); truth[c[10][:14]] = dict(tl=float(c[4]), pa=float(c[5]))
    seen, best = set(), None
    for path in sorted(glob.glob(f"{ATL}/stamps/*/diff/*_diff.fits")):
        key = os.path.basename(path).replace("_diff.fits", "")[:14]
        if key in seen or key not in truth:
            continue
        seen.add(key)
        e = prepare_stamp(path); cx, cy = e.diff.shape[1] / 2, e.diff.shape[0] / 2
        cands = detect_streaks(e.white, e.psf_sigma_px, tile=e.diff.shape[0],
                               overlap=0, min_length=6, threshold=5.0)
        for c in cands:
            ref = refine_candidate(e.white, c["x"], c["y"], c["pa_rad"],
                                   e.psf_sigma_px, lengths=(4, 6, 8, 12, 16, 24, 32))
            if ref["snr"] < 8:
                continue
            fit = fit_trail(e.diff, e.mask, e.wcs, ref["x"], ref["y"], e.psf_sigma_px,
                            e.magzp, theta0=ref["pa_rad"], h0=ref["L"] / 2.0)
            if fit is None:
                continue
            dpa = abs((fit["trail_PA"] - truth[key]["pa"]) % 180); dpa = min(dpa, 180 - dpa)
            d = np.hypot(fit["x"] - cx, fit["y"] - cy)
            if (0.4 < fit["trail_len"] / truth[key]["tl"] < 2.5 and dpa < 25 and d < 45
                    and (best is None or ref["snr"] > best[0])):
                best = (ref["snr"], e, fit, path)
    out = []
    if best:
        _, e, fit, path = best
        out.append(cutout(e.diff, fit["x"], fit["y"], fit["h_px"]))
        rpath = path.replace("/diff/", "/reduced/").replace("_diff.fits", ".fits")
        if os.path.exists(rpath):
            red, _ = open2d(rpath)
            red = red - np.median(red[np.isfinite(red)])
            out.append(cutout(red, fit["x"], fit["y"], fit["h_px"]))
    return out


def fake_panels(n=4):
    """Inject bright synthetic trails into the real G96 frame 0, re-difference,
    detect+fit, and keep the ones cleanly recovered (validated fakes)."""
    sys.path.insert(0, os.path.join(ROOT, "bin"))
    from measure_efficiency import build_exp0
    import yaml
    cfg = yaml.safe_load(open(f"{ROOT}/config/g96.yaml"))
    stack = pickle.load(open(f"{ROOT}/work/g96_stack.pkl", "rb"))
    magzp = float(stack["headers"][0].get("MAGZP", 28.0)); psf = stack["psf_sigma"]
    rng = np.random.default_rng(3)
    specs = [(17.0, 55, 40.0), (17.5, 40, 70.0), (18.0, 70, 30.0), (17.0, 90, 120.0),
             (18.0, 30, 55.0), (17.5, 25, 95.0)]
    frame0 = stack["reg"][0].copy(); truths = []
    for mag, Lpx, pa_deg in specs:
        x = rng.uniform(1200, 4000); y = rng.uniform(1200, 4000)
        inject_trail(frame0, x, y, np.radians(pa_deg), Lpx, mag, magzp, psf)
        truths.append((x, y, np.radians(pa_deg), Lpx))
    st = dict(stack); st["reg"] = stack["reg"].copy(); st["reg"][0] = frame0
    e = build_exp0(st, cfg)
    out = []
    for (x, y, pa, Lpx) in truths:
        HW = 130
        win = e.white[int(y) - HW:int(y) + HW, int(x) - HW:int(x) + HW]
        if win.shape != (2 * HW, 2 * HW):
            continue
        cands = detect_streaks(win, e.psf_sigma_px, tile=2 * HW, overlap=0,
                               min_length=8, threshold=5.0)
        rec = None
        for c in cands:
            gx, gy = c["x"] + int(x) - HW, c["y"] + int(y) - HW
            if np.hypot(gx - x, gy - y) > 22:  # FRT centre imprecision; fit polishes it
                continue
            ref = refine_candidate(e.white, gx, gy, c["pa_rad"], e.psf_sigma_px)
            if ref["snr"] < 11:
                continue
            fit = fit_trail(e.diff, e.mask, e.wcs, ref["x"], ref["y"], e.psf_sigma_px,
                            e.magzp, theta0=ref["pa_rad"], h0=ref["L"] / 2.0)
            if fit is None:
                continue
            # validated = recovered within a few px of the injection
            if np.hypot(fit["x"] - x, fit["y"] - y) < 6:
                rec = fit; break
        if rec is not None:
            out.append(cutout(e.diff, rec["x"], rec["y"], rec["h_px"]))
        if len(out) >= n:
            break
    return out


def render(panels, out_path, ncol=None):
    M = len(panels)
    if ncol is None:
        ncol = min(M, 4) if M > 4 else M
    nrow = int(np.ceil(M / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.4 * nrow),
                             facecolor="black")
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for k, sub in enumerate(panels):
        ax = axes[k]
        g = np.isfinite(sub)
        m = np.median(sub[g]); s = 1.4826 * np.median(np.abs(sub[g] - m))
        ax.imshow(np.where(g, sub, m), vmin=m - 2 * s, vmax=m + 8 * s,
                  cmap="gray_r", origin="lower", interpolation="nearest",
                  aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,
                        wspace=0.04, hspace=0.04)
    fig.savefig(out_path, dpi=140, facecolor="black")
    print("wrote", out_path, f"({M} panels)")


def main():
    real = g96_panels() + ztf_panels() + atlas_panels()
    render(real, f"{ROOT}/work/mosaic_real.png", ncol=len(real))  # single row
    fakes = fake_panels(4)
    print(f"validated fakes recovered: {len(fakes)}")
    tot = real + fakes
    render(tot, f"{ROOT}/work/mosaic_with_fakes.png",
           ncol=int(np.ceil(len(tot) / 2)))  # two rows


if __name__ == "__main__":
    main()
