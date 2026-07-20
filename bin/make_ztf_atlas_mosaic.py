#!/usr/bin/env python
"""Mosaic of the best ZTF and ATLAS trails from this run (EXCLUDING a000001).

ZTF: the non-a000001 detections from the a000001 full-quad diffs, passed through
the same genuine-linear-streak quality gate used for G96.
ATLAS: the recovered 2024 KV trails on real ATLAS difference stamps (KV is the
trail nearest each stamp centre; the stamp WCS is unusable for absolute
astrometry, so KV is located by centrality).

Clean difference-image cutouts; trail ends bracketed with caliper ticks (the
streak itself is never overdrawn).
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
from streakradon.adapters.atlas import prepare_stamp        # noqa: E402
from streakradon.frt_driver import detect_streaks           # noqa: E402
from streakradon.mf_snr import refine_candidate             # noqa: E402
from streakradon.trail_fit import fit_trail                 # noqa: E402

ZDIR = "/astro/store/shire/rstrau/ari_a000001_check/reverify_2026-06/ztf_search"
ATL = os.path.join(ROOT, "work", "atlas")
A0 = {"exp1": (154.9703050, 27.6108747), "exp2": (154.8539060, 28.2171955)}


def open2d(p):
    for h in fits.open(p):
        if h.data is not None and getattr(h.data, "ndim", 0) == 2:
            return np.asarray(h.data, float), h.header
    raise ValueError(p)


def metrics(diff, mask, f):
    th, h = f["theta_px"], max(f["h_px"], 6.0)
    ct, st = np.cos(th), np.sin(th)
    hh, ww = diff.shape
    xi, yi = int(round(f["x"])), int(round(f["y"]))
    hw = int(h + 35)
    msub = mask[max(0, yi-hw):yi+hw, max(0, xi-hw):xi+hw]
    nbr = float((msub > 0).mean()) if msub.size else 1.0
    tt = np.linspace(-h, h, max(int(2*h), 16)); prof = []
    for t in tt:
        s, n = 0.0, 0
        for w in range(-2, 3):
            xc, yc = f["x"]+t*ct-w*st, f["y"]+t*st+w*ct
            iy, ix = int(round(yc)), int(round(xc))
            if 0 <= iy < hh and 0 <= ix < ww and np.isfinite(diff[iy, ix]) and mask[iy, ix] == 0:
                s += diff[iy, ix]; n += 1
        prof.append(s/n if n else 0.0)
    prof = np.array(prof); p = np.clip(prof, 0, None)
    if p.sum() <= 0:
        return nbr, 0.0, 0.0, 0.0
    wsum = p.sum(); tc = (tt*p).sum()/wsum
    spread = np.sqrt(((tt-tc)**2*p).sum()/wsum)/(h/np.sqrt(3))
    cover = float((p > 0.5*p.max()).mean()); neg = float(prof.min()/(p.max()+1e-9))
    return nbr, spread, cover, neg


def gather_ztf():
    out = []
    for tag in ("exp1", "exp2"):
        diff, hdr = open2d(f"{ZDIR}/ztf_{tag}_diffimg.fits.fz")
        mask, _ = open2d(f"{ZDIR}/ztf_{tag}_mskimg.fits")
        m = (mask != 0).astype(np.uint8)
        wcs = WCS(hdr)
        ar, ad = A0[tag]
        for f in json.load(open(f"{ROOT}/work/ztf_{tag}_fits.json")):
            sep = np.hypot((f["ra"]-ar)*np.cos(np.radians(ad)), f["dec"]-ad)*3600
            if sep < 8:
                continue  # a000001 -- excluded
            nbr, spread, cover, neg = metrics(diff, m, f)
            if nbr < 0.12 and spread > 0.45 and cover > 0.6 and neg > -0.7:
                f = dict(f); f["survey"] = f"ZTF {tag}"; f["diff"] = diff; f["mask"] = m
                out.append(f)
    out.sort(key=lambda f: -f["mf_snr"])
    return out


def gather_atlas():
    truth = {}
    for ln in open(f"{ATL}/kv_matches.csv").read().strip().split("\n")[1:]:
        c = ln.split(","); truth[c[10][:14]] = dict(
            traillen=float(c[4]), celpa=float(c[5]), obscode=c[12], mjd=float(c[0]))
    out = []
    seen = set()
    for path in sorted(glob.glob(f"{ATL}/stamps/*/diff/*_diff.fits")):
        key = os.path.basename(path).replace("_diff.fits", "")[:14]
        if key in seen or key not in truth:
            continue
        seen.add(key)
        e = prepare_stamp(path)
        cx0, cy0 = e.diff.shape[1]/2.0, e.diff.shape[0]/2.0
        cands = detect_streaks(e.white, e.psf_sigma_px, tile=e.diff.shape[0],
                               overlap=0, min_length=6, threshold=5.0)
        catlen = truth[key]["traillen"]
        best = None
        for c in cands:
            # ATLAS NEO trails are short (~5 px at 30 s); cap the length search to
            # the physical scale so the fit doesn't over-integrate onto longer
            # residuals in the artifact-heavy 30 s diff.
            ref = refine_candidate(e.white, c["x"], c["y"], c["pa_rad"],
                                   e.psf_sigma_px, lengths=(4, 6, 8, 12, 16, 24, 32))
            if ref["snr"] < 6:
                continue
            fit = fit_trail(e.diff, e.mask, e.wcs, ref["x"], ref["y"], e.psf_sigma_px,
                            e.magzp, theta0=ref["pa_rad"], h0=ref["L"]/2.0)
            if fit is None or not (3 <= fit["trail_len"] <= 40):
                continue
            fit["mf_snr"] = ref["snr"]
            fit["dcen"] = np.hypot(fit["x"]-cx0, fit["y"]-cy0)
            # length- AND PA-consistent with the catalogued KV trail (real KV,
            # not a latch-on onto another residual near the stamp centre)
            dpa = abs((fit["trail_PA"] - truth[key]["celpa"]) % 180.0)
            dpa = min(dpa, 180.0 - dpa)
            if not (0.4 < fit["trail_len"] / catlen < 2.5) or dpa > 25.0:
                continue
            if fit["dcen"] < 45 and (best is None or fit["dcen"] < best["dcen"]):
                best = fit
        if best:
            best["survey"] = f"ATLAS {truth[key]['obscode']}"
            best["diff"] = e.diff; best["mask"] = e.mask
            best["cat_len"] = truth[key]["traillen"]; best["cat_pa"] = truth[key]["celpa"]
            best["path"] = path
            out.append(best)
    out.sort(key=lambda f: -f["mf_snr"])
    return out


def panel(ax, f):
    diff = f["diff"]; x, y = f["x"], f["y"]
    h = max(f["h_px"], 8.0); hw = int(max(40, 1.7*h+18))
    xi, yi = int(round(x)), int(round(y))
    y0, y1 = max(0, yi-hw), min(diff.shape[0], yi+hw)
    x0, x1 = max(0, xi-hw), min(diff.shape[1], xi+hw)
    sub = diff[y0:y1, x0:x1]; g = np.isfinite(sub)
    mm = np.median(sub[g]); s = 1.4826*np.median(np.abs(sub[g]-mm))
    ax.imshow(np.where(g, sub, mm), vmin=mm-2*s, vmax=mm+8*s, cmap="gray_r",
              origin="lower", extent=[x0, x1, y0, y1], interpolation="nearest")
    th = f["theta_px"]; ct, st = np.cos(th), np.sin(th)
    for sgn in (-1, 1):
        ex, ey = x+sgn*(h+7)*ct, y+sgn*(h+7)*st
        ax.plot([ex-8*(-st), ex+8*(-st)], [ey-8*ct, ey+8*ct], color="#e8552d", lw=1.4)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_xticks([]); ax.set_yticks([])


def reduced_panel(diff_fit):
    """For a clean KV diff detection, fit the same trail on the REDUCED (non-
    differenced) ATLAS stamp at the same pixel position, so KV can be shown in
    both image products."""
    from streakradon.adapters.atlas import _atlas_wcs
    rpath = diff_fit["path"].replace("/diff/", "/reduced/").replace("_diff.fits", ".fits")
    if not os.path.exists(rpath):
        return None
    red, hdr = open2d(rpath)
    wcs = _atlas_wcs(hdr)
    fwhm = hdr.get("FWHM", hdr.get("SEEING", 4.0)) or 4.0
    sig = float(fwhm) / 2.355
    # background-subtract locally so the trail (a positive residual over sky) shows
    med = np.median(red[np.isfinite(red)])
    d = red - med
    mask = (~np.isfinite(red)).astype(np.uint8)
    fit = fit_trail(d, mask, wcs, diff_fit["x"], diff_fit["y"], sig,
                    hdr.get("MAGZP", 22.0), theta0=diff_fit["theta_px"],
                    h0=max(diff_fit["h_px"], 4))
    if fit is None:
        return None
    fit["survey"] = diff_fit["survey"] + " (reduced)"
    fit["mf_snr"] = diff_fit["mf_snr"]; fit["diff"] = d; fit["mask"] = mask
    fit["cat_len"] = diff_fit["cat_len"]; fit["cat_pa"] = diff_fit["cat_pa"]
    return fit


def main():
    ztf = gather_ztf()
    atl = gather_atlas()
    print(f"ZTF genuine non-a000001 trails: {len(ztf)}")
    print(f"ATLAS 2024 KV trails recovered: {len(atl)}")
    # show each clean KV recovery in BOTH the difference and reduced image
    atl_panels = []
    for f in atl[:4]:
        atl_panels.append(f)
        r = reduced_panel(f)
        if r is not None:
            atl_panels.append(r)
    picks = [("ZTF", f) for f in ztf[:4]] + [("ATLAS", f) for f in atl_panels]
    if not picks:
        print("nothing to show"); return
    M = len(picks); ncol = min(5, M); nrow = int(np.ceil(M/ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1*ncol, 3.7*nrow), facecolor="white")
    axes = np.atleast_1d(axes).ravel()
    fig.subplots_adjust(hspace=0.5, wspace=0.15)
    for k, (src, f) in enumerate(picks):
        ax = axes[k]; panel(ax, f)
        ax.set_title(f"{f['survey']}   SNR {f['mf_snr']:.0f}", fontsize=10, pad=3)
        if src == "ATLAS":
            lab = (f"len {f['trail_len']:.1f}\" (cat {f['cat_len']:.1f})\n"
                   f"PA {f['trail_PA']:.0f}° (cat {f['cat_pa']:.0f})  mag {f['mag']:.1f}")
        else:
            lab = (f"len {f['trail_len']:.0f}\"  PA {f['trail_PA']:.0f}°\n"
                   f"mag {f['mag']:.1f}  χ²ᵣ {f['chi2r']:.2f}")
        ax.text(0.5, -0.02, lab, transform=ax.transAxes, ha="center", va="top", fontsize=8.5)
    for k in range(M, len(axes)):
        axes[k].axis("off")
    fig.suptitle("Best ZTF & ATLAS trails from this run (excluding a000001)\n"
                 "streak_radon on real ZTF and ATLAS difference images; "
                 "ATLAS panels are the known NEO 2024 KV vs its trailmain catalog value",
                 fontsize=12, y=0.99)
    out = os.path.join(ROOT, "work", "ztf_atlas_mosaic.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.3)
    print("wrote", out)
    for src, f in picks:
        extra = f" cat_len {f['cat_len']:.1f} cat_pa {f['cat_pa']:.0f}" if src == "ATLAS" else ""
        print(f"{f['survey']} SNR {f['mf_snr']:.1f} len {f['trail_len']:.1f}\" "
              f"PA {f['trail_PA']:.1f} mag {f['mag']:.2f}{extra}")


if __name__ == "__main__":
    main()
