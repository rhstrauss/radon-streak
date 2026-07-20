#!/usr/bin/env python
"""Mosaic of the N best-quality G96 streak detections (ranked by integrated
matched-filter SNR). Each panel is a difference-image cutout oriented as
detected, with the fitted trail axis marked and the measured properties
labelled."""
import json
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # for the DiffExposure class pickled in g96_exps.pkl
import streakradon  # noqa: F401,E402
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10

exps = pickle.load(open(os.path.join(ROOT, "work", "g96_exps.pkl"), "rb"))
fits_by_exp = json.load(open(os.path.join(ROOT, "work", "g96_fits.json")))

pixscale = 1.52


def streak_metrics(e, f):
    """Physical quality metrics separating real linear streaks from stellar /
    mask artifacts, measured on the difference image along the fitted axis:
      nbr    : masked fraction in a box around the streak (isolation from
               bright-star halos / bleed columns / frame masks)
      spread : RMS of flux-weighted position along axis / uniform-line RMS
               (~1 for a real extended streak, ~0 for a compact stellar residual)
      cover  : fraction of along-axis bins carrying real flux (real streak ~1;
               a spurious line fit over empty background -> 0)
      neg    : min flux along axis / peak (a stellar dipole has a strong
               over-subtracted negative lobe -> large negative)
    """
    th, h = f["theta_px"], max(f["h_px"], 6.0)
    ct, st = np.cos(th), np.sin(th)
    d, m = e.diff, e.mask
    hh, ww = d.shape
    xi, yi = int(round(f["x"])), int(round(f["y"]))
    hw = int(h + 35)
    msub = m[max(0, yi - hw):yi + hw, max(0, xi - hw):xi + hw]
    nbr = float((msub > 0).mean()) if msub.size else 1.0
    tt = np.linspace(-h, h, max(int(2 * h), 16))
    prof = []
    for t in tt:
        s, n = 0.0, 0
        for w in range(-2, 3):
            xc, yc = f["x"] + t * ct - w * st, f["y"] + t * st + w * ct
            iy, ix = int(round(yc)), int(round(xc))
            if 0 <= iy < hh and 0 <= ix < ww and np.isfinite(d[iy, ix]) and m[iy, ix] == 0:
                s += d[iy, ix]; n += 1
        prof.append(s / n if n else 0.0)
    prof = np.array(prof); p = np.clip(prof, 0, None)
    if p.sum() <= 0:
        return nbr, 0.0, 0.0, 0.0
    wsum = p.sum(); tc = (tt * p).sum() / wsum
    spread = np.sqrt(((tt - tc) ** 2 * p).sum() / wsum) / (h / np.sqrt(3))
    cover = float((p > 0.5 * p.max()).mean())
    neg = float(prof.min() / (p.max() + 1e-9))
    return nbr, spread, cover, neg


allf = []
for ei, flist in enumerate(fits_by_exp):
    for f in flist:
        f = dict(f); f["exp"] = ei
        f["nbr"], f["spread"], f["cover"], f["neg"] = streak_metrics(exps[ei], f)
        allf.append(f)

# keep only genuine linear streaks: isolated from masks, flux spread along the
# whole trail, real flux coverage, and no stellar-dipole negative lobe.
clean = [f for f in allf if f["nbr"] < 0.09 and f["spread"] > 0.45
         and f["cover"] > 0.6 and f["neg"] > -0.7]
clean.sort(key=lambda f: -f["mf_snr"])
best = clean[:N]
for f in best:
    f["rate"] = f["trail_len"] / 30.0 * 86400.0 / 3600.0  # deg/day (len/exptime)
print(f"{len(allf)} detections; {len(clean)} are genuine linear streaks "
      f"(isolated + flux-spread + non-dipole); showing {len(best)}")

M = len(best)
ncol = min(M, 3) if M else 1
nrow = int(np.ceil(M / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 4.2 * nrow),
                         facecolor="white")
axes = np.atleast_1d(axes).ravel()
fig.subplots_adjust(hspace=0.55, wspace=0.15)

for k, f in enumerate(best):
    ax = axes[k]
    e = exps[f["exp"]]
    diff = e.diff
    x, y = f["x"], f["y"]
    h = max(f["h_px"], 8.0)
    hw = int(max(38, 1.7 * h + 16))
    xi, yi = int(round(x)), int(round(y))
    y0, y1 = max(0, yi - hw), min(diff.shape[0], yi + hw)
    x0, x1 = max(0, xi - hw), min(diff.shape[1], xi + hw)
    sub = diff[y0:y1, x0:x1]
    good = np.isfinite(sub)
    if good.sum() > 20:
        m = np.median(sub[good]); s = 1.4826 * np.median(np.abs(sub[good] - m))
        vmin, vmax = m - 2 * s, m + 8 * s
    else:
        vmin, vmax = np.nanmin(sub), np.nanmax(sub)
    ax.imshow(np.where(good, sub, m if good.sum() else 0),
              vmin=vmin, vmax=vmax, cmap="gray_r", origin="lower",
              extent=[x0, x1, y0, y1], interpolation="nearest")
    # mark the fitted trail axis (dashed) via endpoints
    th = f["theta_px"]
    ax.plot([x - h * np.cos(th), x + h * np.cos(th)],
            [y - h * np.sin(th), y + h * np.sin(th)],
            color="#e8552d", lw=1.1, ls="--", alpha=0.9)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"#{k+1}  SNR {f['mf_snr']:.0f}   exp{f['exp']}", fontsize=10,
                 pad=3)
    ax.text(0.5, -0.02,
            f"len {f['trail_len']:.0f}\"  PA {f['trail_PA']:.0f}°  {f['rate']:.0f}°/day\n"
            f"mag {f['mag']:.1f}  χ²ᵣ {f['chi2r']:.2f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5)

for k in range(M, len(axes)):
    axes[k].axis("off")

fig.suptitle(f"Catalina G96 (2024-05-01, field N25057) — {M} genuine linear streaks "
             "(of 70 raw detections)\nstreak_radon FRT+Veres; stellar residuals & "
             "mask/bleed-edge artifacts removed; difference-image cutouts, fitted "
             "axis dashed", fontsize=12, y=0.99)
out = os.path.join(ROOT, "work", "g96_streak_mosaic.png")
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.3)
print("wrote", out)
# also dump the table
for k, f in enumerate(best):
    print(f"#{k+1} exp{f['exp']} ({f['x']:.0f},{f['y']:.0f}) SNR {f['mf_snr']:.1f} "
          f"len {f['trail_len']:.1f}\" PA {f['trail_PA']:.1f} mag {f['mag']:.2f} "
          f"chi2r {f['chi2r']:.2f}")
