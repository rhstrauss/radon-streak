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


def edge_fraction(e, f):
    """Fraction of the fitted trail axis running within 4 px of a masked pixel.
    Mask-edge/halo-boundary artifacts trace the mask -> high fraction; real
    streaks on clean background -> ~0."""
    th, h = f["theta_px"], max(f["h_px"], 6.0)
    ct, st = np.cos(th), np.sin(th)
    tt = np.linspace(-h, h, max(int(2 * h), 12))
    m = e.mask
    hh, ww = m.shape
    near = 0
    for t in tt:
        xc, yc = f["x"] + t * ct, f["y"] + t * st
        sl = m[max(0, int(yc - 4)):int(yc + 5), max(0, int(xc - 4)):int(xc + 5)]
        if sl.size and (sl > 0).any():
            near += 1
    return near / len(tt)


# flatten with exposure index; compute a cleanliness-aware quality rank
allf = []
for ei, flist in enumerate(fits_by_exp):
    for f in flist:
        f = dict(f); f["exp"] = ei
        f["edge_frac"] = edge_fraction(exps[ei], f)
        allf.append(f)

# quality gate: reject mask-edge artifacts (trail hugging a mask boundary) and
# the near-ceiling lengths that are satellites/masking rows rather than clean
# streaks; among the rest, rank by matched-filter SNR.
clean = [f for f in allf if f["edge_frac"] < 0.25 and f["trail_len"] < 250.0]
clean.sort(key=lambda f: -f["mf_snr"])
best = clean[:N]
for f in best:
    f["rate"] = f["trail_len"] / 30.0 * 86400.0 / 3600.0  # deg/day (len/exptime)
print(f"{len(allf)} detections; {len(clean)} pass the cleanliness gate "
      f"(edge_frac<0.25, len<250\"); showing top {min(N,len(best))} by SNR")

ncol = 5
nrow = int(np.ceil(N / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.4 * nrow),
                         facecolor="white")
axes = np.atleast_1d(axes).ravel()

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

for k in range(N, len(axes)):
    axes[k].axis("off")

fig.suptitle("Catalina G96 (2024-05-01, field N25057) — 10 best-quality streak "
             "detections\nstreak_radon FRT+Veres, ranked by matched-filter SNR; "
             "difference-image cutouts, fitted axis dashed", fontsize=12, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(ROOT, "work", "g96_streak_mosaic.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
# also dump the table
for k, f in enumerate(best):
    print(f"#{k+1} exp{f['exp']} ({f['x']:.0f},{f['y']:.0f}) SNR {f['mf_snr']:.1f} "
          f"len {f['trail_len']:.1f}\" PA {f['trail_PA']:.1f} mag {f['mag']:.2f} "
          f"chi2r {f['chi2r']:.2f}")
