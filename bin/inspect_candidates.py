#!/usr/bin/env python
"""Diagnostic: render isolated, non-dipole streak candidates (relaxed spread/
cover) so genuine linear streaks can be picked visually. Writes work/g96_candidates.png."""
import json, os, pickle, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); import streakradon  # noqa

exps = pickle.load(open(f"{ROOT}/work/g96_exps.pkl", "rb"))
fbe = json.load(open(f"{ROOT}/work/g96_fits.json"))


def metrics(e, f):
    th, h = f["theta_px"], max(f["h_px"], 6.0); ct, st = np.cos(th), np.sin(th)
    d, m = e.diff, e.mask; hh, ww = d.shape
    xi, yi = int(round(f["x"])), int(round(f["y"])); hw = int(h + 35)
    msub = m[max(0, yi-hw):yi+hw, max(0, xi-hw):xi+hw]
    nbr = float((msub > 0).mean()) if msub.size else 1.0
    tt = np.linspace(-h, h, max(int(2*h), 16)); prof = []
    for t in tt:
        s = 0.0; n = 0
        for w in range(-2, 3):
            xc = f["x"]+t*ct-w*st; yc = f["y"]+t*st+w*ct; iy, ix = int(round(yc)), int(round(xc))
            if 0 <= iy < hh and 0 <= ix < ww and np.isfinite(d[iy, ix]) and m[iy, ix] == 0:
                s += d[iy, ix]; n += 1
        prof.append(s/n if n else 0.0)
    prof = np.array(prof); p = np.clip(prof, 0, None)
    if p.sum() <= 0: return nbr, 0, 0, 0
    wsum = p.sum(); tc = (tt*p).sum()/wsum
    spread = np.sqrt(((tt-tc)**2*p).sum()/wsum)/(h/np.sqrt(3))
    cover = float((p > 0.5*p.max()).mean()); neg = float(prof.min()/(p.max()+1e-9))
    return nbr, spread, cover, neg


rows = []
for ei, fl in enumerate(fbe):
    for f in fl:
        ff = dict(f); ff["exp"] = ei
        ff["nbr"], ff["spread"], ff["cover"], ff["neg"] = metrics(exps[ei], ff)
        rows.append(ff)
# isolated + not a strong dipole -> visual-inspection set
cand = [r for r in rows if r["nbr"] < 0.10 and r["neg"] > -1.0]
cand.sort(key=lambda r: -r["mf_snr"])
cand = cand[:12]

ncol = 4; nrow = 3
fig, axes = plt.subplots(nrow, ncol, figsize=(3*ncol, 3.5*nrow), facecolor="white")
axes = axes.ravel()
for k, f in enumerate(cand):
    ax = axes[k]; e = exps[f["exp"]]; d = e.diff
    h = max(f["h_px"], 8.0); hw = int(max(38, 1.7*h+16))
    xi, yi = int(round(f["x"])), int(round(f["y"]))
    y0, y1 = max(0, yi-hw), min(d.shape[0], yi+hw); x0, x1 = max(0, xi-hw), min(d.shape[1], xi+hw)
    sub = d[y0:y1, x0:x1]; g = np.isfinite(sub)
    m = np.median(sub[g]); s = 1.4826*np.median(np.abs(sub[g]-m))
    ax.imshow(np.where(g, sub, m), vmin=m-2*s, vmax=m+8*s, cmap="gray_r",
              origin="lower", extent=[x0, x1, y0, y1], interpolation="nearest")
    th = f["theta_px"]
    ax.plot([f["x"]-h*np.cos(th), f["x"]+h*np.cos(th)],
            [f["y"]-h*np.sin(th), f["y"]+h*np.sin(th)], color="#e8552d", lw=1, ls="--")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{k+1}: SNR{f['mf_snr']:.0f} len{f['trail_len']:.0f}\"", fontsize=9)
    ax.text(0.5, -0.02, f"nbr{100*f['nbr']:.0f}% sprd{f['spread']:.2f} "
            f"cov{f['cover']:.2f} neg{f['neg']:+.1f}", transform=ax.transAxes,
            ha="center", va="top", fontsize=8)
for k in range(len(cand), len(axes)): axes[k].axis("off")
fig.tight_layout()
fig.savefig(f"{ROOT}/work/g96_candidates.png", dpi=120, bbox_inches="tight")
print("wrote work/g96_candidates.png")
