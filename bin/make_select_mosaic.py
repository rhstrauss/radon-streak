#!/usr/bin/env python
"""4-panel image-only mosaic: the two a000001 ZTF trails, the G96 satellite
(joint Catalina trail), and one EMPIRICALLY-injected validated fake."""
import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bin"))
import streakradon  # noqa
from make_final_mosaic import cutout, ztf_panels, g96_panels   # reuse builders
from streakradon.psf import measure_psf_stamp                  # noqa
from streakradon.inject import inject_trail_empirical          # noqa
from streakradon.frt_driver import detect_streaks              # noqa
from streakradon.mf_snr import refine_candidate                # noqa
from streakradon.trail_fit import fit_trail                    # noqa
from measure_efficiency import build_exp0                      # noqa
import yaml


def empirical_fake_panel():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "g96.yaml")))
    stack = pickle.load(open(os.path.join(ROOT, "work", "g96_stack.pkl"), "rb"))
    ker = measure_psf_stamp(stack["reg"][0], stack["regm"][0])
    magzp = float(stack["headers"][0].get("MAGZP", 28.0))
    rng = np.random.default_rng(7)
    for _ in range(12):
        x, y = rng.uniform(1300, 3900), rng.uniform(1300, 3900)
        if stack["regm"][0][int(y), int(x)] != 0:
            continue
        pa, L, mag = rng.uniform(0, np.pi), 55.0, 17.5
        frame0 = stack["reg"][0].copy()
        inject_trail_empirical(frame0, x, y, pa, L, mag, magzp, ker)
        st = dict(stack); st["reg"] = stack["reg"].copy(); st["reg"][0] = frame0
        e = build_exp0(st, cfg)
        HW = 140
        win = e.white[int(y) - HW:int(y) + HW, int(x) - HW:int(x) + HW]
        cands = detect_streaks(win, e.psf_sigma_px, tile=2 * HW, overlap=0,
                               min_length=8, threshold=5.0)
        for c in cands:
            gx, gy = c["x"] + int(x) - HW, c["y"] + int(y) - HW
            if np.hypot(gx - x, gy - y) > 22:
                continue
            ref = refine_candidate(e.white, gx, gy, c["pa_rad"], e.psf_sigma_px)
            if ref["snr"] < 11:
                continue
            fit = fit_trail(e.diff, e.mask, e.wcs, ref["x"], ref["y"], e.psf_sigma_px,
                            e.magzp, theta0=ref["pa_rad"], h0=ref["L"] / 2.0)
            if fit and np.hypot(fit["x"] - x, fit["y"] - y) < 6:
                print(f"empirical fake recovered: mag {mag} L {L:.0f}px SNR {ref['snr']:.0f}")
                return cutout(e.diff, fit["x"], fit["y"], fit["h_px"])
    return None


def main():
    ztf = ztf_panels()          # [a000001 exp1, a000001 exp2]
    sat = g96_panels()          # [G96 joint satellite trail]
    fake = empirical_fake_panel()
    panels = [ztf[0], ztf[1], sat[0], fake]
    labels = ["a000001 ZTF trail 1", "a000001 ZTF trail 2", "G96 satellite",
              "injected fake (empirical PSF)"]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.4), facecolor="black")
    axes = axes.ravel()
    for ax, sub in zip(axes, panels):
        g = np.isfinite(sub)
        m = np.median(sub[g]); s = 1.4826 * np.median(np.abs(sub[g] - m))
        ax.imshow(np.where(g, sub, m), vmin=m - 2 * s, vmax=m + 8 * s,
                  cmap="gray_r", origin="lower", interpolation="nearest", aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,
                        wspace=0.03, hspace=0.03)
    out = os.path.join(ROOT, "work", "mosaic_select.png")
    fig.savefig(out, dpi=140, facecolor="black")
    print("wrote", out, "| panel order:", labels)


if __name__ == "__main__":
    main()
