#!/usr/bin/env python
"""P2 QA driver: run the G96 preprocessing chain on the example sequence,
report registration/PSF/variance diagnostics, write QA PNGs."""
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from streakradon.adapters.g96 import prepare_sequence  # noqa: E402

try:
    import yaml
    CFG = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "config", "g96.yaml")))
except Exception:
    CFG = {}

PATHS = sorted(glob.glob(
    "/astro/store/shire/aheinze/Shared/G96_images/G96_20240501_2B_N25057_01_000?.arch.fz"))
QA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qa")


def main():
    t0 = time.time()
    exps = prepare_sequence(PATHS, CFG, verbose=True, qa_dir=QA)
    print(f"\nprepared {len(exps)} exposures in {time.time()-t0:.0f}s")
    for e in exps:
        g = e.mask == 0
        w = e.white[g]
        print(f"exp{e.image_index}: mid_mjd {e.mid_mjd:.6f}  magzp {e.magzp:.3f}  "
              f"masked {100*(1-g.mean()):.2f}%  white mean {w.mean():+.4f} "
              f"std {w.std():.3f} (want ~1)  |white|>5: {(np.abs(w)>5).sum()}")
    # blank-sky chi2/dof check on a central region away from masks
    e = exps[0]
    c = slice(2000, 3000)
    sub = e.white[c, c][e.mask[c, c] == 0]
    print(f"central blank-sky white std {sub.std():.3f} (gate 0.8-1.2), "
          f"kurtosis-ish frac|>3| {(np.abs(sub)>3).mean():.5f} (Gauss 0.0027)")


if __name__ == "__main__":
    main()
