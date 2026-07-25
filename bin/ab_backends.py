#!/usr/bin/env python
"""A/B the two detection backends: our clean-room native FRT vs the vendored
pyradon Finder (the latter WITH the fastmask capsule patch, so the comparison is
fair rather than flattered by pyradon's model-subtract bottleneck).

Reports per-tile runtime, candidate count, and recovery of injected streaks.
"""
import os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from streakradon.frt_driver import detect_streaks   # noqa: E402


def make_tile(n_streaks=8, seed=3, size=1024):
    rng = np.random.default_rng(seed)
    img = rng.normal(0, 1, (size, size))
    truth = []
    for _ in range(n_streaks):
        x0, y0 = rng.integers(80, size - 120, 2)
        ang = rng.uniform(0, np.pi)
        L = rng.uniform(150, 450)
        truth.append((x0 + L * np.cos(ang) / 2, y0 + L * np.sin(ang) / 2))
        for t in np.linspace(0, 1, 800):
            x = int(x0 + L * np.cos(ang) * t); y = int(y0 + L * np.sin(ang) * t)
            if 1 <= y < size - 1 and 1 <= x < size - 1:
                img[y - 1:y + 2, x - 1:x + 2] += 6
    return img, truth


def main():
    img, truth = make_tile()
    res = {}
    for be in ("native", "pyradon"):
        try:
            t = time.time()
            c = detect_streaks(img, 1.05, tile=1024, overlap=128, backend=be)
            dt = time.time() - t
            rec = sum(1 for tx, ty in truth
                      if any(np.hypot(d["x"] - tx, d["y"] - ty) < 40 for d in c))
            res[be] = (dt, len(c), c)
            print(f"{be:8s}: {dt:8.2f}s  {len(c):4d} cands  "
                  f"injected recovered {rec}/{len(truth)}  "
                  f"-> per-exposure est {dt*36:7.0f}s", flush=True)
        except Exception as ex:
            print(f"{be:8s}: FAILED {type(ex).__name__}: {str(ex)[:90]}", flush=True)
    if "native" in res and "pyradon" in res:
        print(f"\nnative is {res['pyradon'][0]/res['native'][0]:.1f}x faster than "
              f"pyradon (pyradon already has the fastmask capsule patch)")
        # positional agreement between backends
        n, p = res["native"][2], res["pyradon"][2]
        matched = 0
        for a in p:
            if any(np.hypot(a["x"] - b["x"], a["y"] - b["y"]) < 25 for b in n):
                matched += 1
        print(f"pyradon candidates with a native counterpart (<25px): "
              f"{matched}/{len(p)}")


if __name__ == "__main__":
    main()
