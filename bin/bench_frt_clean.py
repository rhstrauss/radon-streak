#!/usr/bin/env python
"""Clean re-measurement of the native FRT cost.

The first timings of _fold_detect were taken on a node running at load 40-80
(a 20-worker batch plus other users) and disagreed with cProfile by ~1000x
(114 s wall vs 0.095 s cumtime). Wall-clock on a contended node is not a
measurement. This script:
  * reports system load so the numbers are interpretable,
  * uses time.process_time() (CPU time, immune to contention) alongside wall,
  * repeats and reports the MINIMUM (least-contaminated sample).
"""
import os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from streakradon import frt as F                      # noqa: E402
from streakradon.frt_driver import detect_streaks     # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "bin"))
from opt_frt_prototype import fold_detect_fast        # noqa: E402


def bench(fn, *a, reps=3):
    wall, cpu = [], []
    for _ in range(reps):
        w0, c0 = time.time(), time.process_time()
        r = fn(*a)
        wall.append(time.time() - w0)
        cpu.append(time.process_time() - c0)
    return min(wall), min(cpu), r


def tile(size, nstreak=4, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.normal(0, 1, (size, size))
    for _ in range(nstreak):
        x0, y0 = rng.integers(60, size - 90, 2)
        ang = rng.uniform(0, np.pi); L = rng.uniform(100, 250)
        for t in np.linspace(0, 1, 600):
            x = int(x0 + L * np.cos(ang) * t); y = int(y0 + L * np.sin(ang) * t)
            if 1 <= y < size - 1 and 1 <= x < size - 1:
                img[y - 1:y + 2, x - 1:x + 2] += 6
    return img


def main():
    print("load average (1/5/15):", os.getloadavg(), " cores:", os.cpu_count())
    print()
    for size in (512, 1024):
        img = tile(size)
        w1, c1, r1 = bench(F._fold_detect, img, 8, 5.0, 7, 200)
        w2, c2, r2 = bench(fold_detect_fast, img, 8, 5.0, 7, 200)
        print(f"_fold_detect @ {size}^2  (one family)")
        print(f"  original : wall {w1:7.2f}s   CPU {c1:7.2f}s   {len(r1)} peaks")
        print(f"  optimized: wall {w2:7.2f}s   CPU {c2:7.2f}s   {len(r2)} peaks")
        print(f"  speedup  : wall {w1/max(w2,1e-9):6.1f}x   CPU {c1/max(c2,1e-9):6.1f}x")
        ok = sorted(r1) == sorted(r2)
        print(f"  identical: {ok}")
        print()
    # full detect_streaks (4 families + tiling), which is what the pipeline calls
    img = tile(1024)
    w, c, cands = bench(detect_streaks, img, 1.05, 1024, 128, 8, 5.0, reps=2)
    print(f"detect_streaks @1024^2 (native, 4 families): wall {w:.2f}s  CPU {c:.2f}s"
          f"  {len(cands)} cands")
    print(f"  -> per 5280^2 exposure (36 tiles): {c*36:.0f}s CPU")


if __name__ == "__main__":
    main()
