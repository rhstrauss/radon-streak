#!/usr/bin/env python
"""Profile streakradon.frt._fold_detect and prototype two optimizations,
verifying EXACT equivalence before any change lands in the clean-room module.

Opt 1 -- vectorize the `for T in range(bw)` loop. It is a pure gather-shift-add:
        new[b,T,k] = L[b,T//2,k] + Rt[b,T//2,k+shift(T)]  (Rt term 0 past the edge)
        so the whole level can be done with one take_along_axis instead of bw
        Python iterations (1024 of them at the last level).

Opt 2 -- threshold BEFORE non-maximum suppression. snr = sum/sqrt(bw) has unit
        variance for unit-variance input, so at threshold 5 only ~3e-7 of voxels
        survive (~1 in 3.1M). Running scipy maximum_filter over the full
        (nblocks, bw, Hpad) array (~25 MB, 40x per tile) to find them is
        enormously wasteful; test the few above-threshold points directly.
"""
import os, sys, time, cProfile, pstats, io
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from streakradon import frt as F   # noqa: E402


# --------------------------------------------------------------------------- #
def fold_detect_fast(work, min_length, threshold, nms, top_per_level):
    """Optimized twin of frt._fold_detect (identical output)."""
    H, W = work.shape
    W2 = 1 << (W - 1).bit_length()
    pad = W2
    Hpad = H + 2 * pad
    part = np.zeros((W2, 1, Hpad), dtype=np.float64)
    part[:W, 0, pad:pad + H] = np.asarray(work, dtype=np.float64).T
    nblocks, bw = W2, 1
    out = []
    kcol = np.arange(Hpad)
    while nblocks > 1:
        nblocks //= 2
        bw *= 2
        Lh = part[0::2]
        Rt = part[1::2]
        # ---- Opt 1: vectorized level ----
        T = np.arange(bw)
        tL = T // 2
        shift = tL + (T & 1)                       # (bw,)
        # gather Rt at column k+shift, zero past the edge
        idx = kcol[None, :] + shift[:, None]        # (bw, Hpad)
        valid = idx < Hpad
        idxc = np.where(valid, idx, 0)
        Rt_e = Rt[:, tL, :]                         # (nblocks, bw, Hpad)
        gathered = np.take_along_axis(
            Rt_e, np.broadcast_to(idxc, Rt_e.shape), axis=2)
        gathered = np.where(valid[None, :, :], gathered, 0.0)
        new = Lh[:, tL, :] + gathered
        part = new
        if bw >= min_length:
            # ---- Opt 2: threshold first, NMS only on survivors ----
            snr = new * (1.0 / np.sqrt(bw))
            above = snr > threshold
            if not above.any():
                continue
            cand = np.argwhere(above)
            half = nms // 2
            keep = []
            for (b, t, k) in cand:
                v = snr[b, t, k]
                t0, t1 = max(0, t - 1), min(snr.shape[1], t + 2)
                k0, k1 = max(0, k - half), min(snr.shape[2], k + half + 1)
                if v >= snr[b, t0:t1, k0:k1].max():
                    keep.append((b, t, k, v))
            if len(keep) > top_per_level:
                keep.sort(key=lambda z: -z[3])
                keep = keep[:top_per_level]
            for (b, t, k, v) in keep:
                out.append((bw, int(b) * bw, int(t), int(k) - pad, float(v)))
    return out


# --------------------------------------------------------------------------- #
def main():
    rng = np.random.default_rng(0)
    size = 512                       # smaller tile so the SLOW original finishes
    img = rng.normal(0, 1, (size, size))
    for _ in range(4):
        x0, y0 = rng.integers(60, size - 90, 2)
        ang = rng.uniform(0, np.pi); L = rng.uniform(100, 250)
        for t in np.linspace(0, 1, 600):
            x = int(x0 + L * np.cos(ang) * t); y = int(y0 + L * np.sin(ang) * t)
            if 1 <= y < size - 1 and 1 <= x < size - 1:
                img[y - 1:y + 2, x - 1:x + 2] += 6

    print(f"tile {size}x{size}\n")
    t = time.time(); a = F._fold_detect(img, 8, 5.0, 7, 200); t_old = time.time() - t
    t = time.time(); b = fold_detect_fast(img, 8, 5.0, 7, 200); t_new = time.time() - t
    print(f"  original _fold_detect : {t_old:8.2f}s  -> {len(a)} peaks")
    print(f"  optimized             : {t_new:8.2f}s  -> {len(b)} peaks")
    print(f"  SPEEDUP               : {t_old/max(t_new,1e-9):8.1f}x")

    sa, sb = sorted(a), sorted(b)
    identical = (len(sa) == len(sb)) and all(
        x[:4] == y[:4] and abs(x[4] - y[4]) < 1e-9 for x, y in zip(sa, sb))
    print(f"  IDENTICAL OUTPUT      : {identical}")
    if not identical:
        only_a = [x for x in sa if x not in sb][:5]
        only_b = [x for x in sb if x not in sa][:5]
        print(f"    only original: {only_a}")
        print(f"    only optimized: {only_b}")

    print("\n  --- profile of the ORIGINAL (where the time goes) ---")
    pr = cProfile.Profile(); pr.enable(); F._fold_detect(img, 8, 5.0, 7, 200); pr.disable()
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(6)
    print("\n".join(s.getvalue().splitlines()[4:14]))


if __name__ == "__main__":
    main()
