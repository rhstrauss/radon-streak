#!/usr/bin/env python
"""Unit checks: hldet writer column order, tile grid coverage, dedup, kernel norm.

Run: cd streak_radon && <mpchecker python> tests/test_units.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from streakradon import hldet_io                             # noqa: E402
from streakradon.frt_driver import tile_grid, dedup_candidates, detect_streaks  # noqa: E402
from streakradon.mf_snr import trail_kernel, mf_snr_at       # noqa: E402
from streakradon import frt                                  # noqa: E402


def test_hldet_column_order():
    row = dict(MJD=60431.15319275, RA=136.4911006, Dec=25.4167418, mag=18.5,
               trail_len=30.4, trail_PA=123.4, sigmag=0.08, sig_across=0.35,
               sig_along=1.20, image=3, idstring="G96_test_0001", band="G",
               obscode="G96", known_obj=0, det_qual=0.85)
    with tempfile.NamedTemporaryFile("r", suffix=".csv", delete=False) as f:
        path = f.name
    hldet_io.write_hldet(path, [row])
    lines = open(path).read().strip().split("\n")
    os.unlink(path)
    assert lines[0] == hldet_io.HEADER
    cols = lines[1].split(",")
    assert len(cols) == 15, f"expected 15 cols got {len(cols)}"
    # spot-check the load-bearing positions (1-indexed contract):
    assert abs(float(cols[0]) - row["MJD"]) < 1e-8          # MJDCOL 1
    assert abs(float(cols[4]) - row["trail_len"]) < 0.01    # TRAILLENCOL 5
    assert abs(float(cols[5]) - row["trail_PA"]) < 0.01     # TRAILPACOL 6
    assert cols[10] == row["idstring"]                      # IDCOL 11
    assert cols[12] == row["obscode"]                       # OBSCODECOL 13
    print("test_hldet_column_order OK")


def test_tile_grid():
    tiles = tile_grid((5280, 5280), tile=1024, overlap=128)
    # full coverage incl. clamped last row/col
    ymax = max(t[0] for t in tiles) + 1024
    xmax = max(t[1] for t in tiles) + 1024
    assert ymax == 5280 and xmax == 5280
    assert all(y + 1024 <= 5280 and x + 1024 <= 5280 for y, x in tiles)
    # neighboring strides overlap
    ys = sorted({t[0] for t in tiles})
    assert all(ys[i + 1] - ys[i] <= 1024 - 128 for i in range(len(ys) - 1))
    print(f"test_tile_grid OK ({len(tiles)} tiles)")


def test_dedup():
    a = dict(x=100.0, y=100.0, pa_rad=0.5, L_frt=40.0, snr_frt=10.0)
    b = dict(x=104.0, y=102.0, pa_rad=0.55, L_frt=45.0, snr_frt=12.0)   # dup of a
    c = dict(x=400.0, y=400.0, pa_rad=1.2, L_frt=30.0, snr_frt=8.0)     # distinct
    out = dedup_candidates([a, b, c])
    assert len(out) == 2
    assert any(o["snr_frt"] == 12.0 for o in out)  # keeps max-SNR member
    print("test_dedup OK")


def test_mf_snr_recovers_injected():
    rng = np.random.default_rng(7)
    img = rng.normal(0, 1.0, (301, 301))
    sigma, L, pa = 1.4, 40.0, 0.6
    k = trail_kernel(L, pa, sigma)
    k /= np.sqrt((k * k).sum())            # unit-norm template
    amp = 12.0                             # -> expected SNR ~ amp
    r = k.shape[0] // 2
    img[150 - r:150 + r + 1, 150 - r:150 + r + 1] += amp * k
    snr = mf_snr_at(img, 150, 150, L, pa, sigma, noise=1.0)
    assert snr > 0.8 * amp, f"MF SNR {snr:.1f} vs injected {amp}"
    off = mf_snr_at(img, 150, 40, L, pa + 0.8, sigma, noise=1.0)
    assert abs(off) < 6.0
    print(f"test_mf_snr_recovers_injected OK (snr={snr:.1f})")


def test_frt_native_exact():
    """The clean-room FRT recursion equals a direct sum over its own digital
    lines -- the proof of correctness (independent of any reference impl)."""
    rng = np.random.default_rng(0)
    for (H, W) in [(16, 16), (13, 32)]:
        img = rng.standard_normal((H, W))
        R, row0 = frt.radon_pos(img)
        W2 = R.shape[0]
        for T in (0, 1, W2 // 2, W2 - 1):
            for k in (row0, row0 + H // 2):
                y0 = k - row0
                s = sum(img[r, c] for (r, c) in frt.brady_line(T, y0, W2)
                        if 0 <= r < H and 0 <= c < W)
                assert abs(s - R[T, k]) < 1e-9
    print("test_frt_native_exact OK")


def test_frt_native_detect():
    """native detect_streaks backend finds a short streak in noise with the
    right center and PA, via the multi-length folding path."""
    rng = np.random.default_rng(5)
    N = 256
    img = rng.standard_normal((N, N))
    cx, cy, pa_deg, L = 150.0, 120.0, 35.0, 34
    th = np.radians(pa_deg)
    for t in np.linspace(-L / 2, L / 2, 2 * L):
        xx = int(round(cx + t * np.cos(th))); yy = int(round(cy + t * np.sin(th)))
        img[yy, xx] += 1.6
    cands = detect_streaks(img, 1.4, tile=256, overlap=0,
                           min_length=8, threshold=6.0, backend="native")
    near = [c for c in cands if np.hypot(c["x"] - cx, c["y"] - cy) < 8]
    assert near, f"no candidate near injected streak among {len(cands)}"
    best = max(near, key=lambda c: c["snr_frt"])
    dpa = abs(np.degrees(best["pa_rad"]) - pa_deg) % 180
    assert min(dpa, 180 - dpa) < 12
    print(f"test_frt_native_detect OK ({len(cands)} cands, snr={best['snr_frt']:.1f})")


if __name__ == "__main__":
    test_hldet_column_order()
    test_tile_grid()
    test_dedup()
    test_mf_snr_recovers_injected()
    test_frt_native_exact()
    test_frt_native_detect()
    print("ALL UNIT TESTS PASSED")
