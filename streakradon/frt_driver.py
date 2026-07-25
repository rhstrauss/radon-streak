# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ryder H. Strauss
"""Drive streak detection over a full frame: square tiling, per-tile Fast Radon
Transform (streakradon.frt, our own clean-room FRT -- no third-party code),
candidate extraction in global coordinates, union-find dedup.

We feed PRE-WHITENED tiles (image/sqrt(variance), masked px = 0), set min_length
to a PHYSICAL value (default 8 -- 32 gates out the folding where short-trail SNR
peaks; benchmarked 2026-07 on a000001: SNR 6.3 -> 19.0), and treat FRT (L, SNR)
as hints only: every candidate is re-scored downstream with mf_snr and
re-measured by the Veres fit.
"""
import numpy as np


def tile_grid(shape, tile=1024, overlap=128):
    """Square tile origins covering shape with >=overlap px overlap; final
    row/col clamped so tiles never exceed the frame."""
    H, W = shape
    stride = tile - overlap
    ys = list(range(0, max(H - tile, 0) + 1, stride))
    xs = list(range(0, max(W - tile, 0) + 1, stride))
    if ys[-1] + tile < H:
        ys.append(H - tile)
    if xs[-1] + tile < W:
        xs.append(W - tile)
    return [(y, x) for y in ys for x in xs]


def detect_streaks(whitened, psf_sigma_px, tile=1024, overlap=128,
                   min_length=8, threshold=5.0, finder=None, fast_suppress=True,
                   backend="native"):
    """Run FRT streak detection on a pre-whitened image (streakradon.frt).

    whitened : 2D float array, image/sqrt(var); masked/bad pixels should be 0
               (or NaN -- converted to 0 before the transform).
    backend  : 'native' only; 'pyradon' was removed (see LICENSING.md).
    finder, fast_suppress : accepted for backward compatibility, ignored.
    Returns list of candidate dicts:
      x, y   : streak center, global pixel coords
      pa_rad : pixel-frame angle (atan2 convention, mod pi)
      L_frt  : FRT length hint (px)
      snr_frt: FRT SNR hint
      x1,y1,x2,y2 : endpoints, global
    """
    if backend != "native":
        raise ValueError(
            f"backend {backend!r} unavailable: the pyradon backend was removed; "
            "streakradon uses its own frt.py. Use backend='native'.")
    return _detect_native(whitened, tile, overlap, min_length, threshold)


def _detect_native(whitened, tile=1024, overlap=128, min_length=8, threshold=5.0):
    """Tile the frame, run streakradon.frt.detect per tile, lift candidates to
    global coords, dedup (x, y, pa_rad, L_frt, snr_frt, x1..y2, tile)."""
    from . import frt
    H, W = whitened.shape
    cands = []
    for (y0, x0) in tile_grid((H, W), tile, overlap):
        sub = np.array(whitened[y0:y0 + tile, x0:x0 + tile], dtype=float, copy=True)
        finite = np.isfinite(sub)
        if np.count_nonzero(sub[finite]) < 100:
            continue
        sub[~finite] = 0.0
        for c in frt.detect(sub, min_length=min_length, threshold=threshold):
            cands.append(dict(
                x=c["x"] + x0, y=c["y"] + y0, pa_rad=c["pa_rad"],
                L_frt=c["length"], snr_frt=c["snr"],
                x1=c["x1"] + x0, y1=c["y1"] + y0,
                x2=c["x2"] + x0, y2=c["y2"] + y0, tile=(y0, x0)))
    return dedup_candidates(cands)


def _line_dist(c1, c2):
    """Perpendicular distance from c2's center to c1's infinite line."""
    dx, dy = np.cos(c1['pa_rad']), np.sin(c1['pa_rad'])
    px, py = c2['x'] - c1['x'], c2['y'] - c1['y']
    return abs(-dy * px + dx * py)


def _same_streak(c1, c2, perp_tol=10.0, pa_tol_deg=10.0, along_pad=20.0):
    dpa = abs(np.degrees(c1['pa_rad'] - c2['pa_rad'])) % 180.0
    dpa = min(dpa, 180.0 - dpa)
    if dpa > pa_tol_deg:
        return False
    if _line_dist(c1, c2) > perp_tol and _line_dist(c2, c1) > perp_tol:
        return False
    # along-track proximity: centers within combined half-lengths + pad
    d = np.hypot(c1['x'] - c2['x'], c1['y'] - c2['y'])
    return d <= 0.5 * (c1['L_frt'] + c2['L_frt']) + along_pad


def dedup_candidates(cands, perp_tol=10.0, pa_tol_deg=10.0, along_pad=20.0):
    """Union-find clustering of duplicate candidates (overlapping tiles,
    multiple foldings); keep the max-SNR member of each cluster."""
    n = len(cands)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if _same_streak(cands[i], cands[j], perp_tol, pa_tol_deg, along_pad):
                union(i, j)
    best = {}
    for i, c in enumerate(cands):
        r = find(i)
        if r not in best or c['snr_frt'] > cands[best[r]]['snr_frt']:
            best[r] = i
    return [cands[i] for i in sorted(best.values())]
