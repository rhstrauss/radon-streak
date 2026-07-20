#!/usr/bin/env python
"""Drive the vendored pyradon Finder over a full frame: square tiling, manual
per-tile invocation, candidate extraction in global coordinates, union-find dedup.

Why manual: pyradon's convenience input() path assumes square images and its
absolute SNR normalization is unreliable on real survey diffs. We feed
PRE-WHITENED tiles (image/sqrt(variance), masked px = 0) with scalar variance
1.0, set pars.min_length to a PHYSICAL value (default 8 -- the library default
of 32 gates out the folding where short-trail SNR peaks; benchmarked 2026-07 on
a000001: SNR 6.3 -> 19.0), and treat pyradon's (L, SNR) as hints only. Every
candidate is re-scored downstream with mf_snr and re-measured by the Veres fit.
"""
import numpy as np

from . import import_pyradon


def make_finder(psf_sigma_px, min_length=8, threshold=5.0, num_iterations=5):
    Finder = import_pyradon()
    f = Finder()
    f.pars.use_short = True
    f.pars.min_length = int(min_length)
    f.pars.threshold = float(threshold)
    f.pars.num_iterations = int(num_iterations)
    f.pars.use_sections = False       # we tile ourselves (squares, overlap)
    f.pars.use_exclude = False        # do not blank the tile center bands
    f.pars.use_show = False
    f.pars.verbosity = 0
    f.data.variance = 1.0             # pre-whitened input
    f.data.psf = float(psf_sigma_px)
    return f


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
                   min_length=8, threshold=5.0, finder=None):
    """Run FRT streak detection on a pre-whitened image.

    whitened : 2D float array, image/sqrt(var); masked/bad pixels should be 0
               (or NaN -- converted to 0 after mean subtraction inside pyradon).
    Returns list of candidate dicts:
      x, y   : streak center, global pixel coords
      pa_rad : pixel-frame angle (atan2 convention, mod pi)
      L_frt  : pyradon length hint (px)
      snr_frt: pyradon SNR hint
      x1,y1,x2,y2 : endpoints, global
    """
    if finder is None:
        finder = make_finder(psf_sigma_px, min_length, threshold)
    H, W = whitened.shape
    cands = []
    for (y0, x0) in tile_grid((H, W), tile, overlap):
        sub = np.array(whitened[y0:y0 + tile, x0:x0 + tile], dtype=float, copy=True)
        if not np.isfinite(sub).any() or np.count_nonzero(sub) < 100:
            continue
        sub[~np.isfinite(sub)] = 0.0
        finder.clear()  # resets streaks list + section corner + best_snr
        finder.data._current_section_corner = (y0, x0)
        proc = finder.preprocess(sub)
        finder.scan_thresholds(proc)
        for s in finder.streaks:
            try:
                x1, y1, x2, y2 = float(s.x1f), float(s.y1f), float(s.x2f), float(s.y2f)
            except TypeError:
                continue
            pa = np.arctan2(y2 - y1, x2 - x1) % np.pi
            cands.append(dict(
                x=0.5 * (x1 + x2), y=0.5 * (y1 + y2), pa_rad=float(pa),
                L_frt=float(s.L) if s.L is not None else np.hypot(x2 - x1, y2 - y1),
                snr_frt=float(s.snr), x1=x1, y1=y1, x2=x2, y2=y2,
                tile=(y0, x0)))
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
