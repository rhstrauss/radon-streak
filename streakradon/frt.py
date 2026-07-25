# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ryder H. Strauss
#
# Independent, clean-room implementation of the Fast (discrete) Radon Transform.
# Written from the published algorithm -- Gotz & Druckmuller (1996), Brady (1998,
# SIAM J. Comput. 27:107), and its streak-detection application in Nir, Ofek,
# Ben-Ami & Zackay (2018, AJ 156:229) -- NOT derived from any existing source
# implementation (in particular, not from pyradon). The Radon transform and the
# recursive-doubling DRT are published mathematics and carry no copyright; only a
# specific source expression would. See LICENSING.md / NOTICE.
"""Fast discrete Radon transform for streak detection.

The discrete Radon transform (DRT) of an image sums pixel values along straight
"digital lines." A trailed source (streak) is a straight line of flux, so it
produces a sharp peak in Radon space at the (offset, slope) of the trail. The
naive DRT is O(N^3) for an N x N image; the recursive-doubling FRT is O(N^2 logN)
by reusing partial line-sums between adjacent column blocks.

Digital-line convention (this implementation)
---------------------------------------------
`radon_pos(image)` returns R[T, y0] for lines of NON-NEGATIVE slope that span all
W columns: the line enters at (col 0, row y0) and exits at (col W-1, row y0 + T),
T in 0..W-1 (slope = T / (W-1), i.e. 0 deg .. 45 deg). The line is built by the
same recursive halving used to compute it (see `brady_line`), so `radon_pos` and
a direct sum over `brady_line` agree exactly -- that identity is the module's
correctness test.

Full 0..180 deg coverage is assembled from four calls on the image and its
transpose / vertical flip (`radon_forward`).
"""
import numpy as np


# --------------------------------------------------------------------------- #
# Core recursion (one octant: slopes 0 .. +1)
# --------------------------------------------------------------------------- #
def radon_pos(image):
    """Discrete Radon transform for slopes in [0, 1] (0 deg .. 45 deg).

    image : 2D float array (H, W). W is padded up to a power of two internally
            (zeros on the right); H is padded above and below by W so lines that
            enter/exit through the top/bottom edge are represented.

    Returns
    -------
    R   : (W2, Hpad) array. R[T, k] is the sum along the digital line entering at
          (col 0, row k - W2) and exiting at (col W2-1, row k - W2 + T), where the
          real image occupies rows [0, H) and W2 is the padded (power-of-two) width.
    row0: integer offset such that R[T, k] has entry-row  y0 = k - row0.
    """
    image = np.asarray(image, dtype=np.float64)
    H, W = image.shape
    W2 = 1 << (W - 1).bit_length()          # next power of two >= W
    pad = W2
    Hpad = H + 2 * pad
    # working tensor: part[b, t, k]  (block, in-block rise, entry row in padded frame)
    # base case blockwidth 1: one column per block, only rise t = 0.
    part = np.zeros((W2, 1, Hpad), dtype=np.float64)
    part[:W, 0, pad:pad + H] = image.T      # part[b,0,k] = image[k-pad, b]

    nblocks, blockwidth = W2, 1
    while nblocks > 1:
        nblocks //= 2
        blockwidth *= 2
        newt = blockwidth                   # rises 0 .. blockwidth-1
        L = part[0::2]                       # (nblocks, oldt, Hpad)
        Rt = part[1::2]
        new = np.zeros((nblocks, newt, Hpad), dtype=np.float64)
        for T in range(newt):
            tL = T // 2                     # left in-block rise
            tR = T // 2                     # right in-block rise
            j = T & 1                       # junction step (0 or 1)
            shift = tL + j                  # right block entry-row offset
            # new[:, T, k] = L[:, tL, k] + Rt[:, tR, k + shift]
            if shift == 0:
                new[:, T, :] = L[:, tL, :] + Rt[:, tR, :]
            else:
                new[:, T, :-shift] = L[:, tL, :-shift] + Rt[:, tR, shift:]
                new[:, T, -shift:] = L[:, tL, -shift:]   # off-image tail: line runs out the bottom
        part = new
    return part[0], pad                     # (W2, Hpad), row0=pad


def brady_line(T, y0, W):
    """Pixel path (list of (row, col)) of the digital line `radon_pos` sums for
    rise T, entry row y0, width W. Used only as the exact correctness reference
    and for endpoint geometry. Built by the SAME halving recursion as radon_pos."""
    if W == 1:
        return [(y0, 0)]
    if W % 2:
        raise ValueError("brady_line width must be a power of two")
    half = W // 2
    tL = T // 2
    j = T & 1
    left = brady_line(tL, y0, half)
    right = [(r, c + half) for (r, c) in brady_line(T // 2, y0 + tL + j, half)]
    return left + right


# --------------------------------------------------------------------------- #
# Full-angle forward transform (four octants)
# --------------------------------------------------------------------------- #
def radon_forward(image):
    """Radon transform over all orientations 0..180 deg.

    Returns a dict of four (R, row0) octant results keyed by orientation family:
      'pos'   slopes  0..+1  on the image           (  0.. 45 deg, measured from +x)
      'neg'   slopes  0..-1  (vertical flip)         (135..180 deg)
      'posT'  steep  +1..+inf (transpose)            ( 45.. 90 deg)
      'negT'  steep  -1..-inf (transpose+flip)       ( 90..135 deg)
    Endpoint geometry for each family is recovered by `endpoints_for`.
    """
    H, W = image.shape
    out = {}
    out['pos'] = radon_pos(image)
    out['neg'] = radon_pos(image[::-1, :])
    out['posT'] = radon_pos(image.T)
    out['negT'] = radon_pos(image.T[::-1, :])
    out['_shape'] = (H, W)
    return out


def endpoints_for(family, T, y0, shape):
    """Map an octant peak (family, rise T, entry row y0) back to image-frame
    endpoints (x1, y1, x2, y2). Inverse of the flips/transpose in radon_forward.

    Assumes square, power-of-two tiles (H == W, both powers of two) -- how the
    frame tiler feeds this module. y0 is the entry row *in the transformed frame*
    as returned by the peak search (k - row0), not the injected value.
    """
    H, W = shape
    if family == 'pos':                        # slopes 0..+1
        return (0.0, float(y0), float(W - 1), float(y0 + T))
    if family == 'neg':                        # slopes 0..-1 (vertical flip)
        return (0.0, float(H - 1 - y0), float(W - 1), float(H - 1 - y0 - T))
    if family == 'posT':                       # steep +, transpose
        return (float(y0), 0.0, float(y0 + T), float(H - 1))
    if family == 'negT':                       # steep -, transpose + flip
        return (float(W - 1 - y0), 0.0, float(W - 1 - y0 - T), float(H - 1))
    raise ValueError(family)


# --------------------------------------------------------------------------- #
# Detection: multi-length "foldings" (the min_length fix, built in)
# --------------------------------------------------------------------------- #
def _work_to_image(fam, xw, yw, H, W):
    """Invert the flip/transpose that produced a working frame, mapping a
    working-frame point (col xw, row yw) back to image (x, y)."""
    if fam == 'pos':
        return xw, yw
    if fam == 'neg':
        return xw, H - 1 - yw
    if fam == 'posT':
        return yw, xw
    if fam == 'negT':
        return W - 1 - yw, xw
    raise ValueError(fam)


def _fold_detect(work, min_length, threshold, nms, top_per_level):
    """Run the doubling recursion on one working frame and harvest, at every
    dyadic block width L >= min_length, local-max peaks of SNR = sum/sqrt(L)
    above `threshold`. A length-L trail peaks at the folding nearest its length,
    so short trails are found without the full-width noise dilution.

    Returns list of (L, c0, T, y0, snr): block col-offset c0, in-block rise T,
    entry row y0 (working frame)."""
    from scipy.ndimage import maximum_filter
    H, W = work.shape
    W2 = 1 << (W - 1).bit_length()
    pad = W2
    Hpad = H + 2 * pad
    part = np.zeros((W2, 1, Hpad), dtype=np.float64)
    part[:W, 0, pad:pad + H] = np.asarray(work, dtype=np.float64).T
    nblocks, bw = W2, 1
    out = []
    while nblocks > 1:
        nblocks //= 2
        bw *= 2
        L = part[0::2]
        Rt = part[1::2]
        new = np.zeros((nblocks, bw, Hpad), dtype=np.float64)
        for T in range(bw):
            tL = T // 2
            shift = tL + (T & 1)
            if shift == 0:
                new[:, T, :] = L[:, tL, :] + Rt[:, tL, :]
            else:
                new[:, T, :-shift] = L[:, tL, :-shift] + Rt[:, tL, shift:]
                new[:, T, -shift:] = L[:, tL, -shift:]
        part = new
        if bw >= min_length:
            snr = new / np.sqrt(bw)
            mx = maximum_filter(snr, size=(1, 3, nms))
            hits = np.argwhere((snr == mx) & (snr > threshold))
            if len(hits) > top_per_level:
                order = np.argsort(snr[hits[:, 0], hits[:, 1], hits[:, 2]])[::-1]
                hits = hits[order[:top_per_level]]
            for (b, T, k) in hits:
                out.append((bw, int(b) * bw, int(T), int(k) - pad,
                            float(snr[b, T, k])))
    return out


def detect(image, min_length=8, threshold=5.0, nms=7, top_per_level=200):
    """Locate streak candidates in a pre-whitened image (unit variance; masked
    pixels 0). Square, power-of-two tiles assumed (see frt_driver.tile_grid).

    Returns candidate dicts (tile-local coords): x, y (center), pa_rad, length,
    snr, x1, y1, x2, y2. These are HINTS -- downstream mf_snr re-scores and the
    Veres fit re-measures every candidate (see PATCHES.md / __init__)."""
    H, W = image.shape
    cands = []
    frames = {'pos': image, 'neg': image[::-1, :],
              'posT': image.T, 'negT': image.T[::-1, :]}
    for fam, work in frames.items():
        for (L, c0, T, y0, snr) in _fold_detect(work, min_length, threshold,
                                                 nms, top_per_level):
            x1, y1 = _work_to_image(fam, c0, y0, H, W)
            x2, y2 = _work_to_image(fam, c0 + L - 1, y0 + T, H, W)
            pa = np.arctan2(y2 - y1, x2 - x1) % np.pi
            cands.append(dict(x=0.5 * (x1 + x2), y=0.5 * (y1 + y2),
                              pa_rad=float(pa),
                              length=float(np.hypot(x2 - x1, y2 - y1)),
                              snr=float(snr),
                              x1=float(x1), y1=float(y1),
                              x2=float(x2), y2=float(y2)))
    return cands


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def _selftest():
    rng = np.random.default_rng(0)
    # 1) EXACT identity: radon_pos == direct sum over brady_line, for random images
    #    of power-of-two width (the internal-correctness proof).
    for (H, W) in [(8, 8), (16, 8), (13, 16), (16, 16), (9, 32)]:
        img = rng.standard_normal((H, W))
        R, row0 = radon_pos(img)
        W2 = R.shape[0]
        bad = 0
        for T in range(W2):
            for k in range(R.shape[1]):
                y0 = k - row0
                s = 0.0
                for (r, c) in brady_line(T, y0, W2):
                    if 0 <= r < H and 0 <= c < W:
                        s += img[r, c]
                if abs(s - R[T, k]) > 1e-9:
                    bad += 1
        assert bad == 0, f"radon_pos != brady_line sum ({bad} mismatches) at {(H,W)}"
    print("PASS: radon_pos matches brady_line exactly (5 shapes)")

    # 2) Linearity
    a = rng.standard_normal((16, 16)); b = rng.standard_normal((16, 16))
    Ra, _ = radon_pos(a); Rb, _ = radon_pos(b); Rab, _ = radon_pos(3 * a - 2 * b)
    assert np.allclose(Rab, 3 * Ra - 2 * Rb, atol=1e-9)
    print("PASS: linearity")

    # 3) Streak recovery: inject a unit-flux digital line for each octant, then
    #    verify the GLOBAL peak lands in that octant with the right amplitude AND
    #    that endpoints_for() recovers the injected line's true image-frame
    #    endpoints -- this validates the whole forward + geometry chain.
    N = 64

    def inject(fam, T, y0, amp):
        img = np.zeros((N, N))
        pts = brady_line(T, y0, N)
        if fam == 'pos':
            lit = [(r, c) for (r, c) in pts]
        elif fam == 'neg':
            lit = [(N - 1 - r, c) for (r, c) in pts]
        elif fam == 'posT':
            lit = [(c, r) for (r, c) in pts]
        else:
            lit = [(N - 1 - c, r) for (r, c) in pts]
        for (a, b) in lit:
            if 0 <= a < N and 0 <= b < N:
                img[a, b] += amp
        return img, lit

    for fam, T_true, y0_true, amp in [('pos', 20, 6, 0.7), ('neg', 20, 40, 0.7),
                                      ('posT', 6, 20, 0.7), ('negT', 40, 20, 0.7)]:
        img, lit = inject(fam, T_true, y0_true, amp)
        fwd = radon_forward(img)
        # global winner across octants
        best = max(('pos', 'neg', 'posT', 'negT'),
                   key=lambda f: fwd[f][0].max())
        R, row0 = fwd[best]
        Tpk, kpk = np.unravel_index(np.argmax(R), R.shape)
        assert abs(R[Tpk, kpk] - amp * N) < 1e-6, f"{fam}: peak {R[Tpk,kpk]} != {amp*N}"
        x1, y1, x2, y2 = endpoints_for(best, Tpk, kpk - row0, (N, N))
        # injected line's true endpoints = first/last lit pixel by column then row
        cols = [b for (a, b) in lit]; rows = [a for (a, b) in lit]
        assert min(x1, x2) == min(cols) and max(x1, x2) == max(cols) \
            and min(y1, y2) == min(rows) and max(y1, y2) == max(rows), \
            f"{fam}: recovered endpoints ({x1},{y1})-({x2},{y2}) " \
            f"vs injected x[{min(cols)},{max(cols)}] y[{min(rows)},{max(rows)}]"
    print("PASS: streak recovery + amplitude + endpoint geometry, all four octants")

    # 4) Detector on a NOISY tile with a SHORT streak: the folding path must find
    #    a ~30 px trail in a 128 px tile (full-width integration would bury it).
    N = 128
    rng2 = np.random.default_rng(3)
    for (cx, cy, pa_deg, Lpx) in [(70, 60, 30.0, 30), (40, 90, 115.0, 26)]:
        img = rng2.standard_normal((N, N))          # unit-variance whitened noise
        th = np.radians(pa_deg); amp = 1.6
        for t in np.linspace(-Lpx / 2, Lpx / 2, 2 * Lpx):
            xx = int(round(cx + t * np.cos(th))); yy = int(round(cy + t * np.sin(th)))
            if 0 <= xx < N and 0 <= yy < N:
                img[yy, xx] += amp
        cands = detect(img, min_length=8, threshold=6.0)
        near = [c for c in cands if np.hypot(c['x'] - cx, c['y'] - cy) < 8]
        assert near, f"streak ({cx},{cy},{pa_deg}deg) not detected among {len(cands)} cands"
        best = max(near, key=lambda c: c['snr'])
        dpa = abs(np.degrees(best['pa_rad']) - (pa_deg % 180))
        dpa = min(dpa, 180 - dpa)
        assert dpa < 12, f"PA off: {np.degrees(best['pa_rad']):.1f} vs {pa_deg}"
    print("PASS: detector finds short streaks in noise (both orientations)")
    print("ALL FRT SELF-TESTS PASSED")


if __name__ == "__main__":
    _selftest()
