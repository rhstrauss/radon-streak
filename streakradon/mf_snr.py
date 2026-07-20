#!/usr/bin/env python
"""Integrated matched-filter SNR at a single candidate line.

The significance statistic of the pipeline. pyradon's absolute SNR is erratic on
real survey diffs (normalization blows up near bright sources, power-of-2 folding
granularity over/under-integrates), so every FRT candidate is re-scored here with
the trail-kernel normalization validated on a000001 ground truth
(ztf_streak/bin/streak_prefilter.py): kernel = line(L) (x) Gaussian(sigma),
zero-mean, SNR = sum(img*k) / (noise * sqrt(sum k^2)).

On a PRE-WHITENED image (img/sqrt(var)) pass noise=1.0.
"""
import numpy as np
from scipy.special import erf


def trail_kernel(L, pa_rad, sigma):
    """Unit-ish line(length L) (x) Gaussian(sigma) kernel, tight stamp, mean-subtracted."""
    r = int(np.ceil(L / 2 + 4 * sigma)) + 1
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(float)
    ct, st = np.cos(pa_rad), np.sin(pa_rad)
    u = xx * ct + yy * st
    v = -xx * st + yy * ct
    s2 = np.sqrt(2.0) * sigma
    k = 0.5 * (erf((L / 2 + u) / s2) - erf((-L / 2 + u) / s2)) * np.exp(-0.5 * (v / sigma) ** 2)
    k -= k.mean()  # zero-sum -> insensitive to DC offset in the diff
    return k


def mf_snr_at(img, x, y, L, pa_rad, sigma, noise=1.0, mask=None):
    """Integrated MF SNR of a trail of length L (px) at angle pa_rad centered (x,y).

    img : difference image (pre-whitened -> noise=1.0, else pass the noise scalar).
    Masked / non-finite pixels contribute 0 (kernel norm unchanged -> conservative).
    """
    k = trail_kernel(L, pa_rad, sigma)
    r = k.shape[0] // 2
    xi, yi = int(round(x)), int(round(y))
    y0, y1 = yi - r, yi + r + 1
    x0, x1 = xi - r, xi + r + 1
    ky0 = max(0, -y0); kx0 = max(0, -x0)
    ky1 = k.shape[0] - max(0, y1 - img.shape[0])
    kx1 = k.shape[1] - max(0, x1 - img.shape[1])
    if ky1 <= ky0 or kx1 <= kx0:
        return 0.0
    sub = img[max(0, y0):min(img.shape[0], y1), max(0, x0):min(img.shape[1], x1)].astype(float)
    kk = k[ky0:ky1, kx0:kx1]
    bad = ~np.isfinite(sub)
    if mask is not None:
        bad |= (mask[max(0, y0):min(img.shape[0], y1), max(0, x0):min(img.shape[1], x1)] != 0)
    sub = np.where(bad, 0.0, sub)
    return float((sub * kk).sum() / (noise * np.sqrt((k ** 2).sum())))


def refine_length(img, x, y, pa_rad, sigma, L_max=220, thresh=2.0, mask=None):
    """Re-estimate trail length from the along-line profile (counters FRT
    over-integration). Projects a corridor of half-width 2*sigma onto the line
    axis and measures the contiguous extent above `thresh` (per-column SNR of
    the cross-track-summed profile, noise ~ sqrt(n_cross) for whitened input).

    Returns (L_est_px, x_c, y_c) -- length and flux-weighted center along the line.
    """
    r = int(L_max // 2)
    ct, st = np.cos(pa_rad), np.sin(pa_rad)
    tt = np.arange(-r, r + 1, dtype=float)
    wperp = np.arange(-int(np.ceil(2 * sigma)), int(np.ceil(2 * sigma)) + 1, dtype=float)
    prof = np.zeros(tt.size)
    nn = np.zeros(tt.size)
    h_img, w_img = img.shape
    for w in wperp:
        xs = np.round(x + tt * ct - w * st).astype(int)
        ys = np.round(y + tt * st + w * ct).astype(int)
        ok = (xs >= 0) & (xs < w_img) & (ys >= 0) & (ys < h_img)
        vals = np.zeros(tt.size)
        vals[ok] = img[ys[ok], xs[ok]]
        good = ok & np.isfinite(vals)
        if mask is not None:
            good[ok] &= (mask[ys[ok], xs[ok]] == 0)
        vals[~good] = 0.0
        prof += vals
        nn += good
    snr_prof = prof / np.sqrt(np.maximum(nn, 1))
    above = snr_prof > thresh
    if not above.any():
        return 0.0, x, y
    # contiguous run containing (or nearest) the center
    idx = np.where(above)[0]
    # split runs
    splits = np.where(np.diff(idx) > 3)[0]
    runs = np.split(idx, splits + 1)
    c = tt.size // 2
    run = min(runs, key=lambda rr: min(abs(rr - c)))
    t_lo, t_hi = tt[run[0]], tt[run[-1]]
    L_est = float(t_hi - t_lo + 1)
    wts = np.clip(snr_prof[run], 0, None)
    t_c = float(np.average(tt[run], weights=wts)) if wts.sum() > 0 else 0.0
    return L_est, x + t_c * ct, y + t_c * st
