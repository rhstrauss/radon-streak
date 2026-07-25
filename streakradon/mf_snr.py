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


def mf_snr_kernel(img, x, y, k, knorm, noise=1.0, mask=None):
    """mf_snr_at with a PRE-BUILT kernel -- the kernel depends only on (L, pa),
    not on the center, so it is hoisted out of the center/slide search loop.
    Building trail_kernel (mgrid + erf + exp over a (2r+1)^2 stamp) dominated
    refine_candidate: it was rebuilt for every (L, pa, slide) triple."""
    r = k.shape[0] // 2
    xi, yi = int(round(x)), int(round(y))
    y0, y1 = yi - r, yi + r + 1
    x0, x1 = xi - r, xi + r + 1
    ky0 = max(0, -y0); kx0 = max(0, -x0)
    ky1 = k.shape[0] - max(0, y1 - img.shape[0])
    kx1 = k.shape[1] - max(0, x1 - img.shape[1])
    if ky1 <= ky0 or kx1 <= kx0:
        return 0.0
    sub = img[max(0, y0):min(img.shape[0], y1), max(0, x0):min(img.shape[1], x1)]
    kk = k[ky0:ky1, kx0:kx1]
    bad = ~np.isfinite(sub)
    if mask is not None:
        bad |= (mask[max(0, y0):min(img.shape[0], y1), max(0, x0):min(img.shape[1], x1)] != 0)
    sub = np.where(bad, 0.0, sub)
    return float((sub * kk).sum() / (noise * knorm))


LENGTH_LADDER = (6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192)


def refine_candidate(img, x, y, pa_rad, sigma, mask=None,
                     lengths=LENGTH_LADDER, dpa_deg=8.0, dpa_step_deg=2.0,
                     slide_frac=0.35, n_slide=7, polish_length=True):
    """Refine an FRT candidate by maximizing the integrated MF SNR over a small
    grid of (PA, length, center-slide-along-line).

    The FRT's PA is quantized (a few deg off decorrelates a long thin template:
    at L=66 px, sigma=0.9 px, a 5 deg error displaces the endpoints ~3 sigma
    cross-track) and its length over/under-integrates on the power-of-2 folding
    ladder -- so both must be re-fit against the statistic that matters.

    polish_length : refine L off the discrete ladder by a local search around the
        best rung. The ladder quantizes length (measured median error 11.7% vs
        truth); this recovers a continuous estimate, which matters because the
        fast (no-Veres) path reports trail_len straight from here.

    Returns dict(snr, L, pa_rad, x, y).
    """
    pas = pa_rad + np.radians(np.arange(-dpa_deg, dpa_deg + 0.01, dpa_step_deg))
    best = dict(snr=-np.inf, L=float(lengths[0]), pa_rad=pa_rad, x=x, y=y)

    def scan(L, pa_list, centers=None):
        """Evaluate one length over PAs (+ slides), reusing each (L,pa) kernel."""
        nonlocal best
        slides = np.linspace(-slide_frac * L, slide_frac * L, n_slide)
        for pa in pa_list:
            k = trail_kernel(L, pa, sigma)          # built ONCE per (L, pa)
            knorm = np.sqrt((k ** 2).sum())
            if knorm <= 0:
                continue
            ct, st = np.cos(pa), np.sin(pa)
            for s in (slides if centers is None else centers):
                xs, ys = x + s * ct, y + s * st
                snr = mf_snr_kernel(img, xs, ys, k, knorm, noise=1.0, mask=mask)
                if snr > best['snr']:
                    best = dict(snr=float(snr), L=float(L), pa_rad=float(pa % np.pi),
                                x=float(xs), y=float(ys))

    for L in lengths:
        scan(float(L), pas)

    # length polish: local search around the winning rung (ladder is geometric,
    # so probe fractional steps between the neighbouring rungs)
    if polish_length:
        L0 = best['L']
        for frac in (0.80, 0.90, 1.10, 1.25):
            scan(max(4.0, L0 * frac), [best['pa_rad']])

    # center polish at the best (L, pa). The along-track slide grid is COARSE --
    # spacing 2*slide_frac*L/(n_slide-1), i.e. ~7.7 px (11.7") at L=66 px -- which
    # left the reported center quantized to +-5.8" and showed up as a 2-5" offset
    # against the Veres fit on real data. Polish along-track at ~1 px, then
    # perpendicular, so the fast path's astrometry is not grid-limited.
    k = trail_kernel(best['L'], best['pa_rad'], sigma)
    knorm = np.sqrt((k ** 2).sum())
    ct, st = np.cos(best['pa_rad']), np.sin(best['pa_rad'])
    step = max(2.0 * slide_frac * best['L'] / max(n_slide - 1, 1), 2.0)
    while step > 0.5:                      # bisection-style along-track refine
        improved = False
        for s in (-step, step):
            xs, ys = best['x'] + s * ct, best['y'] + s * st
            snr = mf_snr_kernel(img, xs, ys, k, knorm, mask=mask)
            if snr > best['snr']:
                best.update(snr=float(snr), x=float(xs), y=float(ys))
                improved = True
                break
        if not improved:
            step *= 0.5
    for w in (-2, -1, -0.5, 0.5, 1, 2):    # perpendicular polish
        xs, ys = best['x'] - w * st, best['y'] + w * ct
        snr = mf_snr_kernel(img, xs, ys, k, knorm, mask=mask)
        if snr > best['snr']:
            best.update(snr=float(snr), x=float(xs), y=float(ys))
    return best
