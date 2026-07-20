#!/usr/bin/env python
"""Effective-PSF sigma from field stars (no SEEING header key on CSS arch frames).

The arch images are high-pass filtered and possibly PSF-convolved, so the stars
carry the EFFECTIVE PSF -- exactly what the FRT data.psf and the Veres fit
sigma bound need. scipy-only (no photutils): local maxima above threshold,
isolation cut, second moments on background-subtracted stamps, sigma-clipped
median over the best stars.
"""
import numpy as np
from scipy.ndimage import maximum_filter


def measure_psf_sigma(img, mask=None, nstars=150, thresh_sigma=20.0,
                      stamp=7, isolation=12):
    """Return (sigma_px, n_used). sigma = sqrt of the mean of the two
    eigenvalues of the second-moment matrix (circular-Gaussian equivalent)."""
    good = np.isfinite(img)
    if mask is not None:
        good &= (mask == 0)
    vals = img[good]
    med = np.median(vals)
    mad = 1.4826 * np.median(np.abs(vals - med))
    det_thresh = med + thresh_sigma * mad
    mx = maximum_filter(np.where(good, img, -np.inf), size=isolation)
    peaks = (img == mx) & good & (img > det_thresh)
    # exclude borders
    peaks[:stamp + 1, :] = peaks[-stamp - 1:, :] = False
    peaks[:, :stamp + 1] = peaks[:, -stamp - 1:] = False
    ys, xs = np.where(peaks)
    if ys.size == 0:
        return np.nan, 0
    # brightest first, cap
    order = np.argsort(img[ys, xs])[::-1][:4 * nstars]
    ys, xs = ys[order], xs[order]
    sigs = []
    for y, x in zip(ys, xs):
        sub = img[y - stamp:y + stamp + 1, x - stamp:x + stamp + 1].astype(float) - med
        if mask is not None and (mask[y - stamp:y + stamp + 1, x - stamp:x + stamp + 1] != 0).any():
            continue
        if not np.isfinite(sub).all():
            continue
        sub = np.clip(sub, 0, None)
        tot = sub.sum()
        if tot <= 0:
            continue
        yy, xx = np.mgrid[-stamp:stamp + 1, -stamp:stamp + 1].astype(float)
        cx = (sub * xx).sum() / tot
        cy = (sub * yy).sum() / tot
        if abs(cx) > 2 or abs(cy) > 2:
            continue
        vxx = (sub * (xx - cx) ** 2).sum() / tot
        vyy = (sub * (yy - cy) ** 2).sum() / tot
        vxy = (sub * (xx - cx) * (yy - cy)).sum() / tot
        tr = 0.5 * (vxx + vyy)
        if tr <= 0.05 or tr > (stamp / 1.5) ** 2:
            continue
        # elongated sources (galaxies, doubles, trails!) rejected
        disc = np.sqrt(max((0.5 * (vxx - vyy)) ** 2 + vxy ** 2, 0))
        e1, e2 = tr + disc, tr - disc
        if e2 <= 0 or np.sqrt(e1 / e2) > 1.6:
            continue
        sigs.append(np.sqrt(tr))
        if len(sigs) >= nstars:
            break
    if len(sigs) < 10:
        return np.nan, len(sigs)
    sigs = np.array(sigs)
    for _ in range(3):  # sigma clip
        m, s = np.median(sigs), 1.4826 * np.median(np.abs(sigs - np.median(sigs)))
        keep = np.abs(sigs - m) < 3 * max(s, 1e-3)
        sigs = sigs[keep]
    return float(np.median(sigs)), int(sigs.size)


def measure_psf_stamp(img, mask=None, nstars=120, thresh_sigma=25.0, hw=9,
                      upsample=4, lo=800.0, hi=20000.0):
    """Empirical PSF kernel: sub-pixel-registered, background-subtracted median
    stack of isolated moderate-brightness stars, normalized to sum=1. Captures
    the real (compact, undersampled, all-positive, possibly non-Gaussian) source
    profile so injected trails match real sources exactly. Returns the (2hw+1,
    2hw+1) kernel (or None)."""
    from scipy.ndimage import maximum_filter, shift as ndshift
    good = np.isfinite(img)
    if mask is not None:
        good &= (mask == 0)
    med = np.median(img[good]); mad = 1.4826 * np.median(np.abs(img[good] - med))
    mx = maximum_filter(np.where(good, img, -np.inf), size=2 * hw + 5)
    peaks = (img == mx) & good & (img > med + thresh_sigma * mad) & (img < med + hi)
    peaks[:hw + 2, :] = peaks[-hw - 2:, :] = False
    peaks[:, :hw + 2] = peaks[:, -hw - 2:] = False
    ys, xs = np.where(peaks)
    order = np.argsort(img[ys, xs])[::-1]
    ys, xs = ys[order], xs[order]
    yy, xx = np.mgrid[-hw:hw + 1, -hw:hw + 1].astype(float)
    stacks = []
    for y, x in zip(ys, xs):
        st = img[y - hw:y + hw + 1, x - hw:x + hw + 1].astype(float)
        if st.shape != (2 * hw + 1, 2 * hw + 1) or not np.isfinite(st).all():
            continue
        if mask is not None and (mask[y - hw:y + hw + 1, x - hw:x + hw + 1] != 0).any():
            continue
        st = st - med
        pk = st.max()
        if pk < lo or pk > hi:
            continue
        w = np.clip(st, 0, None); tot = w.sum()
        if tot <= 0:
            continue
        cx = (w * xx).sum() / tot; cy = (w * yy).sum() / tot
        if abs(cx) > 2 or abs(cy) > 2:
            continue
        st = ndshift(st, (-cy, -cx), order=1, mode="constant", cval=0.0)  # recenter
        s = st.sum()
        if s > 0:
            stacks.append(st / s)
        if len(stacks) >= nstars:
            break
    if len(stacks) < 10:
        return None
    ker = np.median(np.array(stacks), axis=0)
    ker = np.clip(ker, 0, None)          # enforce non-negative (no moat here anyway)
    ker /= ker.sum()
    return ker
