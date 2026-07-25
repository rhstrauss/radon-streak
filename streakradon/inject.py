#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ryder H. Strauss
"""Synthetic trail injection -- one injector, three consumers (efficiency
grids, threshold/FP calibration, end-to-end linking test).

Evaluates the SAME Veres model the fitter uses (imported from trail_fit, model
identity guaranteed). Amplitude from mag/MAGZP: flux = 10^(-0.4(mag-zp)),
A = flux / (2h * sigma * sqrt(2pi)). Optional Poisson deviates on added counts
and a sigma-perturbation mode to bound model-mismatch optimism (model-matched
injection yields upper-bound efficiency; state that caveat in reports).
"""
import numpy as np

from .trail_fit import _model


def inject_trail(img, x0, y0, pa_rad, L_px, mag, magzp, psf_sigma_px,
                 gain=None, rng=None, sigma_perturb=0.0):
    """Add a synthetic trail to img IN PLACE. Returns truth dict."""
    sigma = psf_sigma_px * (1.0 + sigma_perturb)
    h = L_px / 2.0
    flux = 10.0 ** (-0.4 * (mag - magzp))
    A = flux / (2 * h * sigma * np.sqrt(2 * np.pi))
    r = int(np.ceil(h + 5 * sigma)) + 2
    x0i, y0i = int(round(x0)), int(round(y0))
    y_lo, y_hi = max(0, y0i - r), min(img.shape[0], y0i + r + 1)
    x_lo, x_hi = max(0, x0i - r), min(img.shape[1], x0i + r + 1)
    if y_hi <= y_lo or x_hi <= x_lo:
        # trail center is off-frame -> nothing to add
        return dict(x=float(x0), y=float(y0), pa_rad=float(pa_rad), L_px=float(L_px),
                    mag=float(mag), flux=float(10.0 ** (-0.4 * (mag - magzp))),
                    A=float(A), sigma_px=float(sigma), offframe=True)
    sl = (slice(y_lo, y_hi), slice(x_lo, x_hi))
    yy, xx = np.mgrid[sl[0], sl[1]].astype(float)
    add = _model([A, x0, y0, pa_rad, h, sigma, 0.0], xx, yy)
    if gain is not None:
        rng = rng or np.random.default_rng()
        e = np.clip(add * gain, 0, None)
        add = rng.poisson(e) / gain
    img[sl] += add
    return dict(x=float(x0), y=float(y0), pa_rad=float(pa_rad), L_px=float(L_px),
                mag=float(mag), flux=float(flux), A=float(A), sigma_px=float(sigma))


def inject_trail_empirical(img, x0, y0, pa_rad, L_px, mag, magzp, psf_stamp,
                           gain=None, rng=None, oversample=5):
    """Add a synthetic trail built by convolving a uniform line source (length
    L_px, angle pa_rad, total flux from mag/magzp) with the EMPIRICAL PSF stamp
    (sum-normalized kernel from psf.measure_psf_stamp) -- reproduces the real
    (compact, undersampled, non-Gaussian) source profile instead of an analytic
    Gaussian. Returns a truth dict."""
    from scipy.signal import fftconvolve
    flux = 10.0 ** (-0.4 * (mag - magzp))
    kh = psf_stamp.shape[0] // 2
    h = L_px / 2.0
    pad = int(np.ceil(h)) + kh + 3
    n = 2 * pad + 1
    line = np.zeros((n, n))
    # lay the line's flux down as oversampled points about the canvas centre,
    # with the sub-pixel offset of (x0,y0) folded in
    x0i, y0i = int(round(x0)), int(round(y0))
    fx, fy = x0 - x0i, y0 - y0i
    ct, st = np.cos(pa_rad), np.sin(pa_rad)
    npts = max(int(np.ceil(L_px * oversample)), 2)
    tt = np.linspace(-h, h, npts)
    wper = flux / npts
    for t in tt:
        px = pad + fx + t * ct
        py = pad + fy + t * st
        ix, iy = int(np.floor(px)), int(np.floor(py))
        dx, dy = px - ix, py - iy
        if 0 <= iy < n - 1 and 0 <= ix < n - 1:      # bilinear splat
            line[iy, ix] += wper * (1 - dx) * (1 - dy)
            line[iy, ix + 1] += wper * dx * (1 - dy)
            line[iy + 1, ix] += wper * (1 - dx) * dy
            line[iy + 1, ix + 1] += wper * dx * dy
    stamp = fftconvolve(line, psf_stamp, mode="same")
    if gain is not None:
        rng = rng or np.random.default_rng()
        stamp = rng.poisson(np.clip(stamp * gain, 0, None)) / gain
    y0c, y1c = y0i - pad, y0i + pad + 1
    x0c, x1c = x0i - pad, x0i + pad + 1
    ys, ye = max(0, y0c), min(img.shape[0], y1c)
    xs, xe = max(0, x0c), min(img.shape[1], x1c)
    if ye > ys and xe > xs:
        img[ys:ye, xs:xe] += stamp[ys - y0c:ye - y0c, xs - x0c:xe - x0c]
    return dict(x=float(x0), y=float(y0), pa_rad=float(pa_rad), L_px=float(L_px),
                mag=float(mag), flux=float(flux), sigma_px=None, empirical=True)


def sequence_positions(x0, y0, rate_px_per_day, pa_rad, mjds):
    """Linear-motion positions of a mover at each exposure epoch (relative to
    the first). Returns list of (x, y)."""
    t0 = mjds[0]
    return [(x0 + rate_px_per_day * (m - t0) * np.cos(pa_rad),
             y0 + rate_px_per_day * (m - t0) * np.sin(pa_rad)) for m in mjds]


def random_positions(shape, mask, n, margin=150, min_sep=200, rng=None):
    """Unmasked random injection sites with mutual separation >= min_sep."""
    rng = rng or np.random.default_rng()
    out = []
    tries = 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        x = rng.uniform(margin, shape[1] - margin)
        y = rng.uniform(margin, shape[0] - margin)
        if mask is not None and mask[int(y), int(x)] != 0:
            continue
        if any(np.hypot(x - a, y - b) < min_sep for a, b in out):
            continue
        out.append((x, y))
    return out
