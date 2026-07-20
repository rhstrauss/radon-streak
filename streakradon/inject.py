#!/usr/bin/env python
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
