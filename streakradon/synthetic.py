#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ryder H. Strauss
"""Self-contained synthetic trailed-mover sequence -- NO external data required.

Builds a short, rapid-cadence sequence of difference images (pure-noise
background = perfectly-subtracted static field, the ideal a real diff
approximates) with one linearly-moving trailed source injected across all
epochs. `n_residuals>0` adds repeating static point-source residuals to exercise
the vetting/repetition filter (off by default so the demo is deterministic;
clutter-robustness is validated on real survey diffs, see REPORT.md). The same
Veres model the fitter uses is injected (model identity guaranteed), so this
doubles as a smoke test of the whole detect -> refine -> fit -> vet -> hldet
chain and (optionally) heliolinx make_trailed_tracklets linking.

Used by `streak-radon demo`.
"""
import numpy as np
from astropy.wcs import WCS

from .adapters.base import DiffExposure
from .inject import inject_trail


def make_wcs(shape, crval=(180.0, 0.0), pixscale_arcsec=1.0):
    """Simple N-up/E-left TAN WCS centered on the frame."""
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = [shape[1] / 2.0 + 0.5, shape[0] / 2.0 + 0.5]
    w.wcs.crval = list(crval)
    d = pixscale_arcsec / 3600.0
    w.wcs.cd = [[-d, 0.0], [0.0, d]]
    return w


def build_sequence(n_epochs=4, shape=(1000, 1000), pixscale_arcsec=1.0,
                   cadence_s=60.0, exptime_s=30.0, t0_mjd=60431.15,
                   rate_deg_day=40.0, pa_px_deg=35.0, mag=18.5, magzp=26.0,
                   psf_sigma_px=1.4, noise=1.0, obscode="500", band="w",
                   n_residuals=0, seed=7):
    """Return (exposures, truth). `truth` holds the mover's per-epoch pixel
    positions, length, and pixel-frame PA. Motion is defined in PIXEL space so
    the injected trail geometry and the between-epoch displacement are exactly
    self-consistent through the (shared) WCS -- what the linker's pair gates need.
    """
    rng = np.random.default_rng(seed)
    wcs = make_wcs(shape, pixscale_arcsec=pixscale_arcsec)
    pa = np.radians(pa_px_deg)
    rate_px_day = rate_deg_day * 3600.0 / pixscale_arcsec
    L_px = rate_px_day * exptime_s / 86400.0
    mjds = [t0_mjd + i * cadence_s / 86400.0 for i in range(n_epochs)]

    # center the whole track on the frame
    span_day = mjds[-1] - mjds[0]
    travel = rate_px_day * span_day
    x0 = shape[1] / 2.0 - 0.5 * travel * np.cos(pa)
    y0 = shape[0] / 2.0 - 0.5 * travel * np.sin(pa)

    # fixed static residual sites (repeat at the same pixel -> same sky -> repetition-filtered)
    res_sites = [(rng.uniform(80, shape[1] - 80), rng.uniform(80, shape[0] - 80))
                 for _ in range(n_residuals)]
    res_mag = [rng.uniform(mag - 1.0, mag + 1.0) for _ in range(n_residuals)]

    exposures, positions = [], []
    for i, mjd in enumerate(mjds):
        dt = mjd - mjds[0]
        x = x0 + rate_px_day * dt * np.cos(pa)
        y = y0 + rate_px_day * dt * np.sin(pa)
        positions.append((x, y))
        diff = rng.normal(0.0, noise, shape)
        inject_trail(diff, x, y, pa, L_px, mag, magzp, psf_sigma_px)
        # static residuals: point-like (very short "trail") at fixed sites
        for (rx, ry), rm in zip(res_sites, res_mag):
            inject_trail(diff, rx, ry, 0.0, 1.0, rm, magzp, psf_sigma_px)
        mask = np.zeros(shape, dtype=np.uint8)
        from .varmap import whiten
        white, var, _ = whiten(diff, mask, grid=128)
        exposures.append(DiffExposure(
            diff=diff.copy(), white=white, var=var, mask=mask, wcs=wcs,
            mid_mjd=float(mjd), magzp=magzp, psf_sigma_px=psf_sigma_px,
            exptime_s=exptime_s, pixscale_arcsec=pixscale_arcsec,
            obscode=obscode, band=band, idbase="synth", image_index=i,
            meta=dict(synthetic=True)))
    truth = dict(positions=positions, L_px=float(L_px), pa_px_deg=float(pa_px_deg),
                 rate_deg_day=float(rate_deg_day), mag=float(mag),
                 trail_len_arcsec=float(L_px * pixscale_arcsec), mjds=mjds,
                 res_sites=res_sites)
    return exposures, truth
