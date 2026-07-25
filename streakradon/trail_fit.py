#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ryder H. Strauss
"""Trailed-source forward-model fit (Veres et al. 2012, PASP 124, 1197).

Model = uniform line source (half-length h) convolved with a Gaussian PSF (sigma):

    I(x,y) = A * 0.5*[erf((h+u)/(sqrt2 sig)) - erf((-h+u)/(sqrt2 sig))] * exp(-v^2/2sig^2) + bg

with (u,v) along/cross-track coords after rotating by theta about center (xc,yc).

Lifted from the validated ztf_streak/bin/trail_fit.py (a000001 ground truth) and
generalized for streak_radon:
  * dynamic cutout half-width hw (long trails no longer truncated at 110 px)
  * half-length upper bound tied to the cutout, not a fixed 60 px
  * conditional sigma-bound widening for undersampled PSFs (sigma0 < 0.7 px)

Center (xc,yc) is the trail GEOMETRIC CENTER = mid-exposure position (the
convention make_trailed_tracklets expects). Errors returned in the trail frame.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf

HW_MIN = 55  # minimum cutout half-width (px)


def _model(p, X, Y):
    A, xc, yc, theta, h, sig, bg = p
    ct, st = np.cos(theta), np.sin(theta)
    u_ = (X - xc) * ct + (Y - yc) * st
    v_ = -(X - xc) * st + (Y - yc) * ct
    s2 = np.sqrt(2.0) * sig
    return A * 0.5 * (erf((h + u_) / s2) - erf((-h + u_) / s2)) * np.exp(-0.5 * (v_ / sig) ** 2) + bg


def fit_trail(diff, mask, wcs, x0, y0, sigma0_px, magzp, theta0=None, h0=None, hw=None):
    """Fit one trail near (x0,y0) in a difference image. Returns dict or None.

    sigma0_px : effective PSF sigma (px), measured from field stars -- tightly
                bounded so a faint-trail fit cannot collapse the PSF width
                (bounds widened automatically when sigma0 < 0.7 px, undersampled).
    theta0,h0 : strong priors (pixel-frame PA rad, half-length px) from the FRT
                candidate; for faint trails these stabilize the fit.
    hw        : cutout half-width; default max(HW_MIN, ceil(1.8*h0)+15).
    """
    if hw is None:
        hw = HW_MIN if h0 is None else max(HW_MIN, int(np.ceil(1.8 * h0)) + 15)
    x0, y0 = int(round(x0)), int(round(y0))
    sub = diff[y0 - hw:y0 + hw + 1, x0 - hw:x0 + hw + 1].astype(float)
    ms = mask[y0 - hw:y0 + hw + 1, x0 - hw:x0 + hw + 1]
    if sub.shape != (2 * hw + 1, 2 * hw + 1):
        return None
    good = np.isfinite(sub) & (ms == 0)
    if good.sum() < 50:
        return None
    med = np.median(sub[good]); mad = 1.4826 * np.median(np.abs(sub[good] - med))
    if mad <= 0:
        return None
    ny, nx = sub.shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    # Angle init: FRT/matched-filter prior if given, else PCA on bright pixels.
    # The CENTER is initialized at the cutout center (= the detector peak,
    # localized at high integrated SNR) -- NOT the bright-pixel centroid, which
    # is noise-dominated for faint trails and sends the fit into a spurious
    # zero-amplitude basin.
    bright = good & (sub > med + 3 * mad)
    if theta0 is not None:
        th0 = theta0
    elif bright.sum() >= 4:
        wt = sub[bright] - med
        bx = np.average(xx[bright], weights=wt); by = np.average(yy[bright], weights=wt)
        cov = np.cov(np.vstack([xx[bright] - bx, yy[bright] - by]), aweights=wt)
        evl, evc = np.linalg.eigh(cov); axis = evc[:, np.argmax(evl)]
        th0 = np.arctan2(axis[1], axis[0])
    else:
        th0 = 0.0
    cx = cy = float(hw)
    A0 = max(sub[good & (sub > med + 2 * mad)].mean() - med if (good & (sub > med + 2 * mad)).any()
             else 5 * mad, 5 * mad)
    h_init = h0 if h0 is not None else 30.0
    # undersampled PSF (G96 at 1.52"/px can have sigma < 0.7 px): pixelization
    # makes the effective width uncertain -> widen the bounds
    slo, shi = (0.85, 1.25) if sigma0_px >= 0.7 else (0.70, 1.50)
    p0 = [A0, cx, cy, th0, h_init, sigma0_px, med]
    lb = [0,          cx - 6, cy - 6, th0 - 0.4, max(3.0, 0.5 * h_init),        slo * sigma0_px, med - 5 * mad]
    ub = [1000 * mad, cx + 6, cy + 6, th0 + 0.4, min(hw - 5.0, 2.5 * h_init),   shi * sigma0_px, med + 5 * mad]
    p0 = np.clip(p0, lb, ub)
    gi = good.ravel()
    Xr, Yr, Zr = xx.ravel()[gi], yy.ravel()[gi], sub.ravel()[gi]
    inv = 1.0 / mad
    try:
        sol = least_squares(lambda p: (_model(p, Xr, Yr) - Zr) * inv, p0,
                            bounds=(lb, ub), method='trf', max_nfev=5000)
    except Exception:
        return None
    A, xc, yc, theta, h, sig, bg = sol.x
    J = sol.jac; dof = max(len(Zr) - 7, 1); chi2r = 2 * sol.cost / dof
    try:
        C = np.linalg.inv(J.T @ J) * chi2r
    except np.linalg.LinAlgError:
        return None
    Cxy = C[1:3, 1:3]
    uvec = np.array([np.cos(theta), np.sin(theta)])
    vvec = np.array([-np.sin(theta), np.cos(theta)])
    pixscale = np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600.0
    sig_along = np.sqrt(max(uvec @ Cxy @ uvec, 0)) * pixscale
    sig_cross = np.sqrt(max(vvec @ Cxy @ vvec, 0)) * pixscale
    flux = A * (2 * h) * (sig * np.sqrt(2 * np.pi))
    sA = np.sqrt(abs(C[0, 0])); sh = np.sqrt(abs(C[4, 4])); ssig = np.sqrt(abs(C[5, 5]))
    rel = np.sqrt((sA / A) ** 2 + (sh / h) ** 2 + (ssig / sig) ** 2) if A > 0 else np.nan
    mag = magzp - 2.5 * np.log10(flux) if flux > 0 else np.nan
    sigmag = 1.0857 * rel
    gx, gy = xc + (x0 - hw), yc + (y0 - hw)
    mid = wcs.pixel_to_world(gx, gy)
    e1 = wcs.pixel_to_world(gx - h * np.cos(theta), gy - h * np.sin(theta))
    e2 = wcs.pixel_to_world(gx + h * np.cos(theta), gy + h * np.sin(theta))
    return dict(ra=float(mid.ra.deg), dec=float(mid.dec.deg),
                mag=float(mag), sigmag=float(sigmag),
                trail_len=float(e1.separation(e2).arcsec),
                trail_PA=float((e1.position_angle(e2).deg + 180.0) % 360.0),  # motion convention
                sig_across=float(sig_cross), sig_along=float(sig_along),
                psf_sigma_arcsec=float(sig * pixscale), chi2r=float(chi2r),
                peak_snr=float(A / mad), x=float(gx), y=float(gy),
                theta_px=float(theta), h_px=float(h), hw=int(hw))
