#!/usr/bin/env python
"""Rough trail measurement from the FRT+MF refine, with APPROXIMATE errors --
the survey-scale fast path (no per-candidate Veres fit).

Rationale (Phase 1 optimization, 2026-07). Profiling showed the Veres
line(x)Gaussian LSQ costs ~917 ms/candidate, ~35% of total pipeline runtime at
~167 candidates/exposure. But heliolinc's tracklet gates only need trail_len
within `siglenscale` (default 0.5, i.e. 50%) and trail_PA within
`sigpascale/trail_len` -- far looser than the Veres fit delivers. Measured
accuracy of the refine path alone, against injected truth:

    trail_PA   0.47 deg median error
    trail_len  1.9 % median error   (with refine_candidate(polish_length=True))

Both are comfortably inside the linker gates, so the rough measurement is
sufficient to LINK. The Veres fit is therefore demoted to an optional
POST-LINKING refinement, run on the few linked detections rather than every
candidate, where its precise astrometry + covariance actually matter (ADES
submission, orbit fitting).

Error model (approximate, deliberately conservative):
    sig_across ~ psf_sigma / SNR                       (cross-track centroid)
    sig_along  ~ hypot(psf_sigma, L/sqrt(12)) / SNR    (along-track: endpoint
                 localisation of a ~uniform line dominates for long trails)
plus an isotropic systematic floor. These are ESTIMATES, not fit covariances --
downstream consumers that need real uncertainties must use the Veres refinement.
"""
import numpy as np

SYS_FLOOR_ARCSEC = 0.06     # Gaia/WCS systematic floor, added in quadrature


def _aperture_flux(diff, mask, x, y, L, pa_rad, sigma):
    """Sum diff flux in the trail capsule (cheap aperture photometry)."""
    hw = max(2.5 * sigma, 2.0)
    ct, st = np.cos(pa_rad), np.sin(pa_rad)
    x1, y1 = x - 0.5 * L * ct, y - 0.5 * L * st
    x2, y2 = x + 0.5 * L * ct, y + 0.5 * L * st
    pad = int(np.ceil(hw)) + 2
    H, W = diff.shape
    y0 = max(0, int(min(y1, y2)) - pad); yb = min(H, int(max(y1, y2)) + pad + 1)
    x0 = max(0, int(min(x1, x2)) - pad); xb = min(W, int(max(x1, x2)) + pad + 1)
    if yb <= y0 or xb <= x0:
        return 0.0, 0
    yy, xx = np.mgrid[y0:yb, x0:xb]
    dx, dy = x2 - x1, y2 - y1
    seg2 = dx * dx + dy * dy
    t = np.clip(((xx - x1) * dx + (yy - y1) * dy) / seg2, 0, 1) if seg2 > 0 else 0
    d = np.hypot(xx - (x1 + t * dx), yy - (y1 + t * dy))
    inside = d <= hw
    sub = diff[y0:yb, x0:xb]
    good = inside & np.isfinite(sub)
    if mask is not None:
        good &= (mask[y0:yb, x0:xb] == 0)
    if not good.any():
        return 0.0, 0
    return float(sub[good].sum()), int(good.sum())


def centroid_trail(diff, mask, x, y, L, pa_rad, sigma):
    """Flux-weighted centroid of the trail within its capsule.

    The MF grid search localises the center only to the slide-grid spacing
    (2*slide_frac*L/(n_slide-1)), and length/center are coupled, which left a
    ~5.5" median center error vs truth -- too coarse for heliolinc's maxGCR
    (3"). Once PA and L are known, a direct flux-weighted centroid over the
    corridor pixels is both far more accurate and essentially free (one pass
    over a thin bbox). Returns (x, y) unchanged if the corridor has no flux.
    """
    hw = max(3.0 * sigma, 2.5)
    ct, st = np.cos(pa_rad), np.sin(pa_rad)
    x1, y1 = x - 0.5 * L * ct, y - 0.5 * L * st
    x2, y2 = x + 0.5 * L * ct, y + 0.5 * L * st
    pad = int(np.ceil(hw)) + 3
    H, W = diff.shape
    y0 = max(0, int(min(y1, y2)) - pad); yb = min(H, int(max(y1, y2)) + pad + 1)
    x0 = max(0, int(min(x1, x2)) - pad); xb = min(W, int(max(x1, x2)) + pad + 1)
    if yb <= y0 or xb <= x0:
        return x, y
    yy, xx = np.mgrid[y0:yb, x0:xb]
    dx, dy = x2 - x1, y2 - y1
    seg2 = dx * dx + dy * dy
    if seg2 <= 0:
        return x, y
    t = ((xx - x1) * dx + (yy - y1) * dy) / seg2
    perp = np.hypot(xx - (x1 + np.clip(t, 0, 1) * dx),
                    yy - (y1 + np.clip(t, 0, 1) * dy))
    sub = diff[y0:yb, x0:xb]
    good = (perp <= hw) & np.isfinite(sub) & (t >= -0.15) & (t <= 1.15)
    if mask is not None:
        good &= (mask[y0:yb, x0:xb] == 0)
    if not good.any():
        return x, y
    w = np.where(good, np.maximum(sub, 0.0), 0.0)   # positive flux only
    tot = w.sum()
    if tot <= 0:
        return x, y
    return float((w * xx).sum() / tot), float((w * yy).sum() / tot)


def rough_fit(diff, mask, wcs, ref, sigma_px, magzp, pixscale_arcsec=None,
              centroid=True):
    """Build a trail_fit-compatible dict from a refine_candidate result.

    ref : dict from mf_snr.refine_candidate (snr, L, pa_rad, x, y)
    Returns None if the measurement is unusable.
    """
    x, y = float(ref["x"]), float(ref["y"])
    L_px = float(ref["L"])
    theta = float(ref["pa_rad"])
    snr = max(float(ref["snr"]), 1e-3)
    if pixscale_arcsec is None:
        pixscale_arcsec = float(np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600.0)
    h = 0.5 * L_px
    ct, st = np.cos(theta), np.sin(theta)
    if centroid:
        x, y = centroid_trail(diff, mask, x, y, L_px, theta, sigma_px)

    # sky position + trail geometry, same convention as trail_fit.fit_trail
    mid = wcs.pixel_to_world(x, y)
    e1 = wcs.pixel_to_world(x - h * ct, y - h * st)
    e2 = wcs.pixel_to_world(x + h * ct, y + h * st)
    trail_len = float(e1.separation(e2).arcsec)
    trail_PA = float((e1.position_angle(e2).deg + 180.0) % 360.0)   # motion convention

    # approximate anisotropic errors
    psf_arc = sigma_px * pixscale_arcsec
    sig_cross = np.hypot(psf_arc / snr, SYS_FLOOR_ARCSEC)
    sig_along = np.hypot(np.hypot(psf_arc, trail_len / np.sqrt(12.0)) / snr,
                         SYS_FLOOR_ARCSEC)

    # photometry: aperture sum along the trail
    flux, npix = _aperture_flux(diff, mask, x, y, L_px, theta, sigma_px)
    if flux > 0:
        mag = float(magzp - 2.5 * np.log10(flux))
        sigmag = float(1.0857 / snr)
    else:
        mag = float("nan"); sigmag = float("nan")

    return dict(ra=float(mid.ra.deg), dec=float(mid.dec.deg),
                mag=mag, sigmag=sigmag,
                trail_len=trail_len, trail_PA=trail_PA,
                sig_across=float(sig_cross), sig_along=float(sig_along),
                psf_sigma_arcsec=float(psf_arc),
                chi2r=1.0,              # not fitted; sentinel so RB chi2 gate passes
                peak_snr=float(snr),
                x=x, y=y, theta_px=theta, h_px=h, hw=int(max(42, 1.25 * h + 22)),
                rough=True)
