#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ryder H. Strauss
"""ATLAS adapter: forced-photometry-service difference/reduced stamps ->
DiffExposure. The `_diff` stamps are ATLAS-native difference images with a valid
WCS, MAGZP, FWHM/SEEING; extreme-negative pixels are masked (subtraction bad
regions). MJD-OBS is shutter-start -> mid = MJD-OBS + exptime/2.
"""
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from ..varmap import whiten
from .base import DiffExposure


def _open2d(path):
    for h in fits.open(path):
        if h.data is not None and getattr(h.data, "ndim", 0) == 2:
            return np.asarray(h.data, float), h.header
    raise ValueError(f"no 2D HDU in {path}")


def _atlas_wcs(hdr):
    """ATLAS stamps carry a high-order TPV distortion (PV1_*/PV2_*) that this
    wcslib build rejects (and a conflicting SIP block). Over a 400 px stamp
    (~7') the TPV distortion is sub-pixel, so we fall back to a clean linear TAN
    WCS built from CRVAL/CRPIX/CD when the full parse fails."""
    try:
        return WCS(hdr)
    except Exception:
        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.crpix = [hdr["CRPIX1"], hdr["CRPIX2"]]
        w.wcs.crval = [hdr["CRVAL1"], hdr["CRVAL2"]]
        if "CD1_1" in hdr:
            w.wcs.cd = [[hdr["CD1_1"], hdr.get("CD1_2", 0.0)],
                        [hdr.get("CD2_1", 0.0), hdr["CD2_2"]]]
        else:
            w.wcs.cdelt = [hdr.get("CDELT1", 1.0), hdr.get("CDELT2", 1.0)]
        return w


def prepare_stamp(diff_path, cfg=None, idbase=None, bad_lo=-1e4):
    """Prepare one ATLAS difference stamp. Pixels below bad_lo (masked/edge
    subtraction residuals, ~ -3e4) are masked."""
    cfg = cfg or {}
    diff, hdr = _open2d(diff_path)
    wcs = _atlas_wcs(hdr)
    fwhm_px = hdr.get("FWHM", hdr.get("SEEING", 4.0))
    if fwhm_px is None or fwhm_px <= 0:
        fwhm_px = 4.0
    exptime = float(hdr.get("EXPTIME", 30.0))
    mjd = hdr.get("MJD-OBS", hdr.get("MJD"))
    mid = float(mjd) + exptime / 2 / 86400.0
    pixscale = np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600.0
    mask = ((diff < bad_lo) | ~np.isfinite(diff)).astype(np.uint8)
    white, var, bg = whiten(diff, mask, grid=cfg.get("preprocess", {}).get("var_grid_px", 100))
    diff = np.where(mask == 0, diff, np.nan)
    return DiffExposure(
        diff=diff, white=white, var=var, mask=mask, wcs=wcs, mid_mjd=mid,
        magzp=float(hdr.get("MAGZP", 22.0)), psf_sigma_px=float(fwhm_px) / 2.355,
        exptime_s=exptime, pixscale_arcsec=pixscale,
        obscode=cfg.get("obscode", str(hdr.get("OBS", "T08"))),
        band=str(hdr.get("FILTER", "o"))[:1] or "o",
        idbase=idbase or diff_path.split("/")[-1].split(".")[0], image_index=0,
        meta=dict(path=diff_path))
