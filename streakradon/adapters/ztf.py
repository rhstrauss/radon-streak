#!/usr/bin/env python
"""ZTF adapter: IRSA scimrefdiffimg (+ mskimg, + sci header for timing) ->
DiffExposure. The diff is already reference-subtracted; we only whiten.

Timing: exact shutter midpoint (SHUTOPEN+SHUTCLSD)/2 from the SCIENCE header
(validated convention -- OBSJD is rounded ~0.7 s early; at 134"/min that is
~1.5" along-track; find_orb residual 0.89" -> 0.27" with the midpoint).
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


def mid_mjd_from_sci(sci_hdr):
    if "SHUTOPEN" in sci_hdr and "SHUTCLSD" in sci_hdr:
        from astropy.time import Time
        t0 = Time(sci_hdr["SHUTOPEN"], format="isot", scale="utc")
        t1 = Time(sci_hdr["SHUTCLSD"], format="isot", scale="utc")
        return 0.5 * (t0.mjd + t1.mjd)
    # fallback: OBSJD (shutter-open, ~0.7 s rounding) + half exposure
    return sci_hdr["OBSJD"] - 2400000.5 + sci_hdr.get("EXPTIME", 30.0) / 2 / 86400.0


def prepare_exposure(diff_path, mask_path, sci_path=None, cfg=None, idbase=None):
    cfg = cfg or {}
    diff, hdr = _open2d(diff_path)
    mask, _ = _open2d(mask_path)
    wcs = WCS(hdr)
    m = (mask != 0).astype(np.uint8)
    white, var, bg = whiten(diff, m, grid=cfg.get("preprocess", {}).get("var_grid_px", 128))
    seeing_px = hdr.get("SEEING", 2.0)
    if sci_path:
        _, sci_hdr = _open2d(sci_path)
        mid = mid_mjd_from_sci(sci_hdr)
        exptime = float(sci_hdr.get("EXPTIME", 30.0))
    else:
        mid = hdr["OBSMJD"] + hdr.get("EXPTIME", 30.0) / 2 / 86400.0 \
            if "OBSMJD" in hdr else hdr["OBSJD"] - 2400000.5 + 15.0 / 86400.0
        exptime = float(hdr.get("EXPTIME", 30.0))
    pixscale = np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600.0
    diff = np.where(m == 0, diff, np.nan)
    return DiffExposure(
        diff=diff, white=white, var=var, mask=m, wcs=wcs, mid_mjd=float(mid),
        magzp=float(hdr.get("MAGZP", 26.0)), psf_sigma_px=float(seeing_px) / 2.355,
        exptime_s=exptime, pixscale_arcsec=pixscale,
        obscode=cfg.get("obscode", "I41"),
        band=str(hdr.get("FILTER", "g")).replace("ZTF_", "").replace("ztf", "")[:1] or "g",
        idbase=idbase or diff_path.split("/")[-1].split(".")[0], image_index=0,
        meta=dict(path=diff_path))
