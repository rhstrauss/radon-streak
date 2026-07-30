#!/usr/bin/env python
"""ZTF adapter: IRSA scimrefdiffimg (+ mskimg) -> DiffExposure.

The IPAC diff is already reference-subtracted, so this adapter only builds the
bad-pixel mask and whitens. Two things here are ZTF-specific, and the first
version of this file got both wrong:

TIMING -- the DIFF header itself carries SHUTOPEN/SHUTCLSD, so the validated
    shutter-midpoint epoch (SHUTOPEN+SHUTCLSD)/2 needs NO sciimg download.
    That matters at survey scale: sciimg is 37.9 MB against the 9.4 MB diff, so
    taking the epoch from the diff cuts the per-quadrant download by ~58%.
    OBSJD/OBSMJD are shutter-OPEN, rounded ~0.7 s early; at ~134"/min for a fast
    NEO that is ~1.5" along-track (find_orb residual 0.89" -> 0.27" with the
    midpoint). `sci_path` is still accepted, but is now only a fallback.

MASK BITS -- ZTF's mskimg is NOT a bad-pixel mask. Its own header documents the
    bits, and TWO OF THEM MARK REAL ASTROPHYSICAL SOURCES:

        bit  0  AIRCRAFT/SATELLITE TRACK
        bit  1  CONTAINS SEXTRACTOR DETECTION           <-- a SOURCE, not a defect
        bit  2  LOW RESPONSIVITY
        bit  3  HIGH RESPONSIVITY
        bit  4  NOISY
        bit  5  GHOST FROM BRIGHT SOURCE
        bit  6  POSSIBLE GHOST FROM CHARGE SPILLAGE
        bit  7  PIXEL SPIKE (POSSIBLE RAD HIT)
        bit  8  SATURATED
        bit  9  DEAD (UNRESPONSIVE)
        bit 10  NAN
        bit 11  CONTAINS PSF-EXTRACTED SOURCE POSITION  <-- also a SOURCE
        bit 12  HALO FROM BRIGHT SOURCE

    The original `mask != 0` therefore NaN'd every pixel where SExtractor found
    something -- it blanked real detections, and for a trailed NEO bright enough
    to be catalogued in the science frame it blanked THE TRAIL ITSELF.
    Measured bit population on a real 2024-09-01 quadrant: bit 1 = 0.336%,
    bit 11 = 0.015%, bit 12 = 1.651% (the only large genuine defect class).

    So the default bad-pixel set deliberately EXCLUDES bits 1 and 11.
    Bit 0 IS masked: IPAC's aircraft/satellite tracks are the dominant streak
    contaminant for a streak detector, and its track finder flags only long
    quadrant-crossing trails, far longer than the ~5-200" NEO trails we want.
    Override with cfg['preprocess']['mask_bits'].
"""
import os

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from ..varmap import whiten
from .base import DiffExposure

# Bits meaning "this pixel is unusable", per the mskimg header above.
# NOT 1 (sextractor detection) and NOT 11 (psf-extracted source position):
# those are source-presence flags, and masking them destroys our signal.
DEFAULT_BAD_BITS = (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12)

# Source-presence bits, named so callers can inspect/QA them explicitly.
SOURCE_BITS = (1, 11)


def bad_pixel_mask(mask, bits=DEFAULT_BAD_BITS):
    """uint8 mask, 1 = unusable pixel. Bit-selective, never `mask != 0`."""
    mi = np.asarray(mask).astype(np.int32)
    m = np.zeros(mi.shape, np.uint8)
    for b in bits:
        m |= ((mi >> int(b)) & 1).astype(np.uint8)
    return m


def _open2d(path):
    for h in fits.open(path):
        if h.data is not None and getattr(h.data, "ndim", 0) == 2:
            return np.asarray(h.data, float), h.header
    raise ValueError(f"no 2D HDU in {path}")


def mid_mjd_from_hdr(hdr):
    """Shutter-midpoint MJD, from a diff header (preferred) or a science one."""
    if "SHUTOPEN" in hdr and "SHUTCLSD" in hdr:
        from astropy.time import Time
        t0 = Time(str(hdr["SHUTOPEN"]).strip(), format="isot", scale="utc")
        t1 = Time(str(hdr["SHUTCLSD"]).strip(), format="isot", scale="utc")
        return 0.5 * (t0.mjd + t1.mjd)
    # Fallbacks: OBSMJD/OBSJD are shutter-OPEN, rounded ~0.7 s early.
    half = float(hdr.get("EXPTIME", 30.0)) / 2.0 / 86400.0
    if "OBSMJD" in hdr:
        return float(hdr["OBSMJD"]) + half
    return float(hdr["OBSJD"]) - 2400000.5 + half


# Back-compat alias: the old name took a science header specifically.
mid_mjd_from_sci = mid_mjd_from_hdr


def prepare_exposure(diff_path, mask_path, sci_path=None, cfg=None, idbase=None,
                     image_index=0):
    cfg = cfg or {}
    pre = cfg.get("preprocess", {}) or {}
    diff, hdr = _open2d(diff_path)
    mask, _ = _open2d(mask_path)
    wcs = WCS(hdr)

    bits = tuple(pre.get("mask_bits", DEFAULT_BAD_BITS))
    m = bad_pixel_mask(mask, bits)

    # FP CALIBRATION PATH: negating the diff turns every real (positive) source
    # negative, so a detector that hunts positive streaks sees a SOURCE-FREE but
    # otherwise fully realistic image -- same noise, same subtraction residuals,
    # same detector artifacts. Every surviving detection is then a false positive
    # by construction, which is how G96's mf_snr_min=11.0 was frozen. Enable with
    # STREAKRADON_NEGATE=1 or preprocess.negate. NEVER set this for science runs.
    if pre.get("negate", False) or os.environ.get("STREAKRADON_NEGATE"):
        diff = -diff

    white, var, bg = whiten(diff, m, grid=pre.get("var_grid_px", 128))

    pixscale = np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600.0

    # ZTF's SEEING keyword is FWHM in ARCSEC. The old code used it as pixels,
    # which was harmless only because ZTF's pixscale happens to be ~1.0"/px.
    seeing_arcsec = float(hdr.get("SEEING", 2.0))
    psf_sigma_px = (seeing_arcsec / max(pixscale, 1e-6)) / 2.355

    if "SHUTOPEN" in hdr and "SHUTCLSD" in hdr:
        mid = mid_mjd_from_hdr(hdr)
        exptime = float(hdr.get("EXPTIME", 30.0))
    elif sci_path:
        _, sci_hdr = _open2d(sci_path)
        mid = mid_mjd_from_hdr(sci_hdr)
        exptime = float(sci_hdr.get("EXPTIME", 30.0))
    else:
        mid = mid_mjd_from_hdr(hdr)
        exptime = float(hdr.get("EXPTIME", 30.0))

    diff = np.where(m == 0, diff, np.nan)
    band = str(hdr.get("FILTER", "g")).replace("ZTF_", "").replace("ztf", "")
    return DiffExposure(
        diff=diff, white=white, var=var, mask=m, wcs=wcs, mid_mjd=float(mid),
        magzp=float(hdr.get("MAGZP", 26.0)), psf_sigma_px=float(psf_sigma_px),
        exptime_s=exptime, pixscale_arcsec=pixscale,
        obscode=cfg.get("obscode", "I41"),
        band=(band[:1] or "g"),
        idbase=idbase or diff_path.split("/")[-1].split(".")[0],
        image_index=image_index,
        meta=dict(path=diff_path, maglim=hdr.get("MAGLIM"),
                  infobits=hdr.get("INFOBITS"), seeing=seeing_arcsec,
                  field=hdr.get("FIELDID"), ccdid=hdr.get("CCDID"),
                  qid=hdr.get("QID")))


def mask_path_for(diff_path):
    """scimrefdiffimg[.fz] -> the sibling mskimg.fits for the same quadrant."""
    for a in ("_scimrefdiffimg.fits.fz", "_scimrefdiffimg.fits"):
        if diff_path.endswith(a):
            return diff_path[: -len(a)] + "_mskimg.fits"
    raise ValueError(f"cannot derive mskimg path from {diff_path}")


def prepare_sequence(diff_paths, cfg=None, mask_paths=None, qa_dir=None):
    """Prepare every exposure of one (night, field, ccdid, qid) quadrant visit set.

    Unlike the CSS adapter there is NO registration or leave-one-out differencing
    to do -- IPAC already reference-subtracted each frame independently -- so a
    sequence is just per-exposure preparation with distinct image indices. The
    sequence still matters: it is what makes the cross-exposure repetition filter
    (static detector artifacts recurring at the same sky position and PA)
    possible, and that is the main purity cut available on ZTF diffs.
    """
    cfg = cfg or {}
    if mask_paths is None:
        mask_paths = [mask_path_for(p) for p in diff_paths]
    exps = [prepare_exposure(d, mk, cfg=cfg, image_index=i)
            for i, (d, mk) in enumerate(zip(diff_paths, mask_paths))]
    # Time-order, so image_index is monotonic in epoch as the image log assumes.
    exps.sort(key=lambda e: e.mid_mjd)
    for i, e in enumerate(exps):
        e.image_index = i
    return exps
