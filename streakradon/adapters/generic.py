#!/usr/bin/env python
"""Survey-agnostic adapter: turn an ALREADY-DIFFERENCED FITS image into a
DiffExposure using a small header-keyword map in the config.

This is the plug-and-play path. If your survey pipeline already produces
reference-subtracted (difference) images with a valid WCS and a magnitude zero
point, you do not need a bespoke adapter -- point `config/generic.yaml` at your
header keywords and go. The survey-specific adapters (g96/ztf/atlas) exist only
because those inputs need extra handling (CSS: not differenced -> within-sequence
differencing; ZTF: exact shutter-midpoint timing; ATLAS: broken stamp WCS).

Config schema (all under a `header:` block, with `default:` fallbacks):

  header:
    hdu: 0                 # which HDU holds the 2D image (int index or EXTNAME);
                           # default = first 2D HDU
    mjd_key:   [MJD-OBS, MJD, OBSMJD]   # first present wins
    mjd_convention: start  # 'start' (shutter-open; mid = MJD + exptime/2) or 'mid'
    exptime_key: [EXPTIME, EXPOSURE]
    magzp_key: [MAGZP, MAGZERO, ZP]
    psf_key:   [FWHM, SEEING]
    psf_unit:  fwhm_px     # 'fwhm_px' | 'fwhm_arcsec' | 'sigma_px' | 'sigma_arcsec'
    obscode_key: [OBSCODE, OBS]
    band_key:  [FILTER, BAND]
  default:
    exptime_s: 30.0
    magzp: 26.0
    psf_fwhm_px: 4.0
    obscode: XXX
    band: w
  preprocess:
    bad_lo: -1.0e4         # pixels below this are masked (subtraction bad regions)
    var_grid_px: 128

A separate mask FITS (nonzero = bad) may be supplied; it is OR'd with the
non-finite + bad_lo mask.
"""
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from ..varmap import whiten
from .base import DiffExposure


def _open2d(path, hdu=None):
    """Return (data, header) for the requested HDU, or the first 2D HDU."""
    hdul = fits.open(path)
    try:
        if hdu is not None:
            h = hdul[hdu]
            if getattr(h.data, "ndim", 0) != 2:
                raise ValueError(f"HDU {hdu!r} of {path} is not 2D")
            return np.asarray(h.data, float), h.header
        for h in hdul:
            if h.data is not None and getattr(h.data, "ndim", 0) == 2:
                return np.asarray(h.data, float), h.header
        raise ValueError(f"no 2D HDU in {path}")
    finally:
        hdul.close()


def _first_present(hdr, keys, default=None):
    """First header keyword present from a list (or a single key), else default."""
    if keys is None:
        return default
    if isinstance(keys, str):
        keys = [keys]
    for k in keys:
        if k in hdr and hdr[k] not in (None, ""):
            return hdr[k]
    return default


def _safe_wcs(hdr):
    """astropy WCS with a linear-TAN fallback for headers whose distortion
    terms this wcslib build rejects (e.g. conflicting TPV+SIP; see atlas)."""
    try:
        w = WCS(hdr)
        # trigger a transform so a lazily-raised parse error surfaces here
        w.pixel_to_world_values(0, 0)
        return w
    except Exception:
        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.crpix = [hdr.get("CRPIX1", 1.0), hdr.get("CRPIX2", 1.0)]
        w.wcs.crval = [hdr.get("CRVAL1", 0.0), hdr.get("CRVAL2", 0.0)]
        if "CD1_1" in hdr:
            w.wcs.cd = [[hdr["CD1_1"], hdr.get("CD1_2", 0.0)],
                        [hdr.get("CD2_1", 0.0), hdr["CD2_2"]]]
        else:
            w.wcs.cdelt = [hdr.get("CDELT1", 1.0), hdr.get("CDELT2", 1.0)]
        return w


def _psf_sigma_px(raw, unit, pixscale):
    """Convert a header PSF measurement to a sigma in PIXELS."""
    unit = (unit or "fwhm_px").lower()
    if unit == "fwhm_px":
        return float(raw) / 2.3548
    if unit == "fwhm_arcsec":
        return float(raw) / 2.3548 / pixscale
    if unit == "sigma_px":
        return float(raw)
    if unit == "sigma_arcsec":
        return float(raw) / pixscale
    raise ValueError(f"unknown psf_unit {unit!r}")


def prepare_diff(diff_path, mask_path=None, cfg=None, idbase=None):
    """Prepare one already-differenced FITS as a DiffExposure.

    diff_path : path to the difference image FITS.
    mask_path : optional bad-pixel mask FITS (nonzero = bad).
    cfg       : dict (see module docstring); typically config/generic.yaml.
    """
    cfg = cfg or {}
    hmap = cfg.get("header", {})
    dflt = cfg.get("default", {})
    pp = cfg.get("preprocess", {})

    diff, hdr = _open2d(diff_path, hmap.get("hdu"))
    wcs = _safe_wcs(hdr)
    pixscale = np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600.0

    # ---- mask: external file OR (non-finite | below bad_lo) ----
    bad_lo = pp.get("bad_lo", None)
    mask = ~np.isfinite(diff)
    if bad_lo is not None:
        mask |= (diff < float(bad_lo))
    if mask_path:
        extmask, _ = _open2d(mask_path)
        mask |= (extmask != 0)
    mask = mask.astype(np.uint8)

    # ---- timing ----
    exptime = float(_first_present(hdr, hmap.get("exptime_key", ["EXPTIME"]),
                                   dflt.get("exptime_s", 30.0)))
    mjd = _first_present(hdr, hmap.get("mjd_key", ["MJD-OBS", "MJD", "OBSMJD"]))
    if mjd is None:
        raise KeyError(
            f"no MJD keyword found in {diff_path} "
            f"(tried {hmap.get('mjd_key', ['MJD-OBS', 'MJD', 'OBSMJD'])}); "
            f"set header.mjd_key in the config")
    mjd = float(mjd)
    conv = str(hmap.get("mjd_convention", "start")).lower()
    if conv in ("start", "shutter_open", "open"):
        mid = mjd + exptime / 2.0 / 86400.0
    elif conv in ("mid", "midpoint"):
        mid = mjd
    else:
        raise ValueError(f"unknown mjd_convention {conv!r} (use 'start' or 'mid')")

    # ---- photometry / PSF ----
    magzp = float(_first_present(hdr, hmap.get("magzp_key", ["MAGZP"]),
                                 dflt.get("magzp", 26.0)))
    psf_raw = _first_present(hdr, hmap.get("psf_key", ["FWHM", "SEEING"]))
    if psf_raw is not None and float(psf_raw) > 0:
        psf_sigma = _psf_sigma_px(psf_raw, hmap.get("psf_unit", "fwhm_px"), pixscale)
    else:
        psf_sigma = float(dflt.get("psf_fwhm_px", 4.0)) / 2.3548

    obscode = str(_first_present(hdr, hmap.get("obscode_key", ["OBSCODE", "OBS"]),
                                 dflt.get("obscode", "XXX")))
    band = str(_first_present(hdr, hmap.get("band_key", ["FILTER", "BAND"]),
                              dflt.get("band", "w")))[:1] or "w"

    white, var, bg = whiten(diff, mask, grid=pp.get("var_grid_px", 128))
    diff = np.where(mask == 0, diff, np.nan)
    return DiffExposure(
        diff=diff, white=white, var=var, mask=mask, wcs=wcs, mid_mjd=float(mid),
        magzp=magzp, psf_sigma_px=float(psf_sigma), exptime_s=exptime,
        pixscale_arcsec=float(pixscale), obscode=obscode, band=band,
        idbase=idbase or diff_path.split("/")[-1].split(".")[0], image_index=0,
        meta=dict(path=diff_path, header_mjd=mjd, mjd_convention=conv))
