#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ryder H. Strauss
"""G96 (Catalina Sky Survey) adapter: arch.fz 4-visit sequence -> DiffExposures.

The arch frames are high-pass filtered survey images with a valid TAN WCS (the
variety A. Heinze identified as the streak-finding input). They are NOT
differenced, so star removal is done here by within-sequence differencing:
all frames are resampled onto the FIRST frame's pixel grid (WCS-to-WCS cubic
spline), photometrically scaled, and each exposure gets a leave-one-out median
template. All resulting diffs therefore share the reference WCS.

Timing: the header MJD convention (shutter-open vs midpoint) is unresolved for
CSS; `mjd_is_shutter_open=None` assumes shutter-open (mid = MJD + exptime/2)
and warns. Resolve empirically (known-object residuals) and set the flag.
"""
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from ..psf import measure_psf_sigma
from ..register import phase_shift, resample_to, wcs_implied_shift
from ..template import loo_diffs, photometric_scale
from ..varmap import edge_mask, saturation_masks, whiten
from .base import DiffExposure


def open_arch(path):
    hdul = fits.open(path)
    for h in hdul:
        if h.data is not None and getattr(h.data, "ndim", 0) == 2:
            return np.asarray(h.data, float), h.header
    raise ValueError(f"no 2D HDU in {path}")


def register_stack(paths, cfg=None, verbose=True):
    """Stage A (expensive, injection-independent): load one CSS visit sequence,
    build masks, measure the PSF, register everything onto frame 0's grid, and
    photometrically scale. Returns a dict reusable across injection rounds."""
    cfg = cfg or {}
    pp = cfg.get("preprocess", {})

    frames, headers, wcss = [], [], []
    for p in paths:
        d, h = open_arch(p)
        frames.append(d)
        headers.append(h)
        wcss.append(WCS(h))
    N = len(frames)
    H, W = frames[0].shape

    # per-frame masks from the arch values themselves (star cores reach SATURATE)
    masks = []
    for d, h in zip(frames, headers):
        m = saturation_masks(
            d, saturate=float(h.get("SATURATE", 65535)),
            margin=pp.get("sat_margin_adu", 2000), dilate=pp.get("sat_dilate_px", 5),
            bleed_min_run=pp.get("bleed_min_run_px", 10),
            bright_frac=pp.get("bright_star_frac", 0.5),
            halo_a=pp.get("halo_a_px", 8), halo_b=pp.get("halo_b_px", 12))
        m |= edge_mask(d.shape, pp.get("edge_px", 16))
        masks.append(m)

    # effective PSF from frame stars (arch may be PSF-convolved -> stars carry it)
    psf_sigmas = []
    for d, m in zip(frames, masks):
        sig, nst = measure_psf_sigma(d, m)
        psf_sigmas.append(sig)
        if verbose:
            print(f"  psf sigma {sig:.2f} px from {nst} stars")
    psf_sigma = float(np.nanmedian(psf_sigmas))

    # registration onto frame 0 grid + WCS sanity check via phase correlation
    reg = np.empty((N, H, W))
    regm = np.empty((N, H, W), dtype=np.uint8)
    reg[0], regm[0] = frames[0], masks[0]
    ref_wcs = wcss[0]
    check_px = pp.get("register_check_px", 0.5)
    for i in range(1, N):
        wdy, wdx = wcs_implied_shift(ref_wcs, wcss[i], (W / 2, H / 2))
        pdy, pdx = phase_shift(frames[i], frames[0])
        err = np.hypot(wdy - pdy, wdx - pdx)
        if verbose:
            print(f"  frame {i}: WCS shift ({wdy:+.2f},{wdx:+.2f}) px, "
                  f"phase-corr ({pdy:+.2f},{pdx:+.2f}) px, |diff| {err:.2f}")
        if err > max(check_px, 0.02 * np.hypot(wdy, wdx) + check_px):
            raise RuntimeError(
                f"registration check failed for {paths[i]}: WCS vs phase-corr "
                f"disagree by {err:.2f} px (>{check_px})")
        reg[i], regm[i] = resample_to(ref_wcs, frames[i], wcss[i], mask=masks[i])

    scales = photometric_scale(reg, regm, [h.get("MAGZP") for h in headers])
    for i in range(N):
        reg[i] = reg[i] * scales[i]

    return dict(reg=reg, regm=regm, headers=headers, ref_wcs=ref_wcs,
                psf_sigma=psf_sigma, scales=scales, paths=list(paths))


def build_exposures(stack, cfg=None, verbose=True):
    """Stage B (cheap, re-run after pixel injection into stack['reg']):
    leave-one-out differencing + whitening -> DiffExposures."""
    cfg = cfg or {}
    pp = cfg.get("preprocess", {})
    exptime = float(cfg.get("exptime_s", 30.0))
    mjd_open = cfg.get("mjd_is_shutter_open", None)
    reg, regm = stack["reg"], stack["regm"]
    headers, ref_wcs = stack["headers"], stack["ref_wcs"]
    psf_sigma, scales, paths = stack["psf_sigma"], stack["scales"], stack["paths"]
    N = reg.shape[0]

    diffs, diff_masks = loo_diffs(reg, regm)

    exposures = []
    for i in range(N):
        mjd = float(headers[i]["MJD"])
        if mjd_open is None:
            print(f"  WARNING: G96 MJD convention unresolved; assuming shutter-open "
                  f"(mid = MJD + {exptime/2:.0f}s)", file=sys.stderr)
            mid = mjd + exptime / 2.0 / 86400.0
        elif mjd_open:
            mid = mjd + exptime / 2.0 / 86400.0
        else:
            mid = mjd
        white, var, bg = whiten(diffs[i], diff_masks[i],
                                grid=pp.get("var_grid_px", 128))
        pixscale = np.sqrt(np.abs(np.linalg.det(ref_wcs.pixel_scale_matrix))) * 3600.0
        base = paths[i].split("/")[-1].split(".")[0]
        exposures.append(DiffExposure(
            diff=diffs[i], white=white, var=var, mask=diff_masks[i], wcs=ref_wcs,
            mid_mjd=mid, magzp=float(headers[i].get("MAGZP", 28.0)),
            psf_sigma_px=psf_sigma, exptime_s=exptime, pixscale_arcsec=pixscale,
            obscode=cfg.get("obscode", "G96"), band=cfg.get("band", "G"),
            idbase=base, image_index=i,
            meta=dict(header_mjd=mjd, scale=float(scales[i]), path=paths[i])))
    return exposures


def prepare_sequence(paths, cfg=None, verbose=True, qa_dir=None):
    """Convenience: register_stack + build_exposures (+ optional QA plots)."""
    stack = register_stack(paths, cfg, verbose)
    exposures = build_exposures(stack, cfg, verbose)
    if qa_dir:
        _write_qa(exposures, stack["reg"], qa_dir)
    return exposures


def _write_qa(exposures, reg, qa_dir):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(qa_dir, exist_ok=True)
    for i, e in enumerate(exposures):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
        c = 2640
        sl = (slice(c - 400, c + 400), slice(c - 400, c + 400))
        for ax, im, title in zip(
                axes, [reg[i][sl], e.diff[sl], e.white[sl]],
                ["registered frame", "LOO diff", "whitened"]):
            fin = im[np.isfinite(im)]
            lo, hi = np.percentile(fin, [5, 99]) if fin.size else (0, 1)
            ax.imshow(im, vmin=lo, vmax=hi, cmap="gray", origin="lower")
            ax.set_title(f"exp{i} {title}")
            ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(qa_dir, f"qa_exp{i}.png"), dpi=90)
        plt.close(fig)
