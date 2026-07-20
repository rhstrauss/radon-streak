#!/usr/bin/env python
"""Variance maps, masks, and pre-whitening.

The FRT/matched-filter significance model assumes unit Gaussian noise, so the
detection image is always pre-whitened: white = (img - local_bg) / sqrt(var).
The variance is the pixelwise max of a physical (Poisson+read-noise) model when
raw counts are available and an empirical local robust variance (grid MAD,
bilinear-interpolated) which captures noise reshaping by filtering/differencing
and registration residuals near stars -- the bright-source failure mode that
breaks scalar-variance normalizations.
"""
import numpy as np
from scipy.ndimage import binary_dilation, zoom


def local_stats(img, mask=None, grid=128):
    """Robust local median and variance on a coarse grid, bilinearly upsampled.

    Returns (bg_map, var_map) full-size. Masked/non-finite pixels excluded from
    the statistics; cells with <50 good px inherit the global values.
    """
    H, W = img.shape
    ny, nx = int(np.ceil(H / grid)), int(np.ceil(W / grid))
    med = np.zeros((ny, nx))
    var = np.zeros((ny, nx))
    good_all = np.isfinite(img)
    if mask is not None:
        good_all &= (mask == 0)
    gvals = img[good_all]
    gmed = np.median(gvals) if gvals.size else 0.0
    gmad = 1.4826 * np.median(np.abs(gvals - gmed)) if gvals.size else 1.0
    for iy in range(ny):
        for ix in range(nx):
            sl = (slice(iy * grid, min((iy + 1) * grid, H)),
                  slice(ix * grid, min((ix + 1) * grid, W)))
            sub = img[sl]
            g = good_all[sl]
            if g.sum() < 50:
                med[iy, ix] = gmed
                var[iy, ix] = gmad ** 2
                continue
            v = sub[g]
            m = np.median(v)
            s = 1.4826 * np.median(np.abs(v - m))
            med[iy, ix] = m
            var[iy, ix] = max(s, 1e-6 * max(gmad, 1e-12)) ** 2
    bg = zoom(med, (H / ny / 1.0, W / nx / 1.0), order=1, grid_mode=True, mode='nearest')[:H, :W]
    vv = zoom(var, (H / ny / 1.0, W / nx / 1.0), order=1, grid_mode=True, mode='nearest')[:H, :W]
    return bg, vv


def whiten(img, mask=None, grid=128, var_floor_frac=0.25, phys_var=None):
    """Pre-whiten a difference image: (img - bg)/sqrt(var); bad px -> 0.

    phys_var : optional physical variance map (Poisson+RN); combined as
               pixelwise max with the empirical local variance.
    Returns (white, var_map, bg_map).
    """
    bg, var = local_stats(img, mask, grid)
    if phys_var is not None:
        var = np.maximum(var, phys_var)
    var = np.maximum(var, var_floor_frac * np.median(var))
    white = (img - bg) / np.sqrt(var)
    bad = ~np.isfinite(white)
    if mask is not None:
        bad |= (mask != 0)
    white[bad] = 0.0
    return white, var, bg


def saturation_masks(raw, saturate, margin=2000, dilate=5, bleed_min_run=10,
                     bright_frac=0.5, halo_a=8, halo_b=12):
    """Masks from RAW counts: saturated cores (dilated), vertical bleed trails,
    bright-star halos. Returns uint8 mask (nonzero = bad)."""
    m = np.zeros(raw.shape, dtype=np.uint8)
    sat = raw >= (saturate - margin)
    if dilate > 0 and sat.any():
        sat = binary_dilation(sat, iterations=dilate)
    m[sat] = 1
    # vertical bleed: CONTIGUOUS saturated runs, extended 2x run length
    satcol = raw >= (saturate - margin)
    for x in np.where(satcol.sum(axis=0) >= bleed_min_run)[0]:
        col = satcol[:, x].astype(np.int8)
        d = np.diff(np.concatenate([[0], col, [0]]))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        for s, t in zip(starts, ends):
            run_len = t - s
            if run_len < bleed_min_run:
                continue
            lo = max(0, s - 2 * run_len)
            hi = min(raw.shape[0], t + 2 * run_len)
            m[lo:hi, max(0, x - 2):x + 3] = 2
    # halos ONLY for truly saturated stars (footprint masking of ordinary stars
    # happens against the template in star_footprint_mask -- circular halos on
    # every bright-ish star over-masked ~15% of the frame)
    bright = raw >= 0.97 * saturate
    if bright.any():
        from scipy.ndimage import label, center_of_mass, maximum
        lab, n = label(bright)
        if n:
            peaks = maximum(raw, lab, index=np.arange(1, n + 1))
            coms = center_of_mass(bright, lab, index=np.arange(1, n + 1))
            npix = np.bincount(lab.ravel())[1:]
            for (cy, cx), pk, npx in zip(np.atleast_2d(coms), np.atleast_1d(peaks),
                                         npix):
                r = halo_a + halo_b * np.sqrt(npx / np.pi)
                sl = (slice(max(0, int(cy - r)), min(raw.shape[0], int(cy + r + 1))),
                      slice(max(0, int(cx - r)), min(raw.shape[1], int(cx + r + 1))))
                yy, xx = np.mgrid[sl]
                d2 = (yy - cy) ** 2 + (xx - cx) ** 2
                m[sl][d2 <= r * r] = np.maximum(m[sl][d2 <= r * r], 3)
    return m


def star_footprint_mask(template, nsigma=6.0, dilate=2):
    """Mask the static-source footprints using the (mover-free) template: pixels
    with significant template flux leave subtraction residuals in the diff --
    89% of >5-sigma whitened outliers sit on these footprints while they cover
    <1% of a G96 frame. A trail crossing a star loses those pixels anyway (the
    fit uses the mask); the rest of the trail survives."""
    fin = template[np.isfinite(template)]
    sky = np.median(fin)
    noise = 1.4826 * np.median(np.abs(fin - sky))
    fp = np.where(np.isfinite(template), template, np.inf) > sky + nsigma * noise
    if dilate > 0:
        fp = binary_dilation(fp, iterations=dilate)
    return fp.astype(np.uint8)


def edge_mask(shape, edge=16):
    m = np.zeros(shape, dtype=np.uint8)
    m[:edge, :] = 4; m[-edge:, :] = 4; m[:, :edge] = 4; m[:, -edge:] = 4
    return m
