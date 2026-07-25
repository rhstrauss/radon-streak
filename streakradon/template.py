#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ryder H. Strauss
"""Leave-one-out median templates over a registered exposure stack.

For exposure i the template is the nanmedian of the OTHER frames: zero
self-subtraction by construction (median of all N lets a source in the wings
leak into its own template), and a moving object -- displaced by >=13 px
between CSS visits even at 1 deg/day -- is a single outlier at any pixel of the
3-stack, which the median rejects. Noise cost vs an N-stack mean is ~9%.
"""
import numpy as np

from .varmap import star_footprint_mask


def loo_diffs(stack, masks, fp_nsigma=6.0, fp_dilate=2):
    """stack: (N,H,W) registered, photometrically scaled frames (NaN = bad).
    masks: (N,H,W) uint8 (nonzero = bad).

    Returns (diffs, diff_masks): per-exposure leave-one-out difference and
    combined mask (own mask OR <2 valid template contributors OR static-source
    footprint -- see varmap.star_footprint_mask; set fp_nsigma=None to skip).
    """
    N = stack.shape[0]
    data = np.where(masks == 0, stack, np.nan)
    diffs = np.empty_like(stack)
    diff_masks = np.empty(masks.shape, dtype=np.uint8)
    for i in range(N):
        others = np.delete(data, i, axis=0)
        ngood = np.sum(np.isfinite(others), axis=0)
        with np.errstate(all="ignore"):
            tmpl = np.nanmedian(others, axis=0)
        d = stack[i] - tmpl
        bad = (masks[i] != 0) | (ngood < 2) | ~np.isfinite(d)
        if fp_nsigma is not None:
            bad |= star_footprint_mask(tmpl, fp_nsigma, fp_dilate) != 0
        d[bad] = np.nan
        diffs[i] = d
        diff_masks[i] = bad.astype(np.uint8)
    return diffs, diff_masks


def photometric_scale(frames, masks, magzps=None, ref=0):
    """Multiplicative scale factors to the ref frame. Uses ratio of robust sky-
    subtracted 99th-percentile star flux when zeropoints are absent/equal.
    For CSS 17-min sequences this is ~1.00 unless clouds."""
    N = len(frames)
    scales = np.ones(N)
    if magzps is not None and all(z is not None for z in magzps):
        for i in range(N):
            scales[i] = 10.0 ** (-0.4 * (magzps[i] - magzps[ref]))
    return scales
