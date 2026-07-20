#!/usr/bin/env python
"""WCS-to-WCS registration without the reproject package.

For each target pixel grid (reference frame), compute the source-frame pixel
coordinates via wcs_ref.pixel -> world -> wcs_src.pixel, then resample with
scipy.ndimage.map_coordinates (cubic spline for data, nearest for masks).
Handles shift + rotation + differential TAN distortion exactly; chunked by rows
to bound memory (a 5280^2 float64 coordinate pair set is ~450 MB per array,
fine, but chunking keeps peak usage low with 4 frames in flight).

An independent FFT phase-correlation check on a central block guards against a
broken WCS (the failure mode that silently poisons the whole differencing).
"""
import numpy as np
from scipy.ndimage import map_coordinates


def resample_to(ref_wcs, src_img, src_wcs, order=3, chunk=512, mask=None):
    """Resample src_img (with src_wcs) onto the pixel grid of ref_wcs.

    Returns (resampled, resampled_mask). Out-of-frame -> NaN (mask=1).
    mask: optional uint8 array; resampled with nearest-neighbor, out-of-frame=1.
    """
    H, W = src_img.shape
    out = np.full((H, W), np.nan)
    mout = np.ones((H, W), dtype=np.uint8)
    xs = np.arange(W, dtype=float)
    for y0 in range(0, H, chunk):
        y1 = min(y0 + chunk, H)
        yy, xx = np.meshgrid(np.arange(y0, y1, dtype=float), xs, indexing="ij")
        world = ref_wcs.pixel_to_world_values(xx.ravel(), yy.ravel())
        sx, sy = src_wcs.world_to_pixel_values(world[0], world[1])
        coords = np.vstack([sy, sx])
        out[y0:y1] = map_coordinates(src_img, coords, order=order, mode="constant",
                                     cval=np.nan, prefilter=(order > 1)).reshape(y1 - y0, W)
        if mask is not None:
            mm = map_coordinates(mask, coords, order=0, mode="constant",
                                 cval=1).reshape(y1 - y0, W)
            mout[y0:y1] = mm
        else:
            inb = ((sx >= 0) & (sx <= W - 1) & (sy >= 0) & (sy <= H - 1)).reshape(y1 - y0, W)
            mout[y0:y1] = (~inb).astype(np.uint8)
    bad = ~np.isfinite(out)
    mout[bad] = np.maximum(mout[bad], 1)
    return out, mout


def phase_shift(img_a, img_b, block=1024):
    """FFT phase-correlation shift (dy, dx) of central blocks (b relative to a),
    to sub-pixel via parabolic peak interpolation. NaNs -> median."""
    H, W = img_a.shape
    y0, x0 = (H - block) // 2, (W - block) // 2
    a = img_a[y0:y0 + block, x0:x0 + block].astype(float)
    b = img_b[y0:y0 + block, x0:x0 + block].astype(float)
    a = np.where(np.isfinite(a), a, np.nanmedian(a))
    b = np.where(np.isfinite(b), b, np.nanmedian(b))
    a -= a.mean(); b -= b.mean()
    Fa, Fb = np.fft.rfft2(a), np.fft.rfft2(b)
    R = Fa * np.conj(Fb)
    R /= np.maximum(np.abs(R), 1e-12)
    corr = np.fft.irfft2(R, s=a.shape)
    peak = np.unravel_index(np.argmax(corr), corr.shape)

    def parab(cm, c0, cp):
        d = (cm - cp) / (2 * (cm - 2 * c0 + cp)) if (cm - 2 * c0 + cp) != 0 else 0.0
        return float(np.clip(d, -1, 1))

    py, px = peak
    dy = py + parab(corr[(py - 1) % block, px], corr[py, px], corr[(py + 1) % block, px])
    dx = px + parab(corr[py, (px - 1) % block], corr[py, px], corr[py, (px + 1) % block])
    if dy > block / 2:
        dy -= block
    if dx > block / 2:
        dx -= block
    return dy, dx


def wcs_implied_shift(ref_wcs, src_wcs, xy):
    """Mean pixel shift the WCS pair implies at position xy=(x, y): where does
    ref pixel (x,y) land in src pixel coords?"""
    ra, dec = ref_wcs.pixel_to_world_values(xy[0], xy[1])
    sx, sy = src_wcs.world_to_pixel_values(ra, dec)
    return float(sy - xy[1]), float(sx - xy[0])
