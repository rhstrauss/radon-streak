#!/usr/bin/env python
"""P2 QA: validate the G96 arch WCS against Gaia DR3.

Detect isolated stars on each arch frame, centroid them, convert via the header
WCS, and match against Gaia DR3 (proper-motion propagated to the observation
epoch). Report median offset + RMS. Gate: RMS < 0.3 px (0.46"), no systematic
offset > 0.2 px.
"""
import glob
import os
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import maximum_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PATHS = sorted(glob.glob(
    "/astro/store/shire/aheinze/Shared/G96_images/G96_20240501_2B_N25057_01_000?.arch.fz"))


def detect_stars(img, nmax=400, thresh_sigma=25.0, stamp=6, isolation=15):
    good = np.isfinite(img)
    med = np.median(img[good])
    mad = 1.4826 * np.median(np.abs(img[good] - med))
    mx = maximum_filter(img, size=isolation)
    peaks = (img == mx) & (img > med + thresh_sigma * mad) & (img < 60000)
    peaks[:stamp + 1, :] = peaks[-stamp - 1:, :] = False
    peaks[:, :stamp + 1] = peaks[:, -stamp - 1:] = False
    ys, xs = np.where(peaks)
    order = np.argsort(img[ys, xs])[::-1][:nmax]
    ys, xs = ys[order], xs[order]
    out = []
    yy, xx = np.mgrid[-stamp:stamp + 1, -stamp:stamp + 1].astype(float)
    for y, x in zip(ys, xs):
        sub = np.clip(img[y - stamp:y + stamp + 1, x - stamp:x + stamp + 1] - med, 0, None)
        tot = sub.sum()
        if tot <= 0:
            continue
        out.append((x + (sub * xx).sum() / tot, y + (sub * yy).sum() / tot))
    return np.array(out)


def gaia_cone(ra, dec, radius_deg, epoch, gmax=17.5):
    from astroquery.gaia import Gaia
    Gaia.ROW_LIMIT = 20000
    q = f"""SELECT ra, dec, pmra, pmdec, phot_g_mean_mag FROM gaiadr3.gaia_source
            WHERE 1=CONTAINS(POINT('ICRS', ra, dec),
                             CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))
            AND phot_g_mean_mag < {gmax}"""
    tab = Gaia.launch_job_async(q).get_results()
    dt = epoch - 2016.0
    ra_ep = np.array(tab["ra"]) + np.nan_to_num(np.array(tab["pmra"])) * dt / 3.6e6 \
        / np.cos(np.radians(np.array(tab["dec"])))
    dec_ep = np.array(tab["dec"]) + np.nan_to_num(np.array(tab["pmdec"])) * dt / 3.6e6
    return ra_ep, dec_ep


def main():
    results = {}
    gaia_cache = None
    for path in PATHS:
        hdul = fits.open(path)
        h = hdul[1]
        hdr = h.header
        img = np.asarray(h.data, float)
        wcs = WCS(hdr)
        epoch = 2000.0 + (hdr["MJD"] - 51544.5) / 365.25
        stars = detect_stars(img)
        sra, sdec = wcs.pixel_to_world_values(stars[:, 0], stars[:, 1])
        if gaia_cache is None:
            cra, cdec = wcs.pixel_to_world_values(img.shape[1] / 2, img.shape[0] / 2)
            gaia_cache = gaia_cone(float(cra), float(cdec), 1.65, epoch)
        gra, gdec = gaia_cache
        # nearest-neighbor match within 3"
        d_ra = []
        d_dec = []
        for r, d in zip(sra, sdec):
            dr = (gra - r) * np.cos(np.radians(d)) * 3600.0
            dd = (gdec - d) * 3600.0
            sep2 = dr ** 2 + dd ** 2
            i = np.argmin(sep2)
            if sep2[i] < 9.0:
                d_ra.append(dr[i])
                d_dec.append(dd[i])
        d_ra, d_dec = np.array(d_ra), np.array(d_dec)
        # 3-sigma clip
        for _ in range(3):
            keep = (np.abs(d_ra - np.median(d_ra)) < 3 * np.std(d_ra) + 1e-9) & \
                   (np.abs(d_dec - np.median(d_dec)) < 3 * np.std(d_dec) + 1e-9)
            d_ra, d_dec = d_ra[keep], d_dec[keep]
        rms = np.sqrt(np.mean(d_ra ** 2 + d_dec ** 2) / 2)
        name = os.path.basename(path)
        results[name] = (len(d_ra), np.median(d_ra), np.median(d_dec), rms)
        print(f"{name}: {len(d_ra)} Gaia matches, median offset "
              f"({np.median(d_ra):+.3f}, {np.median(d_dec):+.3f})\", per-axis RMS {rms:.3f}\"")
    worst_rms = max(v[3] for v in results.values())
    worst_off = max(max(abs(v[1]), abs(v[2])) for v in results.values())
    ok = worst_rms < 0.46 and worst_off < 0.31
    print(f"\nGate: worst RMS {worst_rms:.3f}\" (<0.46\"), worst median offset "
          f"{worst_off:.3f}\" (<0.31\") -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
