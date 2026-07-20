#!/usr/bin/env python
"""hldet_colformat01 CSV writer for heliolinx make_trailed_tracklets.

Column order (verified against make_trailed_tracklets.cpp colformat reader and
/astro/store/shire/rstrau/catalina/config_files/hldet_colformat01.txt):

  1 MJD  2 RA  3 Dec  4 mag  5 trail_len  6 trail_PA  7 sigmag  8 sig_across
  9 sig_along  10 image  11 idstring  12 band  13 obscode  14 known_obj  15 det_qual

trail_len in arcsec, trail_PA in deg (motion convention; 180-deg ambiguity is
handled by the linker). RA/Dec = trail center at MID-exposure. MJD = mid-exposure.
"""

HEADER = ("#MJD,RA,Dec,mag,trail_len,trail_PA,sigmag,sig_across,sig_along,"
          "image,idstring,band,obscode,known_obj,det_qual")

COLUMNS = ["MJD", "RA", "Dec", "mag", "trail_len", "trail_PA", "sigmag",
           "sig_across", "sig_along", "image", "idstring", "band", "obscode",
           "known_obj", "det_qual"]


def format_row(mjd, ra, dec, mag, trail_len, trail_PA, sigmag, sig_across,
               sig_along, image, idstring, band, obscode, known_obj=0, det_qual=0.0):
    return ("%.8f,%.7f,%+.7f,%.4f,%.2f,%.2f,%.4f,%.3f,%.3f,%d,%s,%s,%s,%d,%.3f"
            % (mjd, ra, dec, mag, trail_len, trail_PA, sigmag, sig_across,
               sig_along, image, idstring, band, obscode, known_obj, det_qual))


def write_hldet(path, rows):
    """rows: list of dicts keyed by COLUMNS (case-sensitive). Sorted by MJD."""
    rows = sorted(rows, key=lambda r: r["MJD"])
    with open(path, "w") as f:
        f.write(HEADER + "\n")
        for r in rows:
            f.write(format_row(
                r["MJD"], r["RA"], r["Dec"], r["mag"], r["trail_len"],
                r["trail_PA"], r["sigmag"], r["sig_across"], r["sig_along"],
                r.get("image", -1), r["idstring"], r["band"], r["obscode"],
                r.get("known_obj", 0), r.get("det_qual", 0.0)) + "\n")
    return len(rows)


def fit_to_row(fit, mjd_mid, idstring, band, obscode, image=-1, known_obj=0, qual=0.0):
    """Map a trail_fit.fit_trail dict to an hldet row dict."""
    return dict(MJD=mjd_mid, RA=fit["ra"], Dec=fit["dec"], mag=fit["mag"],
                trail_len=fit["trail_len"], trail_PA=fit["trail_PA"],
                sigmag=fit["sigmag"], sig_across=fit["sig_across"],
                sig_along=fit["sig_along"], image=image, idstring=idstring,
                band=band, obscode=obscode, known_obj=known_obj, det_qual=qual)
