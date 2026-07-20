#!/usr/bin/env python
"""P3: resolve the G96 header-MJD convention empirically from the MPC record.

CSS submitted this night's G96 measurements to the MPC; mpcat.bin (the
pre-indexed 511M-record MPC measurement catalog) contains them. The epochs CSS
reports to the MPC are, by MPC convention, the MID-exposure times. So for the
four example exposures, clustering the G96 measurement epochs in this field and
differencing against the four header MJDs yields the offset directly:
  offset ~ +15 s  -> header MJD is shutter-OPEN
  offset ~   0 s  -> header MJD is already mid-exposure
"""
import mmap
import os
import struct
import sys

import numpy as np

MPCAT = "/astro/store/shire/rstrau/mpcat/mpcat.bin"
RECSIZE = 56
HEADER_MJDS = [60431.1531927475, 60431.1588724098, 60431.1645576108, 60431.1702575017]
FIELD_RA, FIELD_DEC, FIELD_HW = 136.49, 25.42, 1.25  # deg, half-width

MJD_LO, MJD_HI = HEADER_MJDS[0] - 0.002, HEADER_MJDS[-1] + 0.003


def rec(mm, i):
    off = i * RECSIZE
    mjd, ra, dec = struct.unpack_from("<ddd", mm, off)
    mag = struct.unpack_from("<f", mm, off + 24)[0]
    band = mm[off + 28:off + 33].split(b"\0")[0].decode(errors="replace")
    obscode = mm[off + 33:off + 38].split(b"\0")[0].decode(errors="replace")
    packed = mm[off + 38:off + 51].split(b"\0")[0].decode(errors="replace")
    return mjd, ra, dec, mag, band, obscode, packed


def bisect_mjd(mm, n, target):
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if struct.unpack_from("<d", mm, mid * RECSIZE)[0] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def main():
    n = os.path.getsize(MPCAT) // RECSIZE
    with open(MPCAT, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        i0, i1 = bisect_mjd(mm, n, MJD_LO), bisect_mjd(mm, n, MJD_HI)
        print(f"MJD window [{MJD_LO:.4f},{MJD_HI:.4f}]: records {i0}..{i1} "
              f"({i1-i0} in window)")
        hits = []
        cosd = np.cos(np.radians(FIELD_DEC))
        for i in range(i0, i1):
            mjd, ra, dec, mag, band, obscode, packed = rec(mm, i)
            if obscode != "G96":
                continue
            if abs(dec - FIELD_DEC) > FIELD_HW or abs(ra - FIELD_RA) * cosd > FIELD_HW:
                continue
            hits.append((mjd, ra, dec, mag, packed))
        mm.close()
    print(f"{len(hits)} G96 measurements in this field/window")
    if not hits:
        print("no MPC measurements found -> cannot infer; keep shutter-open assumption")
        sys.exit(2)
    mjds = np.array([h[0] for h in hits])
    print("\nper-exposure offsets (MPC epoch - header MJD):")
    votes = []
    for hm in HEADER_MJDS:
        sel = mjds[np.abs(mjds - hm) < 0.0008]  # within ~70 s
        if sel.size == 0:
            print(f"  header {hm:.7f}: no matches")
            continue
        offs = (sel - hm) * 86400.0
        med = np.median(offs)
        print(f"  header {hm:.7f}: {sel.size} meas, median offset {med:+.2f} s "
              f"(scatter {1.4826*np.median(np.abs(offs-med)):.2f} s)")
        votes.append(med)
    if not votes:
        sys.exit(2)
    overall = float(np.median(votes))
    print(f"\noverall median offset: {overall:+.2f} s")
    if 10.0 <= overall <= 20.0:
        print("=> header MJD is SHUTTER-OPEN (MPC epochs ~= header + exptime/2).")
        print("   set config g96.yaml: mjd_is_shutter_open: true")
    elif -5.0 <= overall <= 5.0:
        print("=> header MJD is already MID-exposure.")
        print("   set config g96.yaml: mjd_is_shutter_open: false")
    else:
        print("=> ambiguous offset; investigate (rounding of MPC epochs? different exp time?)")
    # sample identified objects
    print("\nsample measurements:")
    for h in hits[:8]:
        print(f"  {h[0]:.7f} RA {h[1]:9.5f} Dec {h[2]:+9.5f} mag {h[3]:4.1f} [{h[4]}]")


if __name__ == "__main__":
    main()
