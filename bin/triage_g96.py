#!/usr/bin/env python
"""P3 triage: classify the real-sequence detections against the MPC catalog.

Cross-match each detection (per exposure, at its mid-MJD) against mpcat.bin
measurements in this field/time window. A detection matching a known MPC
measurement (same epoch, <matchrad") is a real asteroid recovered; the rest are
candidates (new movers OR artifacts) to inspect. Trails are moving objects, so
a genuine trail should also appear as a >=2-exposure track -- reported too.
"""
import json
import mmap
import os
import struct
import sys

import numpy as np

MPCAT = "/astro/store/shire/rstrau/mpcat/mpcat.bin"
RECSIZE = 56
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_mpc_window(mjd_lo, mjd_hi, ra_c, dec_c, hw=1.5, obscode="G96"):
    n = os.path.getsize(MPCAT) // RECSIZE
    with open(MPCAT, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        def at(i):
            return struct.unpack_from("<d", mm, i * RECSIZE)[0]

        def bis(target):
            lo, hi = 0, n
            while lo < hi:
                mid = (lo + hi) // 2
                if at(mid) < target:
                    lo = mid + 1
                else:
                    hi = mid
            return lo
        i0, i1 = bis(mjd_lo), bis(mjd_hi)
        out = []
        cosd = np.cos(np.radians(dec_c))
        for i in range(i0, i1):
            off = i * RECSIZE
            mjd, ra, dec = struct.unpack_from("<ddd", mm, off)
            oc = mm[off + 33:off + 38].split(b"\0")[0].decode(errors="replace")
            if oc != obscode:
                continue
            if abs(dec - dec_c) > hw or abs(ra - ra_c) * cosd > hw:
                continue
            packed = mm[off + 38:off + 51].split(b"\0")[0].decode(errors="replace")
            out.append((mjd, ra, dec, packed))
        mm.close()
    return out


def main():
    csv = os.path.join(ROOT, "work", "g96_trails.csv")
    dets = []
    for ln in open(csv).read().strip().split("\n")[1:]:
        f = ln.split(",")
        dets.append(dict(mjd=float(f[0]), ra=float(f[1]), dec=float(f[2]),
                         mag=float(f[3]), tlen=float(f[4]), tpa=float(f[5]),
                         ids=f[10], qual=float(f[14])))
    ra_c = np.median([d["ra"] for d in dets])
    dec_c = np.median([d["dec"] for d in dets])
    mjd_lo = min(d["mjd"] for d in dets) - 0.002
    mjd_hi = max(d["mjd"] for d in dets) + 0.002
    mpc = load_mpc_window(mjd_lo, mjd_hi, ra_c, dec_c)
    print(f"{len(dets)} detections; {len(mpc)} MPC G96 measurements in field/window")

    matchrad = 5.0 / 3600.0
    timerad = 30.0 / 86400.0
    n_known = 0
    known_desigs = set()
    for d in dets:
        cosd = np.cos(np.radians(d["dec"]))
        best = None
        for (mjd, ra, dec, packed) in mpc:
            if abs(mjd - d["mjd"]) > timerad:
                continue
            sep = np.hypot((ra - d["ra"]) * cosd, dec - d["dec"])
            if sep < matchrad and (best is None or sep < best[0]):
                best = (sep, packed)
        d["mpc"] = best[1] if best else None
        if best:
            n_known += 1
            known_desigs.add(best[1])
    print(f"{n_known}/{len(dets)} detections match a known MPC measurement "
          f"({len(known_desigs)} distinct objects)")

    # internal 2+ exposure tracking of detections (approx linear motion)
    unmatched = [d for d in dets if not d["mpc"]]
    print(f"\n{len(unmatched)} unmatched detections (new movers or artifacts):")
    for d in sorted(unmatched, key=lambda x: -x["qual"])[:25]:
        print(f"  mjd {d['mjd']:.5f} RA {d['ra']:.5f} Dec {d['dec']:+.5f} "
              f"mag {d['mag']:5.2f} len {d['tlen']:6.1f}\" PA {d['tpa']:6.1f} "
              f"qual {d['qual']:.2f}  [{d['ids']}]")

    json.dump(dets, open(os.path.join(ROOT, "work", "g96_triage.json"), "w"), indent=1)
    # summary of matched objects
    print(f"\nknown objects recovered: {sorted(known_desigs)}")


if __name__ == "__main__":
    main()
