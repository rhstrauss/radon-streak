#!/usr/bin/env python
"""Merge per-field streak_radon trail CSVs into one MJD-sorted hldet catalog.

  merge_catalog.py -o work/cat/g96_h2_2024.csv work/week work/month work/h2_2024

Applies the TRAIL_LEN FLOOR POLICY established by the 2-night MPC audit:

The MF refine grid cannot resolve a trail shorter than 4.8 px = 7.3", so every
unresolved source is reported at exactly that value. In the audit, 13,796
detections matched known MPC objects; for those the sky motion implied by the
object's own multi-visit track was 0.22-0.35" per 30 s exposure -- i.e. genuinely
untrailed -- yet trail_len came back 7.28-9.13", a factor of 21-38x too large.
Zero percent passed make_trailed_tracklets' 50% -siglenscale gate.

So a reported 7.3" means "shorter than I can measure", NOT "a 7.3" trail". Passing
it through as measured would assert ~21 deg/day of motion for every stationary MBA
and poison tracklet formation. Default policy is therefore --floor-policy ptsrc:
set trail_len/trail_PA to 0 below the floor so heliolinx treats those detections
as point sources.

Caveat (known completeness loss): the injection test recovered 23/23 long trails
(43.8-75") with a median length error of -2.5%, but 4/23 (17%) also collapsed to
the floor -- all of them in exposure 4 of the sequence. Those genuine fast movers
are demoted to point sources by this policy. Fixing that collapse is tracked
separately; it is a detector bug, not a merge bug.
"""
import argparse, collections, csv, glob, os, sys

# trail_len is only meaningful in the INTERIOR of the MF refine search. That search
# is a fixed geometric ladder in PIXELS (streakradon/mf_snr.py:78):
#
#   LENGTH_LADDER = (6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192)
#
# so both boundaries are pixel quantities and their arcsec values differ per
# telescope. Hardcoding G96's numbers would silently mis-handle 703.
#
#   FLOOR  ~4.8 px (below the bottom rungs, after polish_length): unresolved
#           sources all report this ->  7.3" on G96, 14.4" on 703. Confirmed
#           empirically: the 703 pilot's p10 and p25 are both exactly 14.4".
#   CEILING 192 px (the last rung): ANY longer trail is reported AS 192 px ->
#           292" on G96, 578" on 703. This is why both telescopes max out at an
#           identical *pixel* length despite 2x different plate scales -- the tell
#           that it is algorithmic, not astrophysical. Affects 1.9% of G96 and
#           8.8% of 703 detections.
#
# A detection at either boundary is a bound, not a measurement, and must not be
# handed to make_trailed_tracklets as though it were one: the floor would assert
# ~21 deg/day for every stationary MBA, and the ceiling UNDER-states the motion of
# the very fastest objects (their true trail is longer than reported).
GRID_FLOOR_PX = 4.8
GRID_CEIL_PX = 192.0
# From real frame headers; see config/*.yaml.
#   I41 = ZTF/Palomar 48". Its floor is 4.8 * 1.012 = 4.86", which is BELOW the
#   ZTF vet gate len_min_arcsec = 8.0, so ZTF detections are largely immune to the
#   floor artifact that hits 51.7% of G96 -- floor-policy is close to a no-op on
#   I41 and is left enabled only as a guard. The ceiling still applies: 192 px =
#   194.3" on ZTF, i.e. >6480"/hr in a 30 s exposure.
PIXSCALE = {"G96": 1.52, "703": 3.0087, "I41": 1.012}


def _pixscale(obscode):
    ps = PIXSCALE.get(str(obscode).strip().upper())
    if ps is None:
        ps = PIXSCALE.get(str(obscode).strip())
    return ps


# MEASURED floor overrides. GRID_FLOOR_PX * pixscale is only the floor when the
# MF refine is actually allowed to polish down that far. It is not on ZTF:
# config/ztf.yaml sets detect.min_length = 8 px and vet.len_min_arcsec = 8.0", so
# the lowest length that can be both searched and survive vetting is the ladder's
# 8-px rung = 8.10" -- and a pilot unit put 7 of 13 detections (54%) at exactly
# 8.10". Computing 4.8 * 1.012 = 4.86" instead would mean the floor policy NEVER
# FIRES on I41, handing 8.10" to make_trailed_tracklets as a measurement and
# thereby asserting ~6.5 deg/day for every marginal/unresolved ZTF source.
# G96 keeps the computed value: its floor was measured empirically at 7.3", i.e.
# polish_length does push it below that survey's min_length.
FLOOR_ARCSEC = {"I41": 8.2}          # 8 px x 1.012" + epsilon


def floor_arcsec(obscode):
    """Length below which 'trail_len' means 'unresolved', not a measurement."""
    key = str(obscode).strip().upper()
    if key in FLOOR_ARCSEC:
        return FLOOR_ARCSEC[key]
    ps = _pixscale(obscode)
    return None if ps is None else GRID_FLOOR_PX * ps + 0.1


def ceil_arcsec(obscode):
    """Length at/above which 'trail_len' is the ladder's top rung, i.e. a LOWER
    BOUND on the true trail length rather than a measurement."""
    ps = _pixscale(obscode)
    return None if ps is None else GRID_CEIL_PX * ps - 0.1


COLS = ["MJD", "RA", "Dec", "mag", "trail_len", "trail_PA", "sigmag",
        "sig_across", "sig_along", "image", "idstring", "band", "obscode",
        "known_obj", "det_qual"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="run dirs (each with a trails/ subdir)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--floor-policy", choices=["ptsrc", "inflate", "keep"],
                    default="ptsrc",
                    help="ptsrc: zero trail_len/PA below the floor (default). "
                         "inflate: keep length but set sig_along to the length. "
                         "keep: pass through unchanged (diagnostic only).")
    ap.add_argument("--ceil-policy", choices=["inflate", "keep", "drop"],
                    default="inflate",
                    help="at the ladder's 192px top rung trail_len is a LOWER "
                         "BOUND. inflate: widen sig_along to the length (default). "
                         "keep: pass through unchanged. drop: discard the row.")
    ap.add_argument("--drop-nan-mag", action="store_true",
                    help="drop rows with non-finite mag (~19%% of the pilot)")
    a = ap.parse_args()

    files = []
    for d in a.dirs:
        t = d if os.path.basename(d) == "trails" else os.path.join(d, "trails")
        files += sorted(glob.glob(os.path.join(t, "*_trails.csv")))
    if not files:
        sys.exit("no *_trails.csv found under: " + " ".join(a.dirs))

    rows, nnan, nbad, nunk, ndrop = [], 0, 0, 0, 0
    nfloor = collections.Counter()
    nceil = collections.Counter()
    nobs = collections.Counter()
    for fp in files:
        with open(fp) as fh:
            # hldet_io.HEADER is written as "#MJD,RA,..." -- the leading '#' is
            # part of make_trailed_tracklets' colformat convention. Left as-is it
            # names the first DictReader field "#MJD", so every row KeyErrors and
            # the whole catalog silently comes out empty.
            hdr = fh.readline().lstrip("#").strip().split(",")
            for r in csv.DictReader(fh, fieldnames=hdr):
                try:
                    mjd = float(r["MJD"]); L = float(r["trail_len"])
                except (ValueError, KeyError, TypeError):
                    nbad += 1
                    continue
                obs = r.get("obscode", "")
                nobs[obs] += 1
                fl = floor_arcsec(obs)
                cl = ceil_arcsec(obs)
                if fl is None:
                    # Unknown telescope -> we do not know its ladder bounds in
                    # arcsec, so we must not silently apply G96's. Pass through
                    # and report.
                    nunk += 1
                    fl, cl = -1.0, float("inf")
                mag = r.get("mag", "")
                if a.drop_nan_mag:
                    try:
                        m = float(mag)
                        if m != m:
                            nnan += 1
                            continue
                    except ValueError:
                        nnan += 1
                        continue
                if L < fl:
                    nfloor[obs] += 1
                    if a.floor_policy == "ptsrc":
                        r["trail_len"] = "0.0"
                        r["trail_PA"] = "0.0"
                    elif a.floor_policy == "inflate":
                        r["sig_along"] = r["trail_len"]
                elif L > cl:
                    # Top rung of the ladder: the true trail is at least this long,
                    # so the reported value UNDER-states the motion. sig_along is
                    # widened to the length itself, which is the honest statement
                    # "somewhere from here upward" and keeps the detection usable
                    # instead of discarding a genuine fast mover.
                    nceil[obs] += 1
                    if a.ceil_policy == "inflate":
                        r["sig_along"] = r["trail_len"]
                    elif a.ceil_policy == "drop":
                        ndrop += 1
                        continue
                rows.append((mjd, r))

    rows.sort(key=lambda x: x[0])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        # Re-emit the leading '#' so make_trailed_tracklets' colformat reader sees
        # the same header convention hldet_io writes.
        fh.write("#" + ",".join(COLS) + "\n")
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        for _, r in rows:
            w.writerow(r)

    n = len(rows)
    print(f"fields merged   : {len(files)}")
    print(f"detections      : {n}")
    if n:
        print(f"MJD range       : {rows[0][0]:.5f} .. {rows[-1][0]:.5f}")
        print(f"policies        : floor={a.floor_policy} ceil={a.ceil_policy}")
        for obs, cnt in nobs.most_common():
            fl, cl = floor_arcsec(obs), ceil_arcsec(obs)
            fls = f'{fl:.1f}"' if fl else "UNKNOWN"
            cls = f'{cl:.0f}"' if cl else "UNKNOWN"
            print(f"  obscode {obs or '(blank)':<6s} n={cnt}")
            print(f"    floor {fls:<9s} at floor={nfloor[obs]} "
                  f"({100.0*nfloor[obs]/cnt:.1f}%)")
            print(f"    ceil  {cls:<9s} at ceil ={nceil[obs]} "
                  f"({100.0*nceil[obs]/cnt:.1f}%)")
    if ndrop:
        print(f"dropped at ceil : {ndrop}")
    if nunk:
        print(f"WARNING: {nunk} detections from an obscode with no known pixel "
              f"scale -- floor policy NOT applied to them; add it to PIXSCALE.")
    print(f"dropped nan mag : {nnan}")
    print(f"unparseable rows: {nbad}")
    print(f"wrote           : {a.out}")


if __name__ == "__main__":
    main()
