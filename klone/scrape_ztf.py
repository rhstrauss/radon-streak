#!/usr/bin/env python
"""Build a ZTF work-unit list from the IRSA/IBE `ztf/products/sci` metadata table.

Downloads NO pixels. One night at a time (the table is ~33k rows/night across all
64 readout channels), grouped into WORK UNITS:

    unit = one (night, field, ccdid, qid) quadrant, with ALL of that night's
           exposures of it (typically 2 same-band revisits, up to 4 across g+r)

That grouping is deliberate: it is what lets run_ztf_sequence.py apply the
cross-exposure repetition filter, the main purity cut on ZTF diffs.

Output: work/<TAG>/<TAG>_units.txt, one line per unit, whitespace separated:

    <night YYYYMMDD> <field> <ccdid> <qid> <nexp> <ffd,ffd,...> <filt,filt,...>

Usage:
  scrape_ztf.py --tag ztf_h2_2024 --start 2024-09-01 --end 2024-12-31
                [--ccdid 1,2] [--qid 1] [--max-seeing 4.0] [--append]
"""
import argparse
import csv
import io
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, timedelta

IBE = "https://irsa.ipac.caltech.edu/ibe/search/ztf/products/sci"
COLUMNS = "field,ccdid,qid,filtercode,obsjd,filefracday,infobits,seeing,maglimit"
SR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# IRSA 403s urllib's default User-Agent on some hosts; be explicit everywhere.
UA = "streak_radon/1.0 (klone; rstrau@uw.edu)"


def mjd_of(d):
    """MJD at 00:00 UT of a calendar date (proleptic Gregorian)."""
    return d.toordinal() - date(1858, 11, 17).toordinal()


def fetch_night(d, retries=5):
    """One IBE query covering the observing night that STARTS on date d.

    The window is JD [noon d, noon d+1) so a whole night lands in one bucket
    regardless of UT rollover at Palomar (UT-8).
    """
    jd0 = mjd_of(d) + 2400000.5 + 0.5           # noon UT on d
    jd1 = jd0 + 1.0
    where = f"obsjd>{jd0:.4f} AND obsjd<{jd1:.4f}"
    url = (f"{IBE}?WHERE={urllib.parse.quote(where)}"
           f"&COLUMNS={COLUMNS}&ct=csv")
    last = None
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=900) as r:
                return list(csv.DictReader(io.StringIO(r.read().decode())))
        except Exception as e:                    # noqa: BLE001
            last = e
            time.sleep(5 * (a + 1))
    print(f"  WARN {d}: metadata query failed after {retries} tries: {last}",
          file=sys.stderr)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--ccdid", default=None, help="comma list, default all 16")
    ap.add_argument("--qid", default=None, help="comma list, default all 4")
    ap.add_argument("--max-seeing", type=float, default=5.0,
                    help='reject exposures with SEEING above this (arcsec). A '
                         'trail is spread over FWHM x length, so poor seeing '
                         'costs trail SNR faster than point-source SNR.')
    ap.add_argument("--require-infobits-zero", action="store_true", default=True)
    ap.add_argument("--allow-infobits", dest="require_infobits_zero",
                    action="store_false")
    ap.add_argument("--append", action="store_true",
                    help="add to an existing unit list instead of replacing it")
    args = ap.parse_args()

    ccds = set(int(x) for x in args.ccdid.split(",")) if args.ccdid else None
    qids = set(int(x) for x in args.qid.split(",")) if args.qid else None

    w = os.path.join(SR, "work", args.tag)
    os.makedirs(w, exist_ok=True)
    outp = os.path.join(w, f"{args.tag}_units.txt")
    mode = "a" if args.append else "w"

    d0 = date.fromisoformat(args.start)
    d1 = date.fromisoformat(args.end)
    n_units = n_exp = n_nights = 0
    n_rej_bits = n_rej_seeing = 0
    failed = []

    with open(outp, mode) as out:
        d = d0
        while d <= d1:
            rows = fetch_night(d)
            if rows is None:
                failed.append(d.isoformat())
                d += timedelta(days=1)
                continue
            groups = defaultdict(list)
            for r in rows:
                try:
                    ccd, qid = int(r["ccdid"]), int(r["qid"])
                except (TypeError, ValueError):
                    continue
                if ccds and ccd not in ccds:
                    continue
                if qids and qid not in qids:
                    continue
                if args.require_infobits_zero:
                    try:
                        if int(float(r["infobits"] or 0)) != 0:
                            n_rej_bits += 1
                            continue
                    except ValueError:
                        n_rej_bits += 1
                        continue
                try:
                    if float(r["seeing"]) > args.max_seeing:
                        n_rej_seeing += 1
                        continue
                except (TypeError, ValueError):
                    pass          # missing seeing: keep, the vetter will judge
                groups[(int(r["field"]), ccd, qid)].append(
                    (float(r["obsjd"]), r["filefracday"], r["filtercode"]))
            night = d.strftime("%Y%m%d")
            for (field, ccd, qid), exps in sorted(groups.items()):
                exps.sort()
                ffds = ",".join(e[1] for e in exps)
                filts = ",".join(e[2] for e in exps)
                out.write(f"{night} {field} {ccd} {qid} {len(exps)} "
                          f"{ffds} {filts}\n")
                n_units += 1
                n_exp += len(exps)
            if groups:
                n_nights += 1
            print(f"  {night}: {len(groups):6d} units, "
                  f"{sum(len(v) for v in groups.values()):6d} exposures")
            d += timedelta(days=1)

    print(f"\nTAG={args.tag}  nights={n_nights}  units={n_units}  "
          f"quadrant-exposures={n_exp}")
    print(f"  rejected: infobits!=0 {n_rej_bits}, seeing>{args.max_seeing}\" "
          f"{n_rej_seeing}")
    # 9.4 MB diff + 18.9 MB mask per quadrant-exposure; sciimg is NOT needed
    # because the diff header carries SHUTOPEN/SHUTCLSD.
    print(f"  download volume ~= {n_exp * 28.3 / 1e6:.2f} TB "
          f"(diff+mask only; sciimg avoided)")
    print(f"  -> {outp}")
    if failed:
        print(f"  METADATA QUERY FAILED for {len(failed)} night(s): "
              f"{','.join(failed)}\n  re-run with --append and a narrower range.")
        sys.exit(2)


if __name__ == "__main__":
    main()
