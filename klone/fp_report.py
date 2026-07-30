#!/usr/bin/env python
"""Summarise a negated-diff run: every detection is a false positive, so this is
the FP rate and the mf_snr distribution that sets `detect.mf_snr_min`.

  fp_report.py --trails 'work/ztf_fpcal/trails/*_trails.csv' --nunits 40

The catalog carries `det_qual` rather than raw mf_snr, so the SNR proxy used here
is trail SNR reconstructed from mag/sigmag (2.5/ln10 / sigmag). Report both that
and det_qual, and read the threshold off the high tail: mf_snr_min should sit
above the bulk of this distribution at whatever FP-per-quadrant rate you are
willing to hand the linker. G96's equivalent measurement gave FPs up to SNR 15.5
with the 5th-highest at 11.2, which is where its 11.0 came from.
"""
import argparse
import csv
import glob
import math


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trails", required=True, help="glob over per-unit trail CSVs")
    ap.add_argument("--nunits", type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(args.trails))
    snr, qual, lens = [], [], []
    for p in files:
        with open(p) as f:
            # hldet_io writes a '#'-prefixed header, so DictReader names the first
            # column '#MJD'. Anything keying on 'MJD' silently reads zero rows --
            # that bug once produced an empty 604k-row merge.
            rdr = csv.DictReader(x.lstrip("#") if i == 0 else x
                                 for i, x in enumerate(f))
            for r in rdr:
                try:
                    sg = float(r["sigmag"])
                    if sg > 0:
                        snr.append(2.5 / math.log(10) / sg)
                    qual.append(float(r["det_qual"]))
                    lens.append(float(r["trail_len"]))
                except (TypeError, ValueError, KeyError):
                    continue

    n = len(qual)
    print(f"negated-diff FP calibration: {len(files)} unit CSVs, {n} FALSE POSITIVES")
    if args.nunits:
        print(f"  units processed: {args.nunits}  ->  {n/args.nunits:.2f} FP/unit")
    if not n:
        print("  ZERO false positives at the current threshold -- either the "
              "threshold is very conservative (check recall before relaxing it) "
              "or the negate path did not take effect.")
        return

    def pct(v, q):
        v = sorted(v)
        return v[min(len(v) - 1, int(q / 100.0 * len(v)))]

    for name, v in (("snr(from sigmag)", snr), ("det_qual", qual),
                    ("trail_len[\"]", lens)):
        if not v:
            continue
        print(f"  {name:>18s}: min {min(v):7.2f}  p50 {pct(v,50):7.2f}  "
              f"p90 {pct(v,90):7.2f}  p99 {pct(v,99):7.2f}  max {max(v):7.2f}")
    top = sorted(snr, reverse=True)[:10]
    print("  10 highest FP snr:", " ".join(f"{x:.1f}" for x in top))
    print("  -> set detect.mf_snr_min above the tail you refuse to hand the "
          "linker, then re-check recall with bin/gate_recall.py before freezing.")


if __name__ == "__main__":
    main()
