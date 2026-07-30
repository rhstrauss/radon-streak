#!/usr/bin/env python
"""Negated-diff FP calibration report: purity, and where to put `mf_snr_min`.

  fp_report.py --fits 'work/ztf_fpcal/fits/*.json' \
               --real-fits 'work/ztf_pilot_a/fits/*.json' [--nunits 40]

WHY NEGATION WORKS: negating the diff turns every real (positive) source negative,
so a detector hunting positive streaks sees a source-free image with completely
realistic noise, subtraction residuals and detector artifacts. Every surviving
detection is a false positive by construction.

WHAT TO READ OFF IT: the FP count alone is not purity. Purity needs the SAME units
run both ways -- real detections are (artifacts + true sources), negated are
(artifacts alone), so the EXCESS is the real population. First ZTF measurement, on
40 shared units: real 394 vs negated 335 = **0.85 FP fraction, i.e. ~15% purity**
at mf_snr_min = 6.0. Sane detection *density* concealed poor *purity*.

USE --fits, NOT the trail CSVs. The hldet CSV carries no mf_snr, so the only proxy
available there is 2.5/ln10/sigmag, and that does NOT discriminate: measured p50 134
for pure false positives. `det_qual` is likewise saturated at 1.00 for both real and
FP detections, so it cannot be used as a cut either. Dump the fits with FITJSON=1.
"""
import argparse
import glob
import json
import math


def load_snr(pattern):
    """-> (list of mf_snr, list of trail_len, n_files). Tolerates the CSV fallback."""
    snr, lens = [], []
    files = sorted(glob.glob(pattern)) if pattern else []
    for p in files:
        try:
            per_exp = json.load(open(p))
        except (ValueError, OSError):
            continue
        # run_ztf_sequence --json writes a list (per exposure) of lists of fits.
        for fits_i in per_exp:
            if isinstance(fits_i, dict):
                fits_i = [fits_i]
            for f in fits_i:
                v = f.get("mf_snr")
                if isinstance(v, (int, float)) and v == v:
                    snr.append(float(v))
                    lens.append(float(f.get("trail_len", 0.0) or 0.0))
    return snr, lens, len(files)


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q / 100.0 * len(v)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", required=True, help="glob over NEGATED per-unit fit JSONs")
    ap.add_argument("--real-fits", default=None,
                    help="glob over the SAME units run un-negated; enables purity")
    ap.add_argument("--nunits", type=int, default=0)
    ap.add_argument("--targets", default="1.0,0.5,0.2,0.1",
                    help="FP-per-unit targets to report a threshold for")
    a = ap.parse_args()

    fp, fplen, nf = load_snr(a.fits)
    print(f"negated-diff FP calibration: {nf} unit fit-dumps, "
          f"{len(fp)} FALSE POSITIVES")
    if a.nunits:
        print(f"  units processed: {a.nunits}  ->  {len(fp)/a.nunits:.2f} FP/unit")
    if not fp:
        print("  NO false positives parsed. Either the threshold is very "
              "conservative, the negate path did not take effect, or the runs were "
              "made without FITJSON=1 (in which case there are no fit dumps to read).")
        return

    print(f"  FP mf_snr : min {min(fp):7.2f}  p50 {pct(fp,50):7.2f}  "
          f"p90 {pct(fp,90):7.2f}  p99 {pct(fp,99):7.2f}  max {max(fp):7.2f}")
    print(f"  FP trail_len: p50 {pct(fplen,50):6.2f}\"  p90 {pct(fplen,90):6.2f}\"  "
          f"max {max(fplen):6.2f}\"")

    # Threshold to reach a target FP rate: the (1 - target*nunits/N)th percentile.
    if a.nunits:
        print("\n  mf_snr_min needed for a given FP rate (purity cost unknown until "
              "recall is re-checked):")
        for t in [float(x) for x in a.targets.split(",")]:
            keep = t * a.nunits
            if keep >= len(fp):
                print(f"    {t:5.2f} FP/unit : already met at the current threshold")
                continue
            q = 100.0 * (1.0 - keep / len(fp))
            print(f"    {t:5.2f} FP/unit : mf_snr_min >= {pct(fp, q):7.2f}")

    if a.real_fits:
        real, _, nr = load_snr(a.real_fits)
        if real:
            frac = len(fp) / len(real)
            print(f"\n  PURITY (same units, real vs negated):")
            print(f"    real {len(real)} dets over {nr} units, "
                  f"negated {len(fp)} over {nf}")
            print(f"    FP fraction {frac:.2f}  ->  purity ~{100*(1-frac):.0f}%")
            if frac > 0.5:
                print("    ARTIFACT-DOMINATED. Do not run a large campaign at this "
                      "threshold: the catalog would poison the linker it feeds.")
            # Where does the real excess actually live?
            print("    real-excess by mf_snr decile (real_n - fp_n, scaled to "
                  "equal units):")
            k = (nr / nf) if nf else 1.0
            edges = [pct(real + fp, q) for q in range(0, 100, 10)] + [
                max(real + fp) + 1]
            for i in range(10):
                lo, hi = edges[i], edges[i + 1]
                rn = sum(1 for v in real if lo <= v < hi)
                fn = sum(1 for v in fp if lo <= v < hi) * k
                print(f"      mf_snr {lo:8.2f}-{hi:8.2f}: real {rn:5d} "
                      f"fp~{fn:7.1f} excess {rn-fn:+8.1f}")

    print("\n  -> pick mf_snr_min from the table above, then RE-CHECK RECALL "
          "(bin/gate_recall.py / injected trails) before freezing it. This script "
          "measures purity only.")


if __name__ == "__main__":
    main()
