#!/usr/bin/env python
"""P6: run the FRT pipeline on ZTF quadrant diffs (head-to-head vs ztf_streak MF).

Usage:
  run_ztf.py --diff D.fits.fz --mask M.fits [--sci S.fits] --out trails.csv
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from streakradon.adapters.ztf import prepare_exposure      # noqa: E402
from streakradon.pipeline import process_exposure          # noqa: E402
from streakradon import rb                                 # noqa: E402
from streakradon.hldet_io import fit_to_row, write_hldet   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_cfg():
    import yaml
    return yaml.safe_load(open(os.path.join(ROOT, "config", "ztf.yaml")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--sci", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cfg = load_cfg()
    t0 = time.time()
    e = prepare_exposure(args.diff, args.mask, args.sci, cfg)
    t_prep = time.time() - t0
    fits_out = process_exposure(e, cfg, survey="ztf")
    rows = []
    for j, fit in enumerate(fits_out):
        q = rb.det_qual(fit["mf_snr"], fit["chi2r"])
        rows.append(fit_to_row(fit, e.mid_mjd, f"{e.idbase}_c{j:03d}", e.band,
                               e.obscode, image=0, qual=q))
    write_hldet(args.out, rows)
    print(f"prep {t_prep:.0f}s, total {time.time()-t0:.0f}s; "
          f"{len(rows)} detections -> {args.out}")
    if args.json:
        json.dump([{k: v for k, v in f.items() if k != "cand"} for f in fits_out],
                  open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
