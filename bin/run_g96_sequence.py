#!/usr/bin/env python
"""Full G96 pipeline over one CSS 4-visit sequence:

  arch.fz x4 -> preprocess (register/LOO-diff/whiten) -> FRT per exposure
  -> dedup -> MF refine (PA/L/center grid) -> Veres fit -> RB vetting
  -> cross-exposure repetition filter -> hldet_colformat01 CSV.

Usage:
  run_g96_sequence.py --out trails.csv [--imgs imagelog.txt] [--frames glob]
                      [--mf-snr-min 6] [--qa qa/] [--json cands.json]
"""
import argparse
import glob as globmod
import json
import os
import sys




sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from streakradon.adapters.g96 import prepare_sequence      # noqa: E402
from streakradon.pipeline import process_exposure          # noqa: E402
from streakradon import rb                                 # noqa: E402
from streakradon.hldet_io import fit_to_row, write_hldet   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FRAMES = "/astro/store/shire/aheinze/Shared/G96_images/G96_20240501_2B_N25057_01_000?.arch.fz"


def load_cfg():
    try:
        import yaml
        return yaml.safe_load(open(os.path.join(ROOT, "config", "g96.yaml")))
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=DEFAULT_FRAMES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgs", default=None, help="write make_trailed_tracklets image log")
    ap.add_argument("--json", default=None, help="dump all surviving fits as JSON")
    ap.add_argument("--mf-snr-min", type=float, default=None)
    ap.add_argument("--qa", default=None)
    ap.add_argument("--exposures", default=None,
                    help="optional: preprocessed DiffExposures via pickle (injection path)")
    args = ap.parse_args()

    cfg = load_cfg()
    if args.mf_snr_min is not None:
        cfg.setdefault("detect", {})["mf_snr_min"] = args.mf_snr_min

    if args.exposures:
        import pickle
        exps = pickle.load(open(args.exposures, "rb"))
    else:
        paths = sorted(globmod.glob(args.frames))
        if not paths:
            sys.exit(f"no frames match {args.frames}")
        print(f"preparing {len(paths)} exposures...")
        exps = prepare_sequence(paths, cfg, qa_dir=args.qa)

    per_exp = [process_exposure(e, cfg) for e in exps]

    keep = rb.repetition_filter(per_exp, [e.mid_mjd for e in exps])
    rows = []
    n_rep = 0
    for e, fits_i, keep_i in zip(exps, per_exp, keep):
        for j, (fit, k) in enumerate(zip(fits_i, keep_i)):
            if not k:
                n_rep += 1
                continue
            q = rb.det_qual(fit["mf_snr"], fit["chi2r"])
            rows.append(fit_to_row(fit, e.mid_mjd, f"{e.idbase}_c{j:03d}",
                                   e.band, e.obscode, image=e.image_index, qual=q))
    print(f"repetition filter removed {n_rep}; writing {len(rows)} detections")
    write_hldet(args.out, rows)
    if args.imgs:
        with open(args.imgs, "w") as f:
            for e in exps:
                cra, cdec = e.wcs.pixel_to_world_values(e.diff.shape[1] / 2,
                                                        e.diff.shape[0] / 2)
                f.write(f"{e.mid_mjd:.8f} {float(cra):.6f} {float(cdec):.6f} "
                        f"{e.obscode} {e.exptime_s:.1f}\n")
    if args.json:
        clean = [[{k: v for k, v in f.items() if k != "cand"} for f in fi]
                 for fi in per_exp]
        json.dump(clean, open(args.json, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
