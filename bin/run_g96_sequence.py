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
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from streakradon.adapters.g96 import prepare_sequence      # noqa: E402
from streakradon.frt_driver import detect_streaks          # noqa: E402
from streakradon.mf_snr import refine_candidate            # noqa: E402
from streakradon.trail_fit import fit_trail                # noqa: E402
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


def near_bad_col(cand, mask, reach=12):
    """Is the candidate near masked bleed/saturation structure? (enables the
    conditional detector-axis cut)."""
    x, y = int(round(cand["x"])), int(round(cand["y"]))
    sl = (slice(max(0, y - reach), y + reach + 1), slice(max(0, x - reach), x + reach + 1))
    return bool((mask[sl] > 0).any())


def process_exposure(e, cfg, verbose=True):
    det = cfg.get("detect", {})
    vet = cfg.get("vet", {})
    rb_cfg = dict(
        mf_snr_min=det.get("mf_snr_min", 6.0),
        len_min_arcsec=vet.get("len_min_arcsec", 6.0),
        len_max_arcsec=vet.get("len_max_arcsec", 300.0),
        chi2r_lo=vet.get("chi2r_lo", 0.3), chi2r_hi=vet.get("chi2r_hi", 3.0),
        sig_along_max_arcsec=vet.get("sig_along_max_arcsec", 4.0),
        edge_px=vet.get("edge_px", 30),
    )
    t0 = time.time()
    cands = detect_streaks(e.white, e.psf_sigma_px,
                           tile=det.get("tile", 1024), overlap=det.get("overlap", 128),
                           min_length=det.get("min_length", 8),
                           threshold=det.get("frt_threshold", 5.0))
    t_frt = time.time() - t0
    fits_out, reasons = [], {}
    for c in cands:
        c["near_bad_col"] = near_bad_col(c, e.mask)
        ref = refine_candidate(e.white, c["x"], c["y"], c["pa_rad"], e.psf_sigma_px)
        c.update(snr=ref["snr"], pa_px_deg=np.degrees(ref["pa_rad"]))
        ok, why = rb.prefit_pass(c, rb_cfg, survey="g96")
        if not ok:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        fit = fit_trail(e.diff, e.mask, e.wcs, ref["x"], ref["y"], e.psf_sigma_px,
                        e.magzp, theta0=ref["pa_rad"], h0=ref["L"] / 2.0)
        ok, why = rb.passes_rb(fit, c, imshape=e.diff.shape, cfg=rb_cfg, survey="g96")
        if not ok:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        fr = rb.flank_ratio(e.diff, fit, e.psf_sigma_px, e.mask)
        if fr < rb_cfg.get("flank_ratio_max", rb.DEFAULTS["flank_ratio_max"]):
            reasons["dipole_flank"] = reasons.get("dipole_flank", 0) + 1
            continue
        fit["mf_snr"] = ref["snr"]
        fit["cand"] = {k: c[k] for k in ("x", "y", "pa_rad", "L_frt", "snr_frt")}
        fits_out.append(fit)
    if verbose:
        print(f"  exp{e.image_index}: {len(cands)} FRT cands ({t_frt:.0f}s), "
              f"{len(fits_out)} pass fit+RB; rejects: {reasons}")
    return fits_out


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
