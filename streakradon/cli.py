#!/usr/bin/env python
"""streak-radon: unified command-line interface.

Subcommands
-----------
  detect           one already-differenced FITS  -> hldet CSV
                   (--survey generic|ztf|atlas)
  detect-sequence  an N-frame un-differenced sequence (CSS/g96: within-sequence
                   leave-one-out differencing) -> hldet CSV
  demo             self-contained synthetic sequence (no external data): inject,
                   detect, measure, optionally link with make_trailed_tracklets
  selftest         run the unit tests

Every path produces a 15-column hldet_colformat01 CSV for heliolinx
make_trailed_tracklets. See README.md for a full tutorial.
"""
import argparse
import json
import os
import sys
import time

from .config import load_cfg
from .hldet_io import fit_to_row, write_hldet
from . import rb
from .pipeline import process_exposure


# --------------------------------------------------------------------------- #
def _write_imgs_log(path, exposures):
    with open(path, "w") as f:
        for e in exposures:
            cra, cdec = e.wcs.pixel_to_world_values(e.diff.shape[1] / 2,
                                                    e.diff.shape[0] / 2)
            f.write(f"{e.mid_mjd:.8f} {float(cra):.6f} {float(cdec):.6f} "
                    f"{e.obscode} {e.exptime_s:.1f}\n")


def _rows_from_exposure(e, fits_list, survey):
    rows = []
    for j, fit in enumerate(fits_list):
        q = rb.det_qual(fit["mf_snr"], fit["chi2r"])
        rows.append(fit_to_row(fit, e.mid_mjd, f"{e.idbase}_c{j:03d}",
                               e.band, e.obscode, image=e.image_index, qual=q))
    return rows


# --------------------------------------------------------------------------- #
def cmd_detect(args):
    survey = args.survey
    cfg = load_cfg(args.config or survey)
    if args.mf_snr_min is not None:
        cfg.setdefault("detect", {})["mf_snr_min"] = args.mf_snr_min

    t0 = time.time()
    if survey == "ztf":
        from .adapters.ztf import prepare_exposure
        e = prepare_exposure(args.diff, args.mask, args.sci, cfg, idbase=args.idbase)
    elif survey == "atlas":
        from .adapters.atlas import prepare_stamp
        e = prepare_stamp(args.diff, cfg, idbase=args.idbase)
    else:  # generic
        from .adapters.generic import prepare_diff
        e = prepare_diff(args.diff, args.mask, cfg, idbase=args.idbase)

    fits_out = process_exposure(e, cfg, survey=survey, verbose=not args.quiet)
    rows = _rows_from_exposure(e, fits_out, survey)
    write_hldet(args.out, rows)
    if args.imgs:
        _write_imgs_log(args.imgs, [e])
    if args.json:
        json.dump([{k: v for k, v in f.items() if k != "cand"} for f in fits_out],
                  open(args.json, "w"), indent=1)
    print(f"{len(rows)} detections in {time.time()-t0:.0f}s -> {args.out}")


def cmd_detect_sequence(args):
    import glob as globmod
    cfg = load_cfg(args.config or "g96")
    if args.mf_snr_min is not None:
        cfg.setdefault("detect", {})["mf_snr_min"] = args.mf_snr_min
    from .adapters.g96 import prepare_sequence

    paths = sorted(globmod.glob(args.frames)) if args.frames else []
    if not paths:
        sys.exit(f"no frames match {args.frames!r}")
    print(f"preparing {len(paths)} exposures ...")
    exps = prepare_sequence(paths, cfg, qa_dir=args.qa)
    per_exp = [process_exposure(e, cfg, survey=cfg.get("survey", "g96"),
                                verbose=not args.quiet) for e in exps]
    keep = rb.repetition_filter(per_exp, [e.mid_mjd for e in exps], cfg=None)
    rows, n_rep = [], 0
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
        _write_imgs_log(args.imgs, exps)
    print(f"wrote {args.out}")


def cmd_demo(args):
    from .synthetic import build_sequence
    from .frt_driver import detect_streaks
    from .mf_snr import refine_candidate
    from .trail_fit import fit_trail
    from .pipeline import rb_config
    import numpy as np

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    cfg = load_cfg("generic")
    cfg.setdefault("detect", {})["mf_snr_min"] = args.mf_snr_min
    # small demo frames -> single-tile FRT
    exps, truth = build_sequence(n_epochs=args.n_epochs, shape=(args.size, args.size),
                                 mag=args.mag, seed=args.seed)
    cfg["detect"]["tile"] = args.size
    cfg["detect"]["overlap"] = 0

    print(f"synthetic sequence: {args.n_epochs} epochs, {args.size}x{args.size}px, "
          f"mover mag {truth['mag']}, trail {truth['trail_len_arcsec']:.1f}\" "
          f"({truth['L_px']:.0f}px), rate {truth['rate_deg_day']:.0f} deg/day")

    rows = []
    for e in exps:
        fits_i = process_exposure(e, cfg, survey="generic", verbose=not args.quiet)
        # associate the strongest surviving fit with the injected mover (truth)
        tx, ty = truth["positions"][e.image_index]
        best, best_d = None, 1e9
        for fit in fits_i:
            d = np.hypot(fit["x"] - tx, fit["y"] - ty)
            if d < 20 and fit["mf_snr"] > (best["mf_snr"] if best else 0):
                best, best_d = fit, d
        tag = "MISS"
        if best is not None:
            tag = (f"SNR {best['mf_snr']:.1f}  len {best['trail_len']:.1f}\" "
                   f"(truth {truth['trail_len_arcsec']:.1f})  "
                   f"center off {best_d:.1f}px  chi2r {best['chi2r']:.2f}")
        print(f"  epoch {e.image_index}: {len(fits_i)} pass vetting; mover -> {tag}")
        for j, fit in enumerate(fits_i):
            q = rb.det_qual(fit["mf_snr"], fit["chi2r"])
            rows.append(fit_to_row(fit, e.mid_mjd, f"synth_e{e.image_index}_c{j:03d}",
                                   e.band, e.obscode, image=e.image_index, qual=q))

    det_csv = os.path.join(out_dir, "demo_trails.csv")
    img_txt = os.path.join(out_dir, "demo_imgs.txt")
    write_hldet(det_csv, rows)
    _write_imgs_log(img_txt, exps)
    print(f"\n{len(rows)} detections -> {det_csv}")

    if args.mosaic:
        _demo_mosaic(exps, truth, os.path.join(out_dir, "demo_mosaic.png"))
        print(f"QA mosaic -> {os.path.join(out_dir, 'demo_mosaic.png')}")

    if args.link:
        _demo_link(det_csv, img_txt, out_dir, args)


def _demo_mosaic(exps, truth, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    n = len(exps)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.2))
    axes = np.atleast_1d(axes)
    for ax, e in zip(axes, exps):
        w = e.white
        lo, hi = np.percentile(w[np.isfinite(w)], [5, 99.5])
        ax.imshow(w, vmin=lo, vmax=hi, cmap="gray", origin="lower")
        tx, ty = truth["positions"][e.image_index]
        ax.plot(tx, ty, "o", mfc="none", mec="C1", ms=22, mew=1.5)
        ax.set_title(f"epoch {e.image_index}  (whitened diff)")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("streak_radon demo: injected fast mover (circled) across a "
                 "synthetic difference-image sequence")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def _demo_link(det_csv, img_txt, out_dir, args):
    import subprocess
    mtt = args.mtt or os.environ.get("MAKE_TRAILED_TRACKLETS")
    earth = args.earth or os.environ.get("HELIO_EARTH")
    obscodes = args.obscodes or os.environ.get("HELIO_OBSCODES")
    colformat = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "config", "hldet_colformat01.txt")
    if not (mtt and earth and obscodes):
        print("\n[link] skipped: need --mtt, --earth, --obscodes (or env "
              "MAKE_TRAILED_TRACKLETS / HELIO_EARTH / HELIO_OBSCODES).")
        return
    cmd = [mtt, "-dets", det_csv, "-imgs", img_txt, "-colformat", colformat,
           "-earth", earth, "-obscode", obscodes, "-exptime", "30.0",
           "-maxvel", "100.0", "-imrad", "2.0", "-maxGCR", "5.0", "-mintrkpts", "2",
           "-siglenscale", "0.5", "-sigpascale", "10.0",
           "-outimgs", f"{out_dir}/demo_outim.txt", "-pairdets", f"{out_dir}/demo_pairdets.csv",
           "-tracklets", f"{out_dir}/demo_tracklets.csv", "-trk2det", f"{out_dir}/demo_trk2det.csv",
           "-forcerun"]
    print("\n[link] running make_trailed_tracklets ...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print(r.stderr[-1500:])
        return
    try:
        t2d = open(f"{out_dir}/demo_trk2det.csv").read().strip().split("\n")[1:]
        from collections import defaultdict
        members = defaultdict(list)
        for ln in t2d:
            c = ln.split(",")
            members[int(c[0])].append(int(c[-1]))
        print(f"[link] {len(members)} tracklet(s) formed "
              f"(largest {max((len(v) for v in members.values()), default=0)} points)")
    except Exception as ex:
        print(f"[link] could not parse trk2det: {ex}")


def cmd_selftest(args):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(here, "tests", "test_units.py")])
    sys.exit(r.returncode)


# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        prog="streak-radon",
        description="FRT-based trailed-source detection for heliolinx "
                    "(FRT finds, Veres fit measures).")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="detect on one already-differenced FITS")
    d.add_argument("--survey", default="generic", choices=["generic", "ztf", "atlas"])
    d.add_argument("--config", default=None, help="config path or survey name (default: --survey)")
    d.add_argument("--diff", required=True, help="difference-image FITS")
    d.add_argument("--mask", default=None, help="bad-pixel mask FITS (nonzero=bad)")
    d.add_argument("--sci", default=None, help="(ztf) science-frame FITS for exact timing")
    d.add_argument("--out", required=True, help="output hldet CSV")
    d.add_argument("--imgs", default=None, help="also write a make_trailed_tracklets image log")
    d.add_argument("--json", default=None, help="dump surviving fits as JSON")
    d.add_argument("--idbase", default=None, help="idstring prefix (default: from filename)")
    d.add_argument("--mf-snr-min", type=float, default=None, help="override detect.mf_snr_min")
    d.add_argument("--quiet", action="store_true")
    d.set_defaults(func=cmd_detect)

    s = sub.add_parser("detect-sequence", help="detect over an un-differenced N-frame sequence (CSS/g96)")
    s.add_argument("--frames", required=True, help="glob of the sequence frames (e.g. 'seq/*.arch.fz')")
    s.add_argument("--config", default=None, help="config path or survey name (default: g96)")
    s.add_argument("--out", required=True, help="output hldet CSV")
    s.add_argument("--imgs", default=None, help="also write a make_trailed_tracklets image log")
    s.add_argument("--qa", default=None, help="directory for preprocessing QA PNGs")
    s.add_argument("--mf-snr-min", type=float, default=None)
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_detect_sequence)

    m = sub.add_parser("demo", help="self-contained synthetic detect (+optional link)")
    m.add_argument("--out-dir", default="demo_out")
    m.add_argument("--n-epochs", type=int, default=4)
    m.add_argument("--size", type=int, default=800, help="frame size (px)")
    m.add_argument("--mag", type=float, default=18.5)
    m.add_argument("--mf-snr-min", type=float, default=8.0)
    m.add_argument("--seed", type=int, default=7)
    m.add_argument("--mosaic", action="store_true", help="write a QA mosaic PNG")
    m.add_argument("--link", action="store_true", help="also run make_trailed_tracklets")
    m.add_argument("--mtt", default=None, help="path to make_trailed_tracklets (or env MAKE_TRAILED_TRACKLETS)")
    m.add_argument("--earth", default=None, help="Earth ephemeris CSV (or env HELIO_EARTH)")
    m.add_argument("--obscodes", default=None, help="ObsCodes.html (or env HELIO_OBSCODES)")
    m.add_argument("--quiet", action="store_true")
    m.set_defaults(func=cmd_demo)

    t = sub.add_parser("selftest", help="run unit tests")
    t.set_defaults(func=cmd_selftest)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
