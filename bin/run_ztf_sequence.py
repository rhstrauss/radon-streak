#!/usr/bin/env python
"""Full streak_radon pipeline over one ZTF quadrant-visit sequence.

A "sequence" here is every exposure of the SAME (night, field, ccdid, qid) --
typically 2 same-band revisits, up to 4 across g+r. Running them together is not
cosmetic: it is what enables the cross-exposure repetition filter, which is the
main purity cut available on ZTF diffs (static subtraction artifacts recur at the
same sky position and PA; a real mover does not).

  scimrefdiffimg + mskimg xN -> bit-selective mask + whiten -> FRT per exposure
  -> dedup -> MF refine -> RB vetting -> repetition filter -> hldet CSV.

IDSTRINGS: heliolinx's hldet.idstring is char[20] and its reader
`stringncopy01(idstring, s, SHORTSTRINGLEN)` SILENTLY TRUNCATES to 19 chars --
it does not error. A natural ZTF id
(`ztf_20240901136065_000577_zr_c01_o_q1_scimrefdiffimg_c000`, 56 chars) would
therefore collapse to `ztf_20240901136065` for every detection in the exposure,
destroying idstring-based dedup and any join back to this catalog. So ids are
built to fit in 18 chars and still be unique and reversible:

    z 240901136065 1f 07
    | |            |  |
    | |            |  +-- candidate index within the exposure, base36 (3 chars)
    | |            +----- readout channel (ccdid-1)*4+(qid-1) = 0..63, hex (2)
    | +------------------ filefracday minus the century: YYMMDDffffff (12)
    +-------------------- survey tag

Usage:
  run_ztf_sequence.py --diffs 'dir/ztf_*_q1_scimrefdiffimg.fits.fz' \
      --out trails.csv [--imgs imgs.txt] [--config ztf] [--stamps s.npz]
"""
import argparse
import glob as globmod
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from streakradon.adapters.ztf import prepare_sequence           # noqa: E402
from streakradon.pipeline import process_exposure               # noqa: E402
from streakradon import rb                                     # noqa: E402
from streakradon.hldet_io import fit_to_row, write_hldet        # noqa: E402
from streakradon.config import load_cfg                        # noqa: E402

# ztf_<filefracday>_<field>_<filtercode>_c<ccdid>_o_q<qid>_scimrefdiffimg.fits.fz
FNAME_RE = re.compile(r"ztf_(\d{14})_(\d{6})_(z[gri])_c(\d{2})_o_q(\d)")

B36 = "0123456789abcdefghijklmnopqrstuvwxyz"

MAX_IDSTRING = 19          # heliolinx SHORTSTRINGLEN 20, minus the NUL


def _b36(n, width=3):
    """Base36, fixed width. Saturates rather than overflowing the id field."""
    n = int(n)
    cap = 36 ** width - 1
    if n > cap:
        n = cap
    s = ""
    while n:
        n, r = divmod(n, 36)
        s = B36[r] + s
    return s.rjust(width, "0")


def idstring_for(diff_path, cand_index, exp=None):
    """<=18-char unique, reversible detection id. See the module docstring."""
    m = FNAME_RE.search(os.path.basename(diff_path))
    if m:
        ffd, _field, _filt, ccd, qid = m.groups()
        rc = (int(ccd) - 1) * 4 + (int(qid) - 1)
        ident = "z%s%02x%s" % (ffd[2:], rc, _b36(cand_index))
    else:
        # Unparseable name (hand-staged file): fall back to a truncated basename
        # plus the candidate index, still inside the 19-char budget.
        stem = os.path.basename(diff_path).split(".")[0]
        ident = (stem[: MAX_IDSTRING - 3] + _b36(cand_index))
    assert len(ident) <= MAX_IDSTRING, (ident, len(ident))
    return ident


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ztf",
                    help="survey name (ztf) or a path to a YAML config")
    ap.add_argument("--diffs", required=True,
                    help="glob over the sequence's scimrefdiffimg files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgs", default=None, help="write make_trailed_tracklets image log")
    ap.add_argument("--json", default=None, help="dump surviving fits as JSON")
    ap.add_argument("--mf-snr-min", type=float, default=None)
    ap.add_argument("--stamps", default=None)
    ap.add_argument("--stamp-size", type=int, default=64)
    ap.add_argument("--stamp-plane", default="diff", choices=["diff", "white"])
    args = ap.parse_args()

    # Loud loader: a missing config raises rather than silently reverting the
    # tuned detect parameters to the permissive library defaults.
    cfg = load_cfg(args.config)
    if args.mf_snr_min is not None:
        cfg.setdefault("detect", {})["mf_snr_min"] = args.mf_snr_min

    paths = sorted(globmod.glob(args.diffs))
    if not paths:
        sys.exit(f"no diffs match {args.diffs}")
    print(f"preparing {len(paths)} exposures (config={args.config})...")
    exps = prepare_sequence(paths, cfg)

    per_exp = [process_exposure(e, cfg, survey="ztf") for e in exps]

    keep = rb.repetition_filter(per_exp, [e.mid_mjd for e in exps],
                                cfg=cfg.get("vet"))
    rows = []
    n_rep = 0
    for e, fits_i, keep_i in zip(exps, per_exp, keep):
        for j, (fit, k) in enumerate(zip(fits_i, keep_i)):
            if not k:
                n_rep += 1
                continue
            q = rb.det_qual(fit["mf_snr"], fit["chi2r"])
            rows.append(fit_to_row(fit, e.mid_mjd,
                                   idstring_for(e.meta["path"], j, e),
                                   e.band, e.obscode, image=e.image_index, qual=q))
    print(f"repetition filter removed {n_rep}; writing {len(rows)} detections")
    write_hldet(args.out, rows)

    if args.stamps:
        from streakradon.stamps import write_field_stamps
        ns = write_field_stamps(args.stamps, exps, per_exp, keep,
                                size=args.stamp_size, plane=args.stamp_plane)
        print(f"wrote {ns} stamps ({args.stamp_size}px {args.stamp_plane})")
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
