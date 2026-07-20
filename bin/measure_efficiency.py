#!/usr/bin/env python
"""P4: injection efficiency + FP calibration for the G96 pipeline.

Fast path: injection happens into the REGISTERED frame 0 (before template
construction, so the LOO design is exercised), but since only frame 0 is
touched, exposure 0's template (median of frames 1-3) is FIXED -> compute it
once and re-difference in O(1) per round.

Modes:
  --grid      : completeness grid over (mag, L); random PA per injection
  --negated   : FP calibration -- run the chain on the NEGATED clean diff
  --clean     : baseline detections on the clean diff (real objects + residual FPs)

Outputs work/efficiency.csv (one row per injection: truth + recovery + errors)
and prints per-(mag,L) completeness.
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from streakradon.adapters.base import DiffExposure         # noqa: E402
from streakradon.inject import inject_trail, random_positions  # noqa: E402
from streakradon.pipeline import process_exposure          # noqa: E402
from streakradon.varmap import whiten                      # noqa: E402
from streakradon.template import loo_diffs                 # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_cfg():
    import yaml
    return yaml.safe_load(open(os.path.join(ROOT, "config", "g96.yaml")))


def build_exp0(stack, cfg, frame0=None):
    """DiffExposure for exposure 0, with optional replaced (injected) frame 0.
    Template from frames 1..N-1 only -- identical to loo_diffs for i=0."""
    reg, regm = stack["reg"], stack["regm"]
    if frame0 is None:
        frame0 = reg[0]
    sub = np.stack([frame0] + [reg[i] for i in range(1, reg.shape[0])])
    diffs, dmasks = loo_diffs(sub, regm)
    pp = cfg.get("preprocess", {})
    white, var, bg = whiten(diffs[0], dmasks[0], grid=pp.get("var_grid_px", 128))
    h = stack["headers"][0]
    exptime = float(cfg.get("exptime_s", 30.0))
    wcs = stack["ref_wcs"]
    pixscale = np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600.0
    return DiffExposure(
        diff=diffs[0], white=white, var=var, mask=dmasks[0], wcs=wcs,
        mid_mjd=float(h["MJD"]) + exptime / 2 / 86400.0,
        magzp=float(h.get("MAGZP", 28.0)), psf_sigma_px=stack["psf_sigma"],
        exptime_s=exptime, pixscale_arcsec=pixscale,
        obscode=cfg.get("obscode", "G96"), band=cfg.get("band", "G"),
        idbase="inj", image_index=0)


def match(truths, fits, rad_px=10.0, pa_tol=12.0):
    """Greedy nearest match of fits to truths. Returns list of fit-or-None."""
    out = [None] * len(truths)
    used = set()
    for i, t in enumerate(truths):
        bd, bj = 1e9, None
        for j, f in enumerate(fits):
            if j in used:
                continue
            d = np.hypot(f["x"] - t["x"], f["y"] - t["y"])
            dpa = abs(np.degrees(f["theta_px"] - t["pa_rad"])) % 180.0
            dpa = min(dpa, 180.0 - dpa)
            if d < rad_px and dpa < pa_tol and d < bd:
                bd, bj = d, j
        if bj is not None:
            out[i] = fits[bj]
            used.add(bj)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="work/g96_stack.pkl")
    ap.add_argument("--out", default="work/efficiency.csv")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--negated", action="store_true")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--mags", default="17.5,18.0,18.5,19.0,19.5,20.0,20.5,21.0")
    ap.add_argument("--lengths", default="5,10,20,40,70,100")
    ap.add_argument("--per-cell", type=int, default=8)
    ap.add_argument("--per-round", type=int, default=24)
    ap.add_argument("--baseline", default=None,
                    help="clean-run fits json to exclude real detections from FP counts")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_cfg()
    stack = pickle.load(open(os.path.join(ROOT, args.stack) if not
                             os.path.isabs(args.stack) else args.stack, "rb"))
    rng = np.random.default_rng(args.seed)

    if args.clean or args.negated:
        e = build_exp0(stack, cfg)
        if args.negated:
            e.diff = -e.diff
            e.white = -e.white
        fits = process_exposure(e, cfg, survey="g96")
        tag = "NEGATED" if args.negated else "CLEAN"
        print(f"[{tag}] {len(fits)} detections pass the full chain")
        for f in fits:
            print(f"  ({f['x']:7.1f},{f['y']:7.1f}) len {f['trail_len']:6.1f}\" "
                  f"PA {f['trail_PA']:6.1f} mag {f['mag']:5.2f} mf_snr {f['mf_snr']:5.1f} "
                  f"chi2r {f['chi2r']:4.2f}")
        import json
        out = args.out.replace(".csv", f"_{tag.lower()}.json")
        json.dump([{k: v for k, v in f.items() if k != "cand"} for f in fits],
                  open(os.path.join(ROOT, out), "w"), indent=1)
        print(f"wrote {out}")
        return

    mags = [float(m) for m in args.mags.split(",")]
    lengths = [float(v) for v in args.lengths.split(",")]
    cells = [(m, L) for m in mags for L in lengths]
    todo = []
    for (m, L) in cells:
        todo += [(m, L)] * args.per_cell
    rng.shuffle(todo)

    rows = []
    hdr = ("mag,L_px,pa_deg,x,y,recovered,mf_snr,fit_len_as,fit_pa,fit_mag,"
           "dcen_px,dlen_frac,dpa_deg,dmag")
    # WINDOWED injection-recovery: inject a well-separated batch into frame 0,
    # build the diff ONCE, then detect each injection in its own 400 px window.
    # This is the standard injection-recovery method and avoids both the full-
    # frame FRT cost and pyradon's per-tile num_iterations crowding limit (which
    # made whole-frame recovery of dense injections unfair).
    HW = 200
    n_round = int(np.ceil(len(todo) / args.per_round))
    print(f"{len(todo)} injections over {n_round} rounds of {args.per_round} (windowed)")
    from streakradon.frt_driver import detect_streaks
    from streakradon.mf_snr import refine_candidate
    from streakradon.trail_fit import fit_trail
    from streakradon import rb
    from streakradon.pipeline import rb_config
    rbc = rb_config(cfg)
    for r in range(n_round):
        batch = todo[r * args.per_round:(r + 1) * args.per_round]
        if not batch:
            break
        frame0 = stack["reg"][0].copy()
        pos = random_positions(frame0.shape, stack["regm"][0], len(batch),
                               min_sep=2 * HW + 40, margin=HW + 30, rng=rng)
        truths = []
        for (m, L), (x, y) in zip(batch, pos):
            pa = rng.uniform(0, np.pi)
            tr = inject_trail(frame0, x, y, pa, L, m,
                              float(stack["headers"][0].get("MAGZP", 28.0)),
                              stack["psf_sigma"])
            truths.append(tr)
        t0 = time.time()
        e = build_exp0(stack, cfg, frame0=frame0)
        pixscale = e.pixscale_arcsec
        nrec = 0
        for t in truths:
            xi, yi = int(round(t["x"])), int(round(t["y"]))
            win = e.white[yi - HW:yi + HW, xi - HW:xi + HW]
            wmask = e.mask[yi - HW:yi + HW, xi - HW:xi + HW]
            cands = detect_streaks(win, e.psf_sigma_px, tile=2 * HW, overlap=0,
                                   min_length=cfg["detect"].get("min_length", 8),
                                   threshold=cfg["detect"].get("frt_threshold", 5.0))
            best = None
            for c in cands:
                gx, gy = c["x"] + xi - HW, c["y"] + yi - HW
                if np.hypot(gx - t["x"], gy - t["y"]) > 12:
                    continue
                ref = refine_candidate(e.white, gx, gy, c["pa_rad"], e.psf_sigma_px)
                if ref["snr"] < rbc["mf_snr_min"]:
                    continue
                fit = fit_trail(e.diff, e.mask, e.wcs, ref["x"], ref["y"],
                                e.psf_sigma_px, e.magzp, theta0=ref["pa_rad"],
                                h0=ref["L"] / 2.0)
                cc = dict(snr=ref["snr"], pa_px_deg=np.degrees(ref["pa_rad"]),
                          near_bad_col=False)
                ok, _ = rb.passes_rb(fit, cc, imshape=e.diff.shape, cfg=rbc, survey="g96")
                if ok and (best is None or ref["snr"] > best[0]):
                    best = (ref["snr"], fit)
            if best is None:
                rows.append((t["mag"], t["L_px"], np.degrees(t["pa_rad"]), t["x"],
                             t["y"], 0) + (np.nan,) * 8)
            else:
                snr, f = best
                dcen = np.hypot(f["x"] - t["x"], f["y"] - t["y"])
                dlen = f["trail_len"] / (t["L_px"] * pixscale) - 1.0
                dpa = abs(np.degrees(f["theta_px"] - t["pa_rad"])) % 180
                dpa = min(dpa, 180 - dpa)
                rows.append((t["mag"], t["L_px"], np.degrees(t["pa_rad"]), t["x"],
                             t["y"], 1, snr, f["trail_len"], f["trail_PA"],
                             f["mag"], dcen, dlen, dpa, f["mag"] - t["mag"]))
                nrec += 1
        print(f"round {r+1}/{n_round}: {nrec}/{len(batch)} recovered ({time.time()-t0:.0f}s)")
    outp = os.path.join(ROOT, args.out)
    with open(outp, "w") as f:
        f.write("#" + hdr + "\n")
        for row in rows:
            f.write(",".join("%.4f" % v if isinstance(v, float) else str(v)
                             for v in row) + "\n")
    print(f"wrote {outp}")
    # summary table
    import collections
    agg = collections.defaultdict(lambda: [0, 0])
    for row in rows:
        agg[(row[0], row[1])][1] += 1
        agg[(row[0], row[1])][0] += row[5]
    print("\ncompleteness (mag rows x L cols):")
    print("mag\\L " + "".join(f"{int(L):>6d}" for L in lengths))
    for m in mags:
        line = f"{m:5.1f} "
        for L in lengths:
            k, n = agg[(m, L)]
            line += f"{k/n if n else 0:6.2f}"
        print(line)


if __name__ == "__main__":
    main()
