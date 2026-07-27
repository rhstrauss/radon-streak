#!/usr/bin/env python
"""Injection recall gate: can the BLIND FULL-FRAME pipeline still find trails?

Prints `RECALL=<fraction>` for the staged survey driver to parse.

This exists because a defect went unseen by every existing test: the acceptance
test (bin/e2e_g96.py) uses WINDOWED detection around known positions, where the
failure mode cannot occur, and bin/regress_a000001.py scores one bright object
in a 512^2 cutout. Blind full-frame detection -- the actual survey path -- was
never exercised, and was measured recovering 0 of 23 bright injected trails.
"""
import argparse, glob, os, sys
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from streakradon.adapters.g96 import prepare_sequence   # noqa: E402
from streakradon.pipeline import process_exposure       # noqa: E402
from streakradon.inject import inject_trail             # noqa: E402

PIX, EXPTIME = 1.52, 30.0
MOVERS = [(40.0, 45.0, 17.0), (55.0, 160.0, 17.0), (50.0, 100.0, 17.0),
          (35.0, 210.0, 17.2), (60.0, 300.0, 17.2), (45.0, 130.0, 17.4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--tol-px", type=float, default=25.0)
    ap.add_argument("--fast", type=int, default=1)
    a = ap.parse_args()
    paths = sorted(glob.glob(a.frames))
    if len(paths) < 3:
        print("RECALL=nan"); print(f"need >=3 frames, got {len(paths)}"); return 2
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "g96.yaml")))
    exps = prepare_sequence(paths, cfg, verbose=False)

    rng = np.random.default_rng(11)
    starts = [(rng.uniform(1200, 4000), rng.uniform(1200, 4000)) for _ in MOVERS]
    truth = []
    for e in exps:
        per = []
        for (rate, pa, mag), (x0, y0) in zip(MOVERS, starts):
            L = rate * 3600.0 / PIX * EXPTIME / 86400.0
            drift = rate * 3600.0 / PIX * (e.mid_mjd - exps[0].mid_mjd)
            x = x0 + drift * np.cos(np.radians(pa))
            y = y0 + drift * np.sin(np.radians(pa))
            if not (60 < x < e.diff.shape[1] - 60 and 60 < y < e.diff.shape[0] - 60):
                continue
            inject_trail(e.diff, x, y, np.radians(pa), L, mag, e.magzp, e.psf_sigma_px)
            inject_trail(e.white, x, y, np.radians(pa), L, mag, e.magzp, e.psf_sigma_px)
            per.append((x, y))
        truth.append(per)

    n_t = sum(len(t) for t in truth)
    rec = n_det = 0
    for e, tl in zip(exps, truth):
        try:
            fits = process_exposure(e, cfg, verbose=False, fast=bool(a.fast))
        except TypeError:                       # older signature без fast=
            fits = process_exposure(e, cfg, verbose=False)
        n_det += len(fits)
        for (tx, ty) in tl:
            if any(np.hypot(f["x"] - tx, f["y"] - ty) <= a.tol_px for f in fits):
                rec += 1
    print(f"RECALL={rec / max(n_t, 1):.4f}")
    print(f"injected {n_t}, recovered {rec}, total vetted detections {n_det} "
          f"across {len(exps)} exposures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
