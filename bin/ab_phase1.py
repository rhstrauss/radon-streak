#!/usr/bin/env python
"""Phase-1 A/B: legacy (stock pyradon subtract + Veres fit) vs fast path
(capsule suppression + rough fit) on a REAL G96 4-visit sequence.

Reports per-exposure runtime and, for every detection matched between the two
runs, the RA/Dec, trail_len and trail_PA agreement -- the quantities heliolinc's
tracklet gates actually consume.
"""
import os, sys, time, argparse
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from streakradon.adapters.g96 import prepare_sequence          # noqa: E402
from streakradon.pipeline import process_exposure              # noqa: E402
from streakradon.fastmask import (install_fast_suppression,    # noqa: E402
                                  uninstall_fast_suppression)

DEFAULT = ("/astro/store/shire/rstrau/css_pds_pilot/data/N25057_20240501/"
           "G96_20240501_2B_N25057_01_000?.arch.fz")


def run(exps, cfg, fast, label):
    t0 = time.time()
    out = []
    for e in exps:
        out.append(process_exposure(e, cfg, verbose=False, fast=fast))
    dt = time.time() - t0
    n = sum(len(f) for f in out)
    print(f"  {label:22s} {dt:7.1f}s   {n:4d} detections "
          f"({dt/max(len(exps),1):.1f}s/exposure)")
    return out, dt, n


def match(a, b, tol_arcsec=3.0):
    """Match detections between runs by sky position."""
    pairs = []
    for fa in a:
        best, bd = None, 1e9
        for fb in b:
            d = np.hypot((fa["ra"] - fb["ra"]) * np.cos(np.radians(fa["dec"])),
                         fa["dec"] - fb["dec"]) * 3600.0
            if d < bd:
                bd, best = d, fb
        if best is not None and bd <= tol_arcsec:
            pairs.append((fa, best, bd))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=DEFAULT)
    a = ap.parse_args()
    import glob
    paths = sorted(glob.glob(a.frames))
    if len(paths) < 2:
        sys.exit(f"need >=2 frames, matched {len(paths)}")
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "g96.yaml")))
    print(f"preparing {len(paths)} exposures ...")
    exps = prepare_sequence(paths, cfg, verbose=False)

    print("\nA/B (same exposures, same config):")
    uninstall_fast_suppression()
    cfg_legacy = dict(cfg); cfg_legacy["detect"] = dict(cfg["detect"]); \
        cfg_legacy["detect"]["fast_suppress"] = False
    legacy, t_leg, n_leg = run(exps, cfg_legacy, False, "LEGACY (stock+Veres)")

    install_fast_suppression()
    fastc = dict(cfg); fastc["detect"] = dict(cfg["detect"]); \
        fastc["detect"]["fast_suppress"] = True
    fastr, t_fast, n_fast = run(exps, fastc, True, "FAST (capsule+rough)")

    print(f"\n  SPEEDUP: {t_leg/t_fast:.1f}x")

    # agreement on matched detections
    dl, dpa, dpos = [], [], []
    for fa_list, fb_list in zip(legacy, fastr):
        for fa, fb, d in match(fa_list, fb_list):
            dpos.append(d)
            dl.append(abs(fa["trail_len"] - fb["trail_len"]) /
                      max(fa["trail_len"], 1e-6) * 100)
            dd = abs(fa["trail_PA"] - fb["trail_PA"]) % 360
            dpa.append(min(dd, 360 - dd))
    print(f"\n  matched detections: {len(dpos)} / {min(n_leg, n_fast)}")
    if dpos:
        print(f"    position agreement : median {np.median(dpos):.2f}\"  max {np.max(dpos):.2f}\"")
        print(f"    trail_len agreement: median {np.median(dl):.1f}%  (gate: siglenscale 50%)")
        print(f"    trail_PA agreement : median {np.median(dpa):.2f} deg")


if __name__ == "__main__":
    main()
