#!/usr/bin/env python
"""P5: end-to-end test — inject a consistent mover across the 4-exposure G96
sequence, detect with the full pipeline, write hldet CSV, run the real
heliolinx make_trailed_tracklets, and assert the injected object is recovered
as a single pure tracklet.

This is the acceptance test that our measured trail_len/trail_PA pass the
linker's pair-consistency gates (trailpred vs trail_len within siglenscale,
PA within sigpascale/trail_len) -- metric accuracy, not just detection.
"""
import os
import pickle
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from streakradon.adapters.g96 import build_exposures       # noqa: E402
from streakradon.inject import inject_trail                # noqa: E402
from streakradon.frt_driver import detect_streaks          # noqa: E402
from streakradon.mf_snr import refine_candidate            # noqa: E402
from streakradon.trail_fit import fit_trail                # noqa: E402
from streakradon.pipeline import rb_config                 # noqa: E402
from streakradon import rb                                 # noqa: E402
from streakradon.hldet_io import fit_to_row, write_hldet   # noqa: E402


def detect_in_window(e, x, y, cfg, hw=110):
    """Windowed detection+measurement around an expected position (the injected
    mover). Returns the best surviving fit dict or None. Uses the SAME detect ->
    refine -> Veres fit -> RB chain as the full pipeline, just localized -- the
    full-frame detection realism is validated separately (ZTF head-to-head)."""
    rbc = rb_config(cfg)
    xi, yi = int(round(x)), int(round(y))
    win = e.white[yi - hw:yi + hw, xi - hw:xi + hw]
    if win.shape != (2 * hw, 2 * hw):
        return None
    cands = detect_streaks(win, e.psf_sigma_px, tile=2 * hw, overlap=0,
                           min_length=cfg["detect"].get("min_length", 8),
                           threshold=cfg["detect"].get("frt_threshold", 5.0))
    best = None
    for c in cands:
        gx, gy = c["x"] + xi - hw, c["y"] + yi - hw
        if np.hypot(gx - x, gy - y) > 15:
            continue
        ref = refine_candidate(e.white, gx, gy, c["pa_rad"], e.psf_sigma_px)
        if ref["snr"] < rbc["mf_snr_min"]:
            continue
        fit = fit_trail(e.diff, e.mask, e.wcs, ref["x"], ref["y"], e.psf_sigma_px,
                        e.magzp, theta0=ref["pa_rad"], h0=ref["L"] / 2.0)
        cc = dict(snr=ref["snr"], pa_px_deg=np.degrees(ref["pa_rad"]), near_bad_col=False)
        ok, _ = rb.passes_rb(fit, cc, imshape=e.diff.shape, cfg=rbc, survey="g96")
        if ok and (best is None or ref["snr"] > best["mf_snr"]):
            fit["mf_snr"] = ref["snr"]
            best = fit
    return best

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Machine-specific paths (arnor defaults); override via environment on any host.
MTT = os.environ.get("MAKE_TRAILED_TRACKLETS",
                     "/astro/store/shire/rstrau/heliolinx/bin/make_trailed_tracklets")
CONFIG_DIR = os.environ.get("HELIO_CONFIG_DIR",
                            "/astro/store/shire/rstrau/catalina/config_files")
EARTH = os.environ.get("HELIO_EARTH", os.path.join(CONFIG_DIR, "Earth1day2020s_02a.csv"))
OBSCODES = os.environ.get("HELIO_OBSCODES", os.path.join(CONFIG_DIR, "ObsCodes.html"))

# injected movers: (rate deg/day, pa deg, mag) -- rates span the trailed regime.
# Bright enough (mag ~17) to clear the frozen mf_snr_min=11 threshold at all four
# epochs; the point of this test is the linker's pair-consistency gate, not
# sensitivity (that is measure_efficiency's job).
MOVERS = [
    (40.0, 45.0, 17.0),    # 49 px trail
    (55.0, 160.0, 17.0),   # 68 px trail
    (50.0, 100.0, 17.0),   # 62 px trail
]


def main():
    import yaml
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "g96.yaml")))
    stack = pickle.load(open(os.path.join(ROOT, "work", "g96_stack.pkl"), "rb"))
    magzp = float(stack["headers"][0].get("MAGZP", 28.0))
    pixscale = cfg.get("pixscale_arcsec", 1.52)
    exptime = cfg.get("exptime_s", 30.0)
    mjds = [float(h["MJD"]) for h in stack["headers"]]

    reg = stack["reg"]
    stack = dict(stack)
    stack["reg"] = reg.copy()
    rng = np.random.default_rng(11)
    truths = []
    for k, (rate, pa_deg, mag) in enumerate(MOVERS):
        pa = np.radians(pa_deg)
        rate_px = rate * 3600.0 / pixscale          # px/day
        L_px = rate * 3600.0 / pixscale * exptime / 86400.0
        # keep the whole 4-exposure track on-frame: total motion = rate_px * span
        span_day = mjds[-1] - mjds[0]
        travel = rate_px * span_day
        cx = 2640.0 - 0.5 * travel * np.cos(pa)
        cy = 2640.0 - 0.5 * travel * np.sin(pa)
        x0 = float(np.clip(cx, 400, 4880))
        y0 = float(np.clip(cy, 400, 4880))
        for i, mjd in enumerate(mjds):
            dt = mjd - mjds[0]
            x = x0 + rate_px * dt * np.cos(pa)
            y = y0 + rate_px * dt * np.sin(pa)
            inject_trail(stack["reg"][i], x, y, pa, L_px, mag, magzp,
                         stack["psf_sigma"])
        truths.append(dict(rate=rate, pa=pa_deg, mag=mag, x0=x0, y0=y0, L_px=L_px))
        print(f"mover {k}: rate {rate} deg/day, PA {pa_deg}, mag {mag}, "
              f"L {L_px:.1f} px, start ({x0:.0f},{y0:.0f})")

    exps = build_exposures(stack, cfg, verbose=False)
    pixscale = cfg.get("pixscale_arcsec", 1.52)
    rows = []
    n_det = 0
    for e in exps:
        for k, t in enumerate(truths):
            dt = e.mid_mjd - (mjds[0])  # mid vs frame-0 header (mid, resolved)
            rate_px = t["rate"] * 3600.0 / pixscale
            x = t["x0"] + rate_px * dt * np.cos(np.radians(t["pa"]))
            y = t["y0"] + rate_px * dt * np.sin(np.radians(t["pa"]))
            fit = detect_in_window(e, x, y, cfg)
            if fit is None:
                print(f"  exp{e.image_index} mover{k}: not detected at ({x:.0f},{y:.0f})")
                continue
            n_det += 1
            rows.append(fit_to_row(fit, e.mid_mjd, f"mover{k}_e{e.image_index}",
                                   e.band, e.obscode, image=e.image_index,
                                   qual=rb.det_qual(fit["mf_snr"], fit["chi2r"])))
    print(f"detected {n_det} mover measurements across 4 exposures")
    os.makedirs(os.path.join(ROOT, "work", "e2e"), exist_ok=True)
    det_csv = os.path.join(ROOT, "work", "e2e", "e2e_trails.csv")
    img_txt = os.path.join(ROOT, "work", "e2e", "e2e_imgs.txt")
    write_hldet(det_csv, rows)
    with open(img_txt, "w") as f:
        for e in exps:
            cra, cdec = e.wcs.pixel_to_world_values(2640, 2640)
            f.write(f"{e.mid_mjd:.8f} {float(cra):.6f} {float(cdec):.6f} "
                    f"{e.obscode} {e.exptime_s:.1f}\n")
    print(f"{len(rows)} detections -> {det_csv}")

    out = os.path.join(ROOT, "work", "e2e")
    cmd = [MTT, "-dets", det_csv, "-imgs", img_txt,
           "-colformat", os.path.join(ROOT, "config", "hldet_colformat01.txt"),
           "-earth", EARTH,
           "-obscode", OBSCODES,
           "-exptime", str(exptime), "-maxvel", "100.0", "-imrad", "2.0",
           "-maxGCR", "3.0", "-mintrkpts", "2",
           "-siglenscale", "0.5", "-sigpascale", "10.0",
           "-outimgs", f"{out}/outim.txt", "-pairdets", f"{out}/pairdets.csv",
           "-tracklets", f"{out}/tracklets.csv", "-trk2det", f"{out}/trk2det.csv",
           "-forcerun"]
    print("running make_trailed_tracklets ...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-2500:])
    if r.returncode != 0:
        print(r.stderr[-2000:])
        sys.exit(1)

    # verify by idstring label: each injected detection is "mover{k}_e{i}"; a
    # pure tracklet for mover k has all members labelled mover{k}_.
    import csv
    from collections import defaultdict
    pd_rows = list(csv.reader(open(f"{out}/pairdets.csv")))
    pd_hdr = pd_rows[0]
    id_col = next(i for i, c in enumerate(pd_hdr) if "idstring" in c.lower() or c.lower() == "id")
    pd_ids = [r[id_col] for r in pd_rows[1:]]
    t2d = [ln.split(",") for ln in open(f"{out}/trk2det.csv").read().strip().split("\n")[1:]]
    members = defaultdict(list)
    for row in t2d:
        trknum, detnum = int(row[0]), int(row[-1])
        members[trknum].append(detnum)
    print(f"{len(members)} tracklets from {len(rows)} detections")
    ok = 0
    for k, t in enumerate(truths):
        lbl = f"mover{k}_"
        pure = None
        for trknum, dets in members.items():
            labels = [pd_ids[d] for d in dets]
            n_mine = sum(s.startswith(lbl) for s in labels)
            if n_mine >= 3 and n_mine == len(labels):
                pure = (trknum, len(labels))
                break
        if pure:
            print(f"  mover{k} (rate {t['rate']:.0f} deg/day, L {t['L_px']:.0f}px): "
                  f"PURE tracklet #{pure[0]} with {pure[1]} points")
            ok += 1
        else:
            got = max((sum(pd_ids[d].startswith(lbl) for d in dets)
                       for dets in members.values()), default=0)
            print(f"  mover{k} (rate {t['rate']:.0f}): NOT a pure >=3-pt tracklet "
                  f"(best {got} pts in one tracklet)")
    print(f"\nE2E: {ok}/{len(truths)} movers recovered as pure tracklets "
          f"-> {'PASS' if ok == len(truths) else 'FAIL'}")
    sys.exit(0 if ok == len(truths) else 1)


if __name__ == "__main__":
    main()
