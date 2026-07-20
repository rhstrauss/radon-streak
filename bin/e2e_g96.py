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
from streakradon.pipeline import process_exposure          # noqa: E402
from streakradon import rb                                 # noqa: E402
from streakradon.hldet_io import fit_to_row, write_hldet   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MTT = "/astro/store/shire/rstrau/heliolinx/bin/make_trailed_tracklets"
CONFIG_DIR = "/astro/store/shire/rstrau/catalina/config_files"

# injected movers: (rate deg/day, pa deg, mag) -- rates span the trailed regime
MOVERS = [
    (20.0, 45.0, 18.0),    # 25 px trail, unambiguous
    (10.0, 160.0, 18.5),   # 12 px trail, near the short end
    (50.0, 100.0, 18.5),   # 62 px trail, fast
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
        x0 = rng.uniform(900, 4300)
        y0 = rng.uniform(900, 4300)
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
    per_exp = [process_exposure(e, cfg, survey="g96") for e in exps]
    keep = rb.repetition_filter(per_exp, [e.mid_mjd for e in exps])
    rows = []
    for e, fits_i, keep_i in zip(exps, per_exp, keep):
        for j, (fit, k) in enumerate(zip(fits_i, keep_i)):
            if not k:
                continue
            rows.append(fit_to_row(fit, e.mid_mjd, f"{e.idbase}_c{j:03d}", e.band,
                                   e.obscode, image=e.image_index,
                                   qual=rb.det_qual(fit["mf_snr"], fit["chi2r"])))
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
           "-earth", os.path.join(CONFIG_DIR, "Earth1day2020s_02a.csv"),
           "-obscode", os.path.join(CONFIG_DIR, "ObsCodes.html"),
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

    # verify: read tracklets + trk2det + pairdets, match against truths
    trk = [ln.split(",") for ln in open(f"{out}/tracklets.csv").read().strip().split("\n")[1:]]
    t2d = [ln.split(",") for ln in open(f"{out}/trk2det.csv").read().strip().split("\n")[1:]]
    pd = open(f"{out}/pairdets.csv").read().strip().split("\n")[1:]
    print(f"{len(trk)} tracklets from {len(rows)} detections")
    from collections import defaultdict
    members = defaultdict(list)
    for trknum, detnum in t2d:
        members[int(trknum)].append(int(detnum))
    ok = 0
    for t in truths:
        found = False
        for tr in trk:
            # tracklets.csv: trailed output; RA/Dec of first point + trknum at end
            trknum = int(tr[-1])
            dets = members[trknum]
            if len(dets) < 3:
                continue
            # check all member dets lie near the truth track (pixel space)
            good = 0
            for d in dets:
                f = pd[d].split(",")
                mjd, ra, dec = float(f[0]), float(f[1]), float(f[2])
                x, y = exps[0].wcs.world_to_pixel_values(ra, dec)
                dt = mjd - (mjds[0] + exptime / 2 / 86400.0)
                rate_px = t["rate"] * 3600.0 / pixscale
                tx = t["x0"] + rate_px * dt * np.cos(np.radians(t["pa"]))
                ty = t["y0"] + rate_px * dt * np.sin(np.radians(t["pa"]))
                if np.hypot(float(x) - tx, float(y) - ty) < 8:
                    good += 1
            if good >= 3 and good == len(dets):
                print(f"  mover rate {t['rate']:.0f}: PURE tracklet #{trknum} "
                      f"with {len(dets)} points")
                found = True
                ok += 1
                break
            elif good >= 3:
                print(f"  mover rate {t['rate']:.0f}: tracklet #{trknum} has "
                      f"{good}/{len(dets)} matching points (impure)")
        if not found:
            print(f"  mover rate {t['rate']:.0f}: NOT recovered as pure >=3-pt tracklet")
    print(f"\nE2E: {ok}/{len(truths)} movers recovered as pure tracklets "
          f"-> {'PASS' if ok == len(truths) else 'FAIL'}")
    sys.exit(0 if ok == len(truths) else 1)


if __name__ == "__main__":
    main()
