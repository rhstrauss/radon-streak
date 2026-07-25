#!/usr/bin/env python
"""Diagnose the native-FRT 6/8 injected-streak recovery gap.

Determines at WHICH stage each injected streak is lost:
  (a) never produced as a raw _fold_detect peak,
  (b) discarded by the top_per_level cap,
  (c) merged away by dedup_candidates,
  (d) present but scored a "miss" only because the match test used
      centre-to-centre distance.

(d) is a real suspect: the FRT harvests peaks per DYADIC BLOCK, so a long streak
is often reported as a sub-segment whose midpoint is displaced from the true
centre by up to L/2 -- a 450 px streak found in a 256 px block can have its
centre ~100 px away while describing the same line perfectly. The physically
correct association is perpendicular distance to the line plus along-track
overlap (what bin/regress_a000001.py uses), not centre distance.
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from streakradon import frt as F                    # noqa: E402
from streakradon.frt_driver import detect_streaks, dedup_candidates  # noqa: E402


def make_tile(size=1024, n=8, seed=3):
    """Same construction as bin/ab_backends.py so numbers are comparable."""
    rng = np.random.default_rng(seed)
    img = rng.normal(0, 1, (size, size))
    truth = []
    for _ in range(n):
        x0, y0 = rng.integers(80, size - 120, 2)
        ang = rng.uniform(0, np.pi)
        L = rng.uniform(150, 450)
        x1, y1 = float(x0), float(y0)
        x2, y2 = x0 + L * np.cos(ang), y0 + L * np.sin(ang)
        truth.append(dict(x1=x1, y1=y1, x2=x2, y2=y2, L=L, ang=ang,
                          cx=0.5 * (x1 + x2), cy=0.5 * (y1 + y2)))
        for t in np.linspace(0, 1, 800):
            x = int(x1 + (x2 - x1) * t); y = int(y1 + (y2 - y1) * t)
            if 1 <= y < size - 1 and 1 <= x < size - 1:
                img[y - 1:y + 2, x - 1:x + 2] += 6
    return img, truth


def line_assoc(t, c, perp_tol=12.0, pa_tol_deg=12.0):
    """Physically correct association: candidate lies along the truth line."""
    tang = np.arctan2(t["y2"] - t["y1"], t["x2"] - t["x1"]) % np.pi
    dpa = abs(np.degrees(tang - c["pa_rad"])) % 180.0
    dpa = min(dpa, 180.0 - dpa)
    if dpa > pa_tol_deg:
        return False
    dx, dy = np.cos(tang), np.sin(tang)
    px, py = c["x"] - t["x1"], c["y"] - t["y1"]
    perp = abs(-dy * px + dx * py)
    along = dx * px + dy * py
    return perp <= perp_tol and -30.0 <= along <= t["L"] + 30.0


def report(name, truth, cands, keyx="x", keyy="y"):
    cen = sum(1 for t in truth
              if any(np.hypot(c[keyx] - t["cx"], c[keyy] - t["cy"]) < 40
                     for c in cands))
    lin = sum(1 for t in truth if any(line_assoc(t, c) for c in cands))
    print(f"  {name:<34} {len(cands):5d} cands | centre-match {cen}/{len(truth)}"
          f" | LINE-match {lin}/{len(truth)}")
    return lin


def main():
    img, truth = make_tile()
    print("injected streaks (length px, angle deg):")
    for i, t in enumerate(truth):
        print(f"   {i}: L={t['L']:6.1f}  pa={np.degrees(t['ang']):6.1f}  "
              f"centre=({t['cx']:6.1f},{t['cy']:6.1f})")
    print()

    # --- stage (a): raw peaks from all four families, no cap, no dedup ---
    H, W = img.shape
    frames = {'pos': img, 'neg': img[::-1, :], 'posT': img.T, 'negT': img.T[::-1, :]}
    for cap in (200, 100000):
        raw = []
        for fam, work in frames.items():
            for (L, c0, T, y0, snr) in F._fold_detect(work, 8, 5.0, 7, cap):
                x1, y1 = F._work_to_image(fam, c0, y0, H, W)
                x2, y2 = F._work_to_image(fam, c0 + L - 1, y0 + T, H, W)
                raw.append(dict(x=0.5 * (x1 + x2), y=0.5 * (y1 + y2),
                                pa_rad=float(np.arctan2(y2 - y1, x2 - x1) % np.pi),
                                L_frt=float(np.hypot(x2 - x1, y2 - y1)),
                                snr_frt=float(snr)))
        report(f"raw peaks (top_per_level={cap})", truth, raw)

    # --- stage (c): after dedup ---
    raw_big = []
    for fam, work in frames.items():
        for (L, c0, T, y0, snr) in F._fold_detect(work, 8, 5.0, 7, 100000):
            x1, y1 = F._work_to_image(fam, c0, y0, H, W)
            x2, y2 = F._work_to_image(fam, c0 + L - 1, y0 + T, H, W)
            raw_big.append(dict(x=0.5 * (x1 + x2), y=0.5 * (y1 + y2),
                                pa_rad=float(np.arctan2(y2 - y1, x2 - x1) % np.pi),
                                L_frt=float(np.hypot(x2 - x1, y2 - y1)),
                                snr_frt=float(snr)))
    report("after dedup_candidates", truth, dedup_candidates(raw_big))

    # --- what the pipeline actually returns ---
    pipe = detect_streaks(img, 1.05, tile=1024, overlap=128, min_length=8,
                          threshold=5.0)
    lin = report("detect_streaks (pipeline)", truth, pipe)

    # --- which truth streaks are missed, and by how much ---
    print("\n  per-streak status (pipeline output):")
    for i, t in enumerate(truth):
        hits = [c for c in pipe if line_assoc(t, c)]
        best_cen = min((np.hypot(c["x"] - t["cx"], c["y"] - t["cy"]) for c in pipe),
                       default=np.inf)
        print(f"   {i}: L={t['L']:6.1f} pa={np.degrees(t['ang']):6.1f} -> "
              f"{'FOUND' if hits else 'MISS '} ({len(hits)} line-matched); "
              f"nearest centre {best_cen:6.1f} px")


if __name__ == "__main__":
    main()
