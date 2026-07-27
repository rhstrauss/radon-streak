#!/usr/bin/env python
"""Staged, gated CSS survey batch: 2 nights -> rest of week -> rest of month.

WHY GATED. The previous 2-night batch (276 fields, 17,321 detections) ran the
blind full-frame path with union-find dedup, which was later measured to recover
0/23 bright injected trails end-to-end -- it silently destroyed nearly every real
streak while still emitting plausible-looking detections. Nothing in the pipeline
noticed, because the acceptance test (bin/e2e_g96.py) uses WINDOWED detection,
where chain-collapse cannot occur.

So each stage here is preceded by an INJECTION RECALL GATE on real frames from
that stage's own data. If recall drops below the threshold the run stops rather
than spending days producing another catalogue of unknown completeness.

Resumable: fields whose <field>_trails.csv already exists are skipped, so a
killed run continues where it left off.

Usage:
  staged_survey.py --stage 2night          # run one stage
  staged_survey.py --auto                  # 2night -> week -> month, gated
  staged_survey.py --gate-only --stage week
"""
import argparse, csv, glob, json, os, subprocess, sys, time
import numpy as np

PILOT = "/astro/store/shire/rstrau/css_pds_pilot"
SR = "/astro/store/shire/rstrau/streak_radon"
PY = "/astro/store/shire/rstrau/miniforge3/envs/mpchecker/bin/python"
BASE = ("https://sbnarchive.psi.edu/pds4/surveys/gbo.ast.catalina.survey/"
        "data_calibrated/G96/2024")
OUT = f"{PILOT}/survey"
STATE = f"{OUT}/state.json"

# CSS nights, in the order we take them. 24Apr04/05 are the already-run pair.
STAGES = {
    "2night": ["24Apr04", "24Apr05"],
    "week":   ["24Apr07", "24Apr08", "24Apr10", "24Apr11", "24Apr12"],
    "month":  ["24Apr13", "24Apr14", "24Apr15", "24Apr16", "24Apr17",
               "24Apr18", "24Apr19", "24Apr20", "24Apr21", "24Apr22",
               "24Apr23", "24Apr24", "24Apr25", "24Apr26", "24Apr27",
               "24Apr28", "24Apr29", "24Apr30"],
}
RECALL_MIN = 0.80          # gate: injected trails recovered end-to-end
GATE_FIELDS = 1            # fields used for the gate (one 4-visit sequence)


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"stages_done": [], "gates": {}}


def save_state(s):
    os.makedirs(OUT, exist_ok=True)
    json.dump(s, open(STATE, "w"), indent=1)


def night_fields(night):
    """Fields on `night` that have a complete 4-visit sequence."""
    import urllib.request
    url = f"{BASE}/{night}/"
    try:
        html = urllib.request.urlopen(url, timeout=120).read().decode("latin-1")
    except Exception as e:
        print(f"  [{night}] listing failed: {e}", flush=True)
        return []
    import re
    files = set(re.findall(r"G96_(\d{8})_2B_([A-Z0-9]+)_01_000([1-4])\.arch\.fz",
                           html))
    byfield = {}
    for ymd, fld, n in files:
        byfield.setdefault((ymd, fld), set()).add(n)
    return sorted((ymd, f) for (ymd, f), ns in byfield.items() if len(ns) == 4)


def run_gate(night, tag):
    """Injection recall gate on one real 4-visit sequence from `night`."""
    fields = night_fields(night)
    if not fields:
        return None
    ymd, fld = fields[len(fields) // 2]
    d = f"{OUT}/_gate/{tag}"
    os.makedirs(d, exist_ok=True)
    for n in ("0001", "0002", "0003", "0004"):
        fn = f"G96_{ymd}_2B_{fld}_01_{n}.arch.fz"
        dst = f"{d}/{fn}"
        if not (os.path.exists(dst) and os.path.getsize(dst) > 0):
            subprocess.run(["curl", "-sS", "--max-time", "600", "-o", dst,
                            f"{BASE}/{night}/{fn}"], check=False)
    r = subprocess.run([PY, f"{SR}/bin/gate_recall.py", "--frames",
                        f"{d}/G96_{ymd}_2B_{fld}_01_000?.arch.fz"],
                       capture_output=True, text=True)
    recall = None
    for line in (r.stdout or "").splitlines():
        if line.startswith("RECALL="):
            try:
                recall = float(line.split("=", 1)[1])
            except ValueError:
                pass
    print(f"  [{tag}] gate field {fld} ({night}): recall={recall}", flush=True)
    if r.returncode != 0 and recall is None:
        print((r.stderr or "")[-800:], flush=True)
    return recall


def run_stage(stage, parallel=20):
    nights = STAGES[stage]
    man = f"{OUT}/{stage}_manifest.txt"
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for night in nights:
        for ymd, fld in night_fields(night):
            rows.append(f"{night} {ymd} {fld}")
    with open(man, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"  [{stage}] {len(rows)} fields ({len(rows)*4} exposures) -> {man}",
          flush=True)
    t0 = time.time()
    subprocess.run(["bash", f"{PILOT}/run_batch_survey.sh", man, str(parallel),
                    f"{OUT}/{stage}"], check=False)
    dt = time.time() - t0
    done = len(glob.glob(f"{OUT}/{stage}/*/*/*_trails.csv"))
    dets = 0
    for p in glob.glob(f"{OUT}/{stage}/*/*/*_trails.csv"):
        with open(p) as fh:
            dets += max(sum(1 for _ in fh) - 1, 0)
    print(f"  [{stage}] {done}/{len(rows)} fields, {dets} detections, "
          f"{dt/3600:.2f} h", flush=True)
    return dict(fields=len(rows), done=done, detections=dets, hours=dt / 3600)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STAGES))
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--parallel", type=int, default=20)
    a = ap.parse_args()
    st = load_state()
    order = list(STAGES) if a.auto else [a.stage]
    for stage in order:
        if stage in st["stages_done"]:
            print(f"[{stage}] already done, skipping", flush=True)
            continue
        print(f"\n=== GATE before {stage} ===", flush=True)
        rec = run_gate(STAGES[stage][0], stage)
        st["gates"][stage] = rec
        save_state(st)
        if rec is None or rec < RECALL_MIN:
            print(f"*** GATE FAILED for {stage}: recall={rec} < {RECALL_MIN}. "
                  f"STOPPING -- not spending compute on a catalogue of unknown "
                  f"completeness.", flush=True)
            return 1
        if a.gate_only:
            continue
        print(f"=== RUN {stage} ===", flush=True)
        res = run_stage(stage, a.parallel)
        st.setdefault("results", {})[stage] = res
        st["stages_done"].append(stage)
        save_state(st)
    print("\nAll requested stages complete.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
