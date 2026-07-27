# streak_radon — handoff

**Status date:** 2026-07-26. Supersedes the earlier `KLONE_HANDOFF.md`.

---

## READ THIS FIRST: blind detection completeness is ~50%, and always was

The headline result of the 2026-07 investigation is not a performance number. It
is that **no test in this project ever exercised the actual survey path**, and
when one was finally written the detector turned out to be losing about half of
even bright, easily-detectable trails.

Measured with `bin/gate_recall.py` — 6 trails injected at mag 17.0–17.4 (well
above the frozen `mf_snr_min=11`) into **real** G96 frames, run through the
**complete blind full-frame chain**, scored on what survives to final hldet
output:

| detector | recall | final detections |
|---|---|---|
| **pyradon (this branch, the baseline)** | **11/23 = 48%** | 31 |
| clean-room FRT, exhaustive + union-find dedup | **0/23 = 0%** | 340 |
| …+ line-space dedup | 14/23 = 61% | 891 |
| …+ multi-representative collapse | 23/23 = 100% | 17,267 |
| CLEAN iteration (9 param combos) | 2/6 = 33%, flat | 62–84 |

**Nothing here is calibrated for completeness.** Treat every existing catalogue
accordingly, including the 2-night batch (276 fields, 17,321 detections) — it was
produced at roughly this recall, so its "empty" fields are not evidence of empty
sky.

### Why it went unnoticed for so long

* `bin/e2e_g96.py` — the acceptance test that reports "3/3 movers recovered" —
  uses **windowed** detection around known positions. The dominant failure mode
  (candidate flood + over-aggressive dedup) cannot occur in a small window.
* `bin/regress_a000001.py` scores **one bright object in a 512² cutout**. It
  passes on detectors that recover nothing at all in blind full-frame mode.

Both still pass on a pipeline with 0% blind recall. **`bin/gate_recall.py` is the
only test that measures what the survey actually does. Run it before trusting any
detector change.**

---

## What this branch is

`revert/pyradon-working`, based on `6a7100c`. The operating baseline: the
vendored pyradon `Finder` (iterative find-and-subtract) plus the Phase 1
performance work.

Chosen because it is the fastest, most stable, best-characterised configuration —
**not** because it is correct. 48% recall is not acceptable for survey work.

Measured on a real G96 4-visit sequence:

| | |
|---|---|
| speed | 356 → **113 s/exposure** (3.2× vs stock) |
| candidates | 11–12 per 1024² tile |
| detections vs stock pyradon | 11/11 agreement, zero lost |
| a000001 ground-truth gate | PASS (0.00″ centre, 0.00° PA) |
| **blind injection recall** | **48%** ← the open problem |

Phase 1 performance work (all backend-independent, all retained):

* `fastmask.py` — geometric capsule replaces pyradon's 4×-oversampled full-tile
  model + `convolve2d`, used only as a boolean footprint. **550×** cheaper,
  validated to cover pyradon's footprint with **zero** missed pixels over 40
  random geometries. ~201 s → ~0.4 s per exposure.
* `mf_snr.refine_candidate` — kernel hoisted out of the slide loop, plus length
  and along-track polish: **153.7 → 31.8 ms/candidate** *and* length error
  11.7% → ~2%.
* `roughfit.py` + `process_exposure(fast=True)` — skip the per-candidate Veres
  LSQ (~917 ms each, ~35% of runtime); emit the FRT+MF trail with approximate
  errors, which is well inside heliolinc's gates (PA 0.5°, length ~2% vs
  `siglenscale` 0.5). Veres becomes a **post-linking** refinement.

## Other branches

* `perf/phase1-fast-detect` (`31307f9`, pushed) — clean-room FRT (`frt.py`),
  pyradon removed, GPU port (`frt_gpu.py`), harnesses. **Do not run surveys from
  it**: 0% blind recall. Worth resuming *only* with `gate_recall.py` as its
  acceptance test. `LICENSING.md` explains what removing pyradon would unlock.
* `frt_gpu.py` — batched torch FRT, validated **exact** against the numpy
  reference (4029/4029 default cap, 14812/14812 cap lifted) and **~55×** faster
  on an L40 (1.86 → 0.034 s/tile). Correct and fast; blocked only by the
  detector-completeness problem above. CUDA torch 2.13.0+cu130 is installed in
  the `mpchecker` env; both L40s are visible.

---

## The open problem, and what is already ruled out

Blind full-frame recall must reach ≥80% before any survey run. Ruled out by
measurement (do not re-litigate):

1. **`top_per_level`** — the FRT finds 8/8 injected streaks at both the default
   cap and with it lifted. Not the cause.
2. **Dedup tolerances** — 27 combinations of
   `(perp_tol, len_ratio_min, along_pad)` moved survivors only between 11,882 and
   25,785, all at full recall. The wrong knob.
3. **`max_iter` / CLEAN threshold** — 9 combinations gave a completely flat 2/6.
   Not a tuning problem.

What *is* established: the clean-room FRT emits **98,289 raw candidates per
exposure** (≈2,700/tile) because one streak recurs at many folding levels,
angles and tiles, whereas pyradon's iterative subtraction yields 11–12/tile.
Dedup was being asked for a ~5000:1 reduction, which no safe merge rule achieves
— the only rule that did (transitive union-find on centre proximity) chained
unrelated streaks through long "bridge" candidates and destroyed everything.

The likely direction is a correct iterative detector on the native FRT — CLEAN
is the right family (published method; only pyradon's *source* is copyrighted) —
but my implementation loses trails for a reason not yet diagnosed, and its
flat parameter response says the fault is structural, not numerical.

---

## Measurement discipline (this bit is not optional)

Timings on these shared nodes have been badly misleading. Measured on gondor,
2026-07-24:

* load average **75** on 128 cores while the CPU was **76% idle** — the load is
  processes blocked in D-state on I/O, not CPU contention;
* `/astro/store/shire` (NFS over RDMA) write+fsync: **p50 673 ms, p99 2.9 s,
  max 3.6 s**, versus 1.0 ms on the TCP-mounted home and 0.04 ms on local disk;
* a full login shell / `conda shell.bash hook` took **>180 s** while
  `bash --norc` and bare `fork+exec` were instant.

Two claims were published from this session and then retracted — a "43× native
slowdown" and a "125× optimization speedup" — both pure wall-clock artifacts.

**So:** use `time.process_time()` or repeats-and-take-minimum; log
`os.getloadavg()` beside every number; call `torch.cuda.synchronize()` before
stopping a GPU clock; keep scratch on `/local/tmp`, never the shared store.

---

## Quick start

```bash
git clone git@github.com:rhstrauss/radon-streak.git streak_radon
cd streak_radon && git checkout revert/pyradon-working
pip install -e .
streak-radon selftest                     # units + integration/FRT
python bin/regress_a000001.py             # ground truth (needs the ZTF data staged)
python bin/gate_recall.py --frames '<a G96 4-visit sequence>'   # THE survey-path test
```

`bin/regress_a000001.py` needs
`/astro/store/shire/rstrau/ari_a000001_check/reverify_2026-06/ztf_search`
(a gondor path — stage it before using this on another machine).

## Layout

```
streakradon/
  frt_driver.py   tiling, pyradon Finder drive, union-find dedup
  fastmask.py     geometric capsule suppression (550x cheaper than the model path)
  mf_snr.py       integrated matched-filter SNR + candidate refinement
  trail_fit.py    Veres (2012) line⊗Gaussian fit   (precise, ~917 ms)
  roughfit.py     approximate trail + errors        (fast path, no Veres)
  pipeline.py     per-exposure chain; process_exposure(fast=) picks the path
  rb.py, varmap.py, register.py, template.py, psf.py, inject.py, hldet_io.py
  adapters/       g96, ztf, atlas, generic -> DiffExposure
bin/
  gate_recall.py       blind-path completeness  <-- run this
  regress_a000001.py   ground-truth recovery
  ab_phase1.py         legacy-vs-fast A/B on real data
  ab_diagnose.py       stage timing + detection-list comparison
```

## Survey driver (built, deliberately NOT run)

`css_pds_pilot/staged_survey.py` runs 2 nights → week → month against the CSS
PDS archive, resumable, with `gate_recall.py` as a hard gate between stages
(threshold 80%). It has never been launched: the baseline is at 48%, and a
month-scale batch at that completeness would bake the problem into ~21,000
exposures of catalogue. The gate is doing its job by refusing.
