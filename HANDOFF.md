# streak_radon — developer handoff (Klone or any fresh machine)

**Audience:** a fresh Claude Code agent (or human) picking this up on Klone.
**Status date:** 2026-07-25.

This supersedes the older `KLONE_HANDOFF.md` written for the pyradon era, which
is now wrong about the detection engine.

---

## What this tool is

A trailed-source (streak) detector for fast solar-system movers. It turns survey
difference images into `hldet_colformat01` CSVs for heliolinx
`make_trailed_tracklets`. Validated on ZTF (the a000001 NEO), ATLAS (2024 KV),
and CSS/G96.

**Detection engine (as of 2026-07): our own clean-room Fast Radon Transform,
`streakradon/frt.py`.** The previously vendored GPLv3 `pyradon` has been removed;
see `LICENSING.md` for the clean-room provenance and the relicensing options that
removal unlocks.

## Quick start

```bash
git clone git@github.com:rhstrauss/radon-streak.git streak_radon
cd streak_radon
git checkout perf/phase1-fast-detect     # where current work lives
pip install -e .
streak-radon selftest                    # unit + integration (exercises the FRT)
streak-radon demo --out-dir demo_out     # synthetic end-to-end
```

`selftest` must end `SELFTEST PASSED (units + integration/FRT)`. The integration
half is the part that catches a broken install — unit tests alone pass even when
the detector is unusable (that was a real bug, see git history).

## Ground-truth gate (run this before trusting any change)

```bash
python bin/regress_a000001.py
```

Blindly rediscovers a real, faint ZTF NEO in two exposures. Current status on the
native backend: **PASS** — centre offset 0.00″, ΔPA 0.00°, length within 0.1″ of
truth, χ²ᵣ ≈ 1.0, both exposures. Needs the ZTF data at
`/astro/store/shire/rstrau/ari_a000001_check/reverify_2026-06/ztf_search`
(gondor path — will need staging on Klone).

---

## State of play: what is done, what is open

### Done and verified

| item | evidence |
|---|---|
| Clean-room FRT replaces pyradon | a000001 gate PASSes with pyradon never imported |
| Phase 1 perf: `mf_snr` 4.8× faster | 153.7 → 31.8 ms/candidate, AND length error 11.7% → ~2% |
| Phase 1 perf: optional no-Veres fast path | `roughfit.py`; PA 0.5°, length ~2%, both well inside heliolinc gates |
| `frt_gpu.py` numerics | **exact** match to the numpy reference (4029/4029 default cap; 14812/14812 cap lifted) |

### Open / needs work

1. **`top_per_level=200` is discarding ~73% of peaks.** Measured: 4029 kept of
   14812 above-threshold peaks per tile at default settings. This is an
   undocumented completeness limit sitting upstream of the frozen
   `mf_snr_min=11` FP budget. It is the leading explanation for a synthetic
   recovery gap (6/8 injected streaks found in one test). **Calibrate this
   against the FP budget before any survey-scale run.**
2. **GPU path is drafted but never executed on a GPU.** `frt_gpu.py` is validated
   on the torch CPU backend only. It needs a CUDA torch and a real device.
3. **Native FRT performance**: ~9.6 s CPU per 1024² tile (4 octants) → ~347 s CPU
   per 5280² exposure. Profiling pointed at `maximum_filter` (~74% of
   `_fold_detect`) and an O(n²) `dedup_candidates`. Both unoptimized.

### A negative result worth not repeating

`bin/opt_frt_prototype.py` vectorizes the FRT fold loop with `take_along_axis`.
It is **3.7–9× SLOWER** than the plain Python `for T in range(bw)` loop. The fold
is memory-bandwidth-bound, not interpreter-bound: materializing the gathered
tensor + index array + mask costs 3–4× the memory traffic of in-place slicing.
Kept in-tree as a documented dead end. (This is precisely why the GPU port is
expected to win — an L40 has ~10× the bandwidth of a CPU socket.)

---

## MEASUREMENT DISCIPLINE — read this before quoting any number

Timings taken on a shared node in this project have been badly wrong. On gondor,
measured 2026-07-24:

* load average **75** on 128 cores while the CPU was **76% idle** — the load is
  processes blocked in D-state on I/O, not CPU contention;
* `/astro/store/shire` (NFS over RDMA) write+fsync latency: **p50 673 ms, p99
  2.9 s, max 3.6 s** — versus 1.0 ms on the TCP-mounted NFS home and 0.04 ms on
  local disk;
* a full login shell / `conda shell.bash hook` took **>180 s**, while
  `bash --norc` and bare `fork+exec` were instant.

Two published-then-retracted claims came out of ignoring this: a "43× native
slowdown" and a "125× optimization speedup", both pure wall-clock artifacts.

**Therefore:** use `time.process_time()` (CPU) or `perf_counter` with repeats and
report the **minimum**; log `os.getloadavg()` next to every number; for GPU work
call `torch.cuda.synchronize()` before stopping the clock. Put scratch on local
disk (`/local/tmp`), never on the shared store.

---

## Layout

```
streakradon/
  frt.py          clean-room Fast Radon Transform (the detector)
  frt_gpu.py      batched torch FRT — GPU port, CPU-validated, GPU-UNTESTED
  frt_driver.py   tiling, backend dispatch, union-find dedup
  mf_snr.py       integrated matched-filter SNR + candidate refinement
  trail_fit.py    Veres (2012) line⊗Gaussian fit  (precise, slow)
  roughfit.py     approximate trail + errors      (fast path, no Veres)
  pipeline.py     per-exposure chain; process_exposure(fast=) selects the path
  rb.py, varmap.py, register.py, template.py, psf.py, inject.py, hldet_io.py
  adapters/       g96, ztf, atlas, generic → DiffExposure
bin/
  regress_a000001.py   THE ground-truth gate
  ab_phase1.py         legacy-vs-fast A/B on a real sequence
  ab_diagnose.py       stage timing + detection-list comparison
  ab_backends.py       backend A/B
  bench_frt_clean.py   CPU-time benchmark (contention-immune)
  diag_recovery.py     which stage loses an injected streak
  opt_frt_prototype.py documented negative result (see above)
```

## Architecture

**FRT finds → MF re-scores → (optionally) Veres measures.**

For survey scale, set `detect.fast: true`: skip the per-candidate Veres LSQ
(~917 ms each, ~35% of runtime) and emit the FRT+MF trail with approximate
cross/along-track errors. heliolinc's tracklet gates are far looser than Veres
needs (`siglenscale` 0.5 = 50% on length), so rough trails link fine. Run the
Veres fit as a **post-linking refinement** on linked detections only, where its
covariance actually matters (ADES submission, orbit fitting).
