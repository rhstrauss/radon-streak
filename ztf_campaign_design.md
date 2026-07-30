# ZTF trail extraction with the GPU detector — design and cost

Re-run of the ZTF trail extraction using the current detector (GPU iterative-CLEAN
FRT + fast mode) instead of the CPU path that produced the earlier catalog, plus
the ZTF-specific corrections found while wiring it up.

Campaign order, as requested: **Sep 1 – Dec 31 2024 first, then back through
Jul and Jun 2024.** Unlike CSS — whose telescopes are dark for the Arizona
monsoon, so Jul/Aug 2024 does not exist at all — ZTF observed continuously
through that period, so the Jun/Jul extension is real data rather than a gap.

## 1. Two defects in the previous ZTF path

Both were silent: no error, no crash, only missing or corrupted science.

### 1.1 The mask was deleting the signal

ZTF's `mskimg` is **not** a bad-pixel mask. Its own header documents the bits,
and two of them mark *real astrophysical sources*:

```
bit  0  AIRCRAFT/SATELLITE TRACK
bit  1  CONTAINS SEXTRACTOR DETECTION           <-- a SOURCE, not a defect
bit  2  LOW RESPONSIVITY          bit  3  HIGH RESPONSIVITY
bit  4  NOISY                     bit  5  GHOST FROM BRIGHT SOURCE
bit  6  POSSIBLE GHOST FROM CHARGE SPILLAGE
bit  7  PIXEL SPIKE (POSSIBLE RAD HIT)
bit  8  SATURATED                 bit  9  DEAD (UNRESPONSIVE)
bit 10  NAN
bit 11  CONTAINS PSF-EXTRACTED SOURCE POSITION  <-- also a SOURCE
bit 12  HALO FROM BRIGHT SOURCE
```

The old adapter did `m = (mask != 0)` and then `diff = where(m == 0, diff, nan)`,
so **every pixel where SExtractor found something was blanked before detection**.
For a trailed NEO bright enough to be catalogued in the science frame, that
blanks the trail itself — the exact detections this pipeline exists to find.

Measured on a real 2024-09-01 quadrant: bit 1 = 0.336%, bit 11 = 0.015%,
bit 12 = 1.651% (the only large genuine defect class). Bit-selective masking
takes the bad fraction from 2.010% to 1.680% and **rescues 32,278 source pixels
per quadrant**.

Bit 0 *is* masked, deliberately: IPAC flags only long quadrant-crossing tracks,
so masking it removes the dominant streak contaminant without touching the
~5–200″ trails we want.

### 1.2 Epochs were ~1.4 s early, and sciimg was being downloaded for nothing

`SHUTOPEN`/`SHUTCLSD` are present in the **diff** header, so the validated
shutter-midpoint epoch `(SHUTOPEN+SHUTCLSD)/2` costs no extra bytes. Measured
against the old `OBSMJD + EXPTIME/2` fallback on a real header: **+1.43 s**,
which is ≈3.2″ along-track for a fast NEO at 134″/min.

Consequence for the campaign: `sciimg` (37.9 MB, the largest product) is never
needed. Per quadrant-exposure the download is **26.3 MB** (measured mean: diff
~7.4 MB + mask 18.9 MB) instead of ~64 MB — for Sep–Dec that is **84.5 TB instead
of ~206 TB** (§3).

IBE serves no gzip Content-Encoding (verified), so 26.3 MB is the floor.

### 1.3 Also found: heliolinx silently truncates idstrings

`hldet.idstring` is `char[20]`, and the reader is
`stringncopy01(idstring, s, SHORTSTRINGLEN)` — it **truncates silently** rather
than erroring. A natural ZTF id
(`ztf_20240901136065_000577_zr_c01_o_q1_scimrefdiffimg_c000`, 56 chars) would
collapse to `ztf_20240901136065` for every detection in the exposure.

ZTF ids are therefore built to 18 chars, unique and reversible:

```
z 240901136065 1f 07
| |            |  +-- candidate index within the exposure, base36 (3)
| |            +----- readout channel (ccdid-1)*4+(qid-1) = 0..63, hex (2)
| +------------------ filefracday minus the century: YYMMDDffffff (12)
+-------------------- survey tag
```

> **This affects the CSS catalogs too.** G96 ids like
> `G96_20240404_2B_N02001_01_0001_c000` (35 chars) truncate to
> `G96_20240404_2B_N02`, which is identical across fields — so idstring-based
> dedup (`dedup_across_windows`) and any join back to the catalog silently
> collapse. That is the same root cause as the known `mpcat_check` join failure.
> Worth fixing in `klone/merge_catalog.py` before the G96 week catalog is linked.

## 2. Best practices carried over from the validated G96 path

`config/ztf.yaml` previously had only `detect` and `vet` blocks. Added:

- **`detect.fast: true`** — this is what took G96 blind recall from 48% to 96%.
  It is not merely a speed knob: it also bypasses a chi2 cut that was killing
  *bright* trails.
- **`detect.fast_suppress: true`** — geometric capsule blanking, same footprint,
  ~5.5× on the detect stage.
- **GPU iterative-CLEAN** via `STREAKRADON_GPU_CLEAN=1` / `CLEAN_ITER=10`, set by
  the submitter. Survey-agnostic, so it applies to ZTF unchanged.
- **Cross-exposure repetition filter** — the main purity cut on ZTF diffs. This is
  why the work unit is a *quadrant-visit sequence* and not a single exposure
  (§3). Tolerances are G96's scaled to the same pixel tolerance: 3.33″, 10°.
  Caveat: **47% of ZTF units on 2024-09-01 have only one exposure**, so the cut
  is unavailable for about half the data.
- **Per-obscode trail_len floor/ceiling policy** — `I41: 1.012"/px` added to
  `merge_catalog.py`, plus a *measured* floor override (see §2.1). The 192 px
  ceiling still applies (194.3″ on ZTF).

### 2.1 The floor artifact is relocated on ZTF, not absent

My first reading of this was wrong and the pilot corrected it. Because ZTF's floor
computes to 4.8 px = 4.86″, *below* the `len_min_arcsec: 8.0` vet gate, it looked
as though ZTF would escape the `trail_len` floor artifact that affects 51.7% of
G96 detections.

It does not. The gate **relocates** the pile-up rather than removing it: with
`min_length: 8` px and the gate at 8.0″, the lowest length that can be both
searched and survive vetting is the ladder's 8-px rung = **8.10″** — and the pilot
unit put **7 of 13 detections (54%) at exactly 8.10″**, the same order as G96's
51.7% at 7.3″.

So `trail_len = 8.10″` on I41 is a **bound, not a measurement**, and computing the
floor as 4.86″ would mean the floor policy **never fires** — handing 8.10″ to
`make_trailed_tracklets` as though it were measured, i.e. asserting ~6.5 deg/day
for every marginal or unresolved ZTF source. That is exactly the failure the CSS
work warned about. `merge_catalog.py` therefore carries an explicit measured
override, `FLOOR_ARCSEC["I41"] = 8.2`, rather than deriving it.

G96 keeps the derived value: its floor was measured empirically at 7.3″, so
`polish_length` does push below that survey's `min_length`.

### Deliberately NOT changed without measurement

- **`mf_snr_min: 6.0`** is left as-is but flagged PROVISIONAL. It came from the P6
  head-to-head against the ztf_streak matched filter, which was a *recall* test
  on two exposures — it never measured the false-positive rate. G96's 11.0 has a
  negated-diff FP calibration behind it; ZTF has nothing equivalent.
  `klone/calibrate_ztf_fp.sh` + `klone/fp_report.py` run that measurement
  (negate the diff → real sources go negative → every surviving detection is an
  FP by construction). **Run this before committing to the full footprint**: an
  uncalibrated threshold at 3.2M quadrants risks a multi-million-row FP flood
  that would only be discovered after 84 TB of downloads.
- **`vet.len_min_arcsec: 8.0`** = 8.0″/30 s = 960″/hr = 6.4 deg/day, the
  trailed/point-source boundary. Kept because ZTF's own alert stream already
  supplies point sources in bulk (the ml-clean I41 catalog is 2.06M rows for one
  month), so this detector's value is the complementary trailed regime. Lowering
  it to ~5.0 would admit the 3.9–6.4 deg/day band at the cost of floor
  contamination and a much larger FP volume — a measurement, not a guess.
- Note the whitened noise measures **σ ≈ 1.17**, not 1.0, on real ZTF diffs
  (correlated noise from the reference subtraction). Thresholds are therefore
  ~17% more permissive than nominal, which is a second reason the FP calibration
  matters.

## 3. Work unit and campaign architecture

**Unit = one (night, field, ccdid, qid) quadrant, with all of that night's
exposures of it** — typically 2 same-band revisits, up to 4 across g+r. Grouping
this way is what makes the repetition filter possible.

Measured on 2024-09-01: **299 units per readout channel per night**, so
**~19,100 units/night** across all 64 channels.

| window | quadrant-exposures | units | download | download-bound wall clock @ CONC=64 |
|---|---|---|---|---|
| Sep 1 – Dec 31 2024 | 3.21 M | ~2.33 M | **84.5 TB** | **~5.1 days** |
| Jun + Jul 2024 | 1.41 M | ~1.03 M | 37.2 TB | ~2.2 days |

Measured from a compute node (job 37901242): **3.00 MB/s per stream**, 26.3 MB per
quadrant-exposure on average (diff ~7.4 MB, mask 18.9 MB) = 8.8 s of stream time
each. At `CONC=64` that is 192 MB/s aggregate, hence the wall clock above. **This
is a floor independent of GPU throughput** — if the detector is faster than 8.8 s
per exposure, the campaign is download-bound, not compute-bound.

### The mask is 73% of the bytes

The uncompressed int16 `mskimg` is 18.9 MB against the diff's ~7.4 MB, and IBE
offers no gzip. There is a tempting 3.4× lever here: most defect bits
(low/high responsivity, dead, noisy, spikes) are *static per readout channel*, and
because ZTF fields are fixed, even the dominant halo bit (12) is nearly static per
`(field, ccdid, qid)`. Caching ~115k masks (~2.2 TB) and reusing them would cut
Sep–Dec to ~25 TB (mostly diffs).

**Not implemented, deliberately.** Bit 0 (`AIRCRAFT/SATELLITE TRACK`) is genuinely
per-frame, and satellites are the dominant streak contaminant for a streak
detector — the repetition filter cannot catch them either, since a satellite
crosses once. Trading per-frame satellite masking for download volume would
undercut the purity of exactly the detections this campaign produces. Flagged as
the lever to pull only if the 5-day download floor proves unacceptable.

Three hard constraints, all learned on the CSS campaigns:

1. **No pre-stage.** 84.5 TB against ~2.5 TB free on `/gscratch/astro` means
   download is per-unit and streaming: fetch → detect → delete, on every exit
   path. That also makes the array itself the download parallelism (N concurrent
   tasks = N IRSA streams), rather than serialising behind one prestage job.
   `CONC` defaults to 64 to stay polite to IRSA.
2. **QOS submit cap.** `MaxSubmitJobsPerUser = 2000` counts every pending array
   task across all jobs, so the budget is 2000 minus what is already queued.
   `submit_ztf.sh` reads live headroom and **checks `sbatch`'s exit status** — a
   rejected submit is otherwise indistinguishable from a successful one, which is
   how the CSS month run silently processed nothing for ~12 h.
3. **Walltime.** ckpt-g2 clamps to ~5:05 regardless of the request, so
   `PACK` is additionally capped at `TIME / SEC_PER_UNIT`. `process_ztf_unit.sh`
   is idempotent, so a preempted or timed-out task resumes at the first
   unfinished unit.

Together these mean the campaign is **a chain of few-night batches**, not one
array. `launch_ztf.sh <tag> <start> <end>` is one link; run it repeatedly as
headroom frees.

## 4. Files

```
streakradon/adapters/ztf.py     bit-selective mask, diff-header timing,
                                prepare_sequence, negate path for FP calibration
config/ztf.yaml                 fast mode, fast_suppress, mask_bits, preprocess,
                                repetition-filter tolerances, edge_px
bin/run_ztf_sequence.py         sequence runner + 18-char idstrings
klone/scrape_ztf.py             IBE metadata -> unit list (no pixels)
klone/process_ztf_unit.sh       download -> detect -> stream-delete (idempotent)
klone/submit_ztf.sh             packed, headroom-aware, rc-checked array submit
klone/launch_ztf.sh             scrape + submit one night batch
klone/calibrate_ztf_fp.sh       negated-diff FP calibration
klone/fp_report.py              FP rate + mf_snr distribution -> mf_snr_min
tests/test_ztf_adapter.py       8 tests, regression-protecting both defects
```

Runtime tree on klone is `/gscratch/astro/rstrau/streak_radon_ztf` (a sibling
copy, following the existing `streak_radon_gpu` / `streak_radon_perf` pattern) so
that rebuilding or editing does not perturb the CSS jobs running live out of
`streak_radon_git`. It depends on the GPU integration
(`frt_clean_gpu.py`, `frt_gpu.py`, `stamps.py`, the `pipeline.py`/`frt_driver.py`
edits), which is still uncommitted in that tree and therefore not part of this
branch.

## 5. Measured throughput — the campaign is CPU-bound inside a GPU allocation

One 4-exposure unit end-to-end on an **A40** (job 37901265): **3m58s**, rc=0,
**13 detections**, repetition filter removed 2. Stage breakdown from the log:

| stage | cost | share of steady state |
|---|---|---|
| CUDA/cupy init | ~72 s **once per task** | amortised away by `PACK` |
| download | 8.8 s/exposure | ~20% |
| **GPU CLEAN detect** | **5 s/exposure** | **~11%** |
| preprocess (whiten) + MF fit + RB vet + I/O | ~28 s/exposure | ~64% |

(The log shows `exp0: 97 FRT cands (77s)` then `5s` for exp1–3 — that 72 s delta is
the one-time CUDA init, not per-exposure detect cost.)

Steady state is therefore **~72 s per unit** (1.72 exposures average), of which the
GPU does ~9 s of work. **The GPU is idle ~89% of the time**, so one worker per GPU
wastes the allocation. Two consequences:

1. `submit_ztf.sh` now runs **`NWORK=4` unit workers per task** sharing the one
   GPU (`xargs -P`), overlapping one worker's CPU/network with another's GPU work.
   `NWORK=1` restores serial behaviour.
2. **The highest-leverage optimisation for this campaign is the existing
   `perf/preprocess-vet` branch**, not anything GPU-side — 64% of the cost is in
   exactly the stages it targets.

Cost for Sep–Dec. `NWORK=4` was then **measured** on pilot `ztf_pilot_a`
(job 37902404): 41 units / 77 quadrant-exposures in 17m13s = **13.4 s per exposure**
against ~41.9 s serial, i.e. a **3.1× speedup** (not the naive 4×, as expected once
four workers contend for one GPU and the same memory bandwidth).

| | CONC=16 | CONC=64 |
|---|---|---|
| NWORK=1 (~41.9 s/exposure) | ~97 days | ~24 days |
| **NWORK=4 (measured 13.4 s/exposure)** | ~31 days | **~7.8 days** |

Jun+Jul adds ~3.4 days on the same settings. Against the ~5.1-day download floor,
`CONC=64 NWORK=4` is a reasonably balanced pipeline: ~256 workers, but each spends
only ~21% of its time downloading, so **average active IRSA streams ≈ 54** — polite —
while delivering ~4.8 exposures/s.

Detection density: 3.25 per quadrant-exposure → **~10.4 M detections for Sep–Dec**,
comparable per-mosaic to G96's 104/image. 23% of pilot rows carry `mag = nan`
(photometry failed); `merge_catalog.py --drop-nan-mag` handles them, and the
default is to keep them.

## 6. Open items

- [x] GPU throughput per unit — measured, §5. `SEC_PER_UNIT = 75`.
- [x] `NWORK=4` scaling factor — measured 3.1x (pilot `ztf_pilot_a`, job 37902404).
- [ ] Negated-diff FP calibration → freeze `mf_snr_min`.
- [ ] Recall gate: inject trails into ZTF diffs and confirm the mask fix actually
      moves recall, so the improvement is quantified rather than argued.
- [ ] Decide priority against the in-flight CSS Sep–Dec campaign: both want the
      same GPUs, the same 2000-task budget, and the same calendar window. Note the
      CSS `sr_month_gpu` array held 76 GPUs during this work and starved a
      single-task ZTF smoke job to a priority of 2e-5; it only ran once submitted
      to `ckpt-all`, which had idle L40S/H200 nodes that `ckpt-g2` did not.
- [ ] Consider landing `perf/preprocess-vet` FIRST — it targets the 64% of
      per-unit cost that dominates this campaign (§5).
- [ ] Fix the CSS idstring truncation in `klone/merge_catalog.py` (§1.3) before
      linking the G96 week catalog.
