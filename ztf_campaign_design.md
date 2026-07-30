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
needed. Per quadrant-exposure the download is **28.3 MB** (diff 9.4 + mask 18.9)
instead of 66.2 MB — for Sep–Dec that is 91 TB instead of 212 TB.

IBE serves no gzip Content-Encoding (verified), so 28.3 MB is the floor.

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
  `merge_catalog.py`. ZTF's floor is 4.8 px = 4.86″, which is *below* the vet
  gate `len_min_arcsec: 8.0`, so **ZTF is largely immune to the floor artifact
  that affects 51.7% of G96 detections**. The 192 px ceiling still applies
  (194.3″ on ZTF).

### Deliberately NOT changed without measurement

- **`mf_snr_min: 6.0`** is left as-is but flagged PROVISIONAL. It came from the P6
  head-to-head against the ztf_streak matched filter, which was a *recall* test
  on two exposures — it never measured the false-positive rate. G96's 11.0 has a
  negated-diff FP calibration behind it; ZTF has nothing equivalent.
  `klone/calibrate_ztf_fp.sh` + `klone/fp_report.py` run that measurement
  (negate the diff → real sources go negative → every surviving detection is an
  FP by construction). **Run this before committing to the full footprint**: an
  uncalibrated threshold at 3.2M quadrants risks a multi-million-row FP flood
  that would only be discovered after 91 TB of downloads.
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

| window | quadrant-exposures | units | download |
|---|---|---|---|
| Sep 1 – Dec 31 2024 | 3.21 M | ~2.33 M | **90.9 TB** |
| Jun + Jul 2024 | 1.41 M | ~1.03 M | 40.0 TB |

Three hard constraints, all learned on the CSS campaigns:

1. **No pre-stage.** 91 TB against ~2.5 TB free on `/gscratch/astro` means
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

## 5. Open items

- [ ] GPU throughput per unit — needed to size `SEC_PER_UNIT` and the real
      wall-clock. Blocked on a free GPU; the CSS month/H2/703 arrays currently
      hold the fairshare.
- [ ] Negated-diff FP calibration → freeze `mf_snr_min`.
- [ ] Recall gate: inject trails into ZTF diffs and confirm the mask fix actually
      moves recall, so the improvement is quantified rather than argued.
- [ ] Decide priority against the in-flight CSS Sep–Dec campaign: both want the
      same GPUs, the same 2000-task budget, and the same calendar window.
- [ ] Fix the CSS idstring truncation in `klone/merge_catalog.py` (§1.3) before
      linking the G96 week catalog.
