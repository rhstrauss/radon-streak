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

- **`mf_snr_min: 6.0` is now MEASURED TO BE FAR TOO PERMISSIVE — see §7. It gives
  ~15% purity, and the campaign is HELD until it is re-set.** Original note follows.
- **`mf_snr_min: 6.0`** was left as-is but flagged PROVISIONAL. It came from the P6
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

### 5.1 Pilot catalog — the delivered product, validated

`merge_catalog.py work/ztf_pilot_a` over the 64-unit pilot (`work/cat/ztf_pilot_a_ptsrc.csv`):

```
fields merged   : 64
detections      : 537
MJD range       : 60555.14719 .. 60555.49751
policies        : floor=ptsrc ceil=inflate
  obscode I41    n=537
    floor 8.2"      at floor=282 (52.5%)
    ceil  194"      at ceil =4 (0.7%)
dropped nan mag : 0
unparseable rows: 0
```

Two things to read off this:

- **The relocated floor (§2.1) is confirmed at scale: 52.5% of 537 detections sit at
  the 8.2″ bound**, matching the 54% seen in the single smoke unit and G96's 51.7%.
  Without the `FLOOR_ARCSEC["I41"]` override, the policy would have passed all 282 of
  them to `make_trailed_tracklets` as measured lengths, asserting ~6.5 deg/day motion
  for each. That is the single most consequential fix for downstream linking.
- The 192 px ceiling fires on 0.7%, the same order as G96's 0.1%.

Density is **4.5 detections per quadrant-exposure** (537/120), so Sep–Dec projects to
**~14 M detections**. 23% of rows carry `mag = nan` (photometry failed);
`merge_catalog.py --drop-nan-mag` removes them and the default keeps them.

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

## 7. FP calibration result — the campaign is HELD

The negated-diff calibration ran (job 37904599, 40 multi-exposure units) and the
answer gates the whole campaign.

```
negated-diff FP calibration: 40 unit CSVs, 335 FALSE POSITIVES
units processed: 40  ->  8.38 FP/unit
```

Negation was **verified to have taken effect** — 0 of 12 shared units produced
identical detection positions between the real and negated runs.

The number that matters is the like-for-like comparison on the **same 40 units**:

| run | detections |
|---|---|
| real | 394 |
| negated (source-free by construction) | 335 |

Real detections are (artifacts + true sources); negated are (artifacts alone). So
the **FP fraction is 0.85 — purity is only ~15%** at `mf_snr_min: 6.0`.

Extrapolated, Sep–Dec's ~14 M detections would be ~12 M artifacts. That is the same
artifact-pileup that fuels the grazer "monster core" and the −82% false-link / −70%
RAM difference between the raw and ml-clean ZTF catalogs, so shipping it would poison
the linker this campaign exists to feed. **`ztf_sep_a` was therefore cancelled after
the measurement, not left running.** Nothing is lost: unit lists persist and
`process_ztf_unit.sh` is idempotent.

Note that **sane detection *density* concealed poor *purity***. 8.4 detections/unit
looked unremarkable next to G96, and I wrongly called the calibration non-urgent on
that basis. Density and purity are independent, and only the negated control
separates them.

### Why the first report could not set the threshold

Two of the discriminants available in the delivered catalog turn out to be useless:

- **`mf_snr` is not in the hldet CSV**, so the first report used
  `2.5/ln10/sigmag` as a proxy — and that does not discriminate: pure false
  positives measured p50 = 134, max = 2714.
- **`det_qual` is saturated at 1.00** for false positives as well as real
  detections (p50 = p90 = p99 = 1.00), so it cannot be used as a cut either.

Fixed: `FITJSON=1` (`process_ztf_unit.sh`, plumbed through `submit_ztf.sh`) dumps
every surviving fit including raw `mf_snr`, and `fp_report.py` now reports the FP
`mf_snr` distribution, the threshold required for a target FP/unit rate, and — given
`--real-fits` from the same units — the purity ratio and the real excess per `mf_snr`
decile. Re-running as `ztf_realcal` (37905290) + `ztf_fpcal` (37905291).

The FP `trail_len` distribution from the first run is itself informative: p50 = 8.11″
(the floor) and p99 = 194.32″ (the ceiling), i.e. **false positives pile up at both
ladder boundaries** — consistent with the floor population being artifact-dominated,
and a further reason to treat both bounds as bounds (§2.1).

### Gate before the full campaign

1. Re-set `mf_snr_min` from the new threshold table.
2. **Re-check recall at that threshold** with injected trails — purity is free if you
   throw everything away, so the threshold is only defensible against a recall number.
   This is the one measurement still missing in both directions.
3. Only then launch Sep–Dec.

### 7.1 …and `mf_snr` turns out not to be the lever at all

The re-run with raw `mf_snr` captured (`ztf_fpcal` 37905291 + `ztf_realcal` 37905290,
same 40 units, `FITJSON=1`) gives purity ~12% (real 404, negated 355) and this:

```
FP mf_snr : min 6.87  p50 124.17  p90 666.38  p99 1662.54  max 2427.67

mf_snr_min needed for a target FP rate:
  1.00 FP/unit : >=  627.71      0.20 FP/unit : >= 1391.20
  0.50 FP/unit : >=  894.03      0.10 FP/unit : >= 1638.74

real excess by mf_snr decile (real_n - fp_n, equal units):
    6.32-  24.75: real 37  fp~38  excess  -1
   24.75-  51.08: real 38  fp~38  excess  +0
   51.08-  69.78: real 45  fp~31  excess +14
   69.78-  97.34: real 44  fp~32  excess +12
   97.34- 122.42: real 41  fp~35  excess  +6
  122.42- 156.20: real 42  fp~34  excess  +8
  156.20- 217.83: real 42  fp~34  excess  +8
  217.83- 338.96: real 42  fp~34  excess  +8
  338.96- 582.29: real 41  fp~35  excess  +6
  582.29-3339.40: real 32  fp~44  excess -12
```

**Raising `mf_snr_min` will not fix this.** The false positives are not low-SNR noise
— they are high-SNR structured residuals whose `mf_snr` distribution is
indistinguishable from the real population's. The real excess is flat at ~+8 per
decile and *negative* in the top decile, so any cut removes real detections at
essentially the same rate as artifacts. The 627.71 needed for 1 FP/unit would leave
almost nothing.

So `mf_snr` is **not a discriminant on ZTF diffs**, joining `det_qual` (saturated at
1.00) and the `sigmag` proxy (p50 = 134 on pure FPs). That reframes the blocker: this
is not a threshold to tune, it is a **missing real-bogus classifier** for the trailed
ZTF path. Note the existing ml-clean ZTF catalog achieves its −82% false links using
ZTF's *own* RB `ml_score >= 0.8`; this detector has no equivalent.

Three ways forward, in rough order of cost:

1. **Let linking do the rejection.** heliolinc demands consistent motion across ≥3
   nights, which is a far stronger artifact filter than any per-detection cut. This is
   what the raw-vs-ml-clean comparison quantifies as *expensive but workable* (−70%
   RAM, −82% false links from cleaning first). Cheapest path: run a short window
   end-to-end and see whether purify's output is usable.
2. **Require multi-exposure units**, so the repetition filter can actually run. It is
   the only per-detection purity cut that works on structured residuals, and it is
   currently unavailable for the 47% of units with a single exposure. Costs ~half the
   sky, buys real purity.
3. **Train an RB classifier on postage stamps** (`stamps.py`, `STAMPS=1` already
   exists and was built for exactly this on the CSS side). This is the real fix and
   the one that makes the full 84.5 TB worth spending.

### 7.2 The MPC cross-check has NO POWER for ZTF — do not use it here

`mpcat_check` over the pilot's 537 detections returned **0 already-in-MPC, 100%
"genuinely new"**. That number is **meaningless**, and it is worth writing down why so
nobody quotes it.

A direct census of `mpcat.bin` over the pilot's own MJD window
(records 447074849–447146453, MJD 60555.147–60555.497) finds **71,604 catalog records
and ZERO from I41**:

```
W68 43018   F52 12971   F51 8790   G96 3300   T08 1707   W84 1192
W24   210   T05   137   H21   53   I52   42   T14   34   H01   32
```

With `-matchobscode 1` there was nothing for a ZTF detection to match, so the test was
**null by construction** — exactly the trap the CSS work hit with 703, where a pilot
night contained zero 703 records and could not be validated astrophysically. Relaxing
to `-matchobscode 0` does not rescue it either: a real asteroid would have to have been
measured by a *different* observatory within `timerad` (10 s), which essentially never
happens.

**Measurement-existence matching cannot validate ZTF on this catalog.** The reality
check has to be **orbit-based attribution** — propagate known orbits to our epochs and
match positions (the `attribute.py` / `stage1c_attribute.py` machinery from the bigrun
vetting chain), not `mpcat_check`.

The ~12% purity conclusion in §7/§7.1 stands regardless: it rests on the negation
control, which needs no external catalog.
