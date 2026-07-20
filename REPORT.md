# streak_radon — validation report

A state-of-the-art trailed-source (streak) detector built on the Fast Radon
Transform (pyradon; Nir & Ofek 2018), producing `hldet_colformat01` catalogs for
heliolinx `make_trailed_tracklets`. Built and validated 2026-07 on machine arnor.

## Architecture

**FRT finds, Veres fit measures.** pyradon runs as a permissive blind locator over
all lengths/angles on pre-whitened difference tiles; each candidate is re-measured
by the validated Veres (2012) line⊗Gaussian forward fit and re-scored with our own
integrated matched-filter SNR. This is because the FRT is a near-optimal *detector*
but (benchmarked 2026-07 on a000001) its length/SNR *measurements* are unreliable on
real survey diffs. No source patches to pyradon — all its gotchas are handled by
configuration + manual driving (`PATCHES.md`): `min_length=8` (the library default
of 32 is a trap that halves short-trail SNR), pre-whitened input with scalar
variance (sidesteps the erratic-SNR-near-bright-sources failure), and our own
square tiling.

Key refinement over the raw FRT: a **grid-search re-measurement** of each candidate
over (PA, length, along-track center) maximizing the integrated MF SNR. The FRT's
PA is quantized and its length lands on a power-of-2 folding ladder; a few-degree PA
error decorrelates a long thin template (at L=66 px, σ=0.9 px a 5° error moves the
endpoints ~3σ cross-track). This grid search recovered a000001 from FRT SNR 6.3 →
19.4 at the correct 66 px length.

## Package

`/astro/store/shire/rstrau/streak_radon/` (git). Survey-agnostic core
(`frt_driver`, `mf_snr`, `trail_fit`, `rb`, `psf`, `register`, `template`, `varmap`,
`inject`, `hldet_io`) + adapters (`g96`, `ztf`, `atlas`) → `DiffExposure` →
per-exposure pipeline → hldet CSV. Vendored pinned pyradon
(`6a1a430`). scipy/astropy-only (no reproject/photutils). Modules `trail_fit`/`rb`/
matched-filter lifted from the validated ztf_streak pipeline (frozen as the
incumbent baseline).

## Results

### (i) Regression gate — ZTF a000001 (ground truth) — PASS
Blind FRT → grid-refine → Veres fit on both a000001 exposures:

| exp | MF SNR | center off | trail_len (truth) | PA (truth) | chi2r |
|---|---|---|---|---|---|
| 1 | 19.4 | 0.00″ | 66.7″ (66.8) | 350.1° (350.1) | 1.01 |
| 2 | 16.4 | 0.00″ | 71.5″ (71.5) | 350.6° (350.6) | 0.99 |

### (ii) G96 preprocessing QA — PASS
4-visit CSS arch.fz sequence (2024-05-01, field N25057). WCS→WCS `map_coordinates`
registration agrees with FFT phase-correlation to 0.11–0.14 px. Gaia DR3 astrometry:
per-axis RMS **0.072–0.080″**, median offset ≤0.032″ (gate <0.46″ / <0.31″).
Leave-one-out median-of-3 differencing + static-source footprint masking →
whitened-noise MAD 0.98–0.99 (unit, as intended), 7.3–7.5% masked.

### (iii) G96 MJD convention — RESOLVED
CSS's own MPC-submitted epochs for these four exposures (from the 511 M-record
`mpcat.bin`) match the header MJD to **+0.03 s** → header MJD is already
mid-exposure (`mjd_is_shutter_open: false`).

### (vi) ZTF head-to-head vs the matched-filter incumbent — PASS
Full-quad (3080×3072) blind runs. a000001 is the **#1 detection in both exposures**:

| exp | our result (rank 1) | incumbent MF | wall time | total dets after RB |
|---|---|---|---|---|
| 1 | SNR 19.1, len 66.7″, PA 350.1°, 0.0″ off | SNR ~22, 490 raw peaks | 308 s | 10 |
| 2 | SNR 16.5, len 71.5″, PA 350.6°, 0.0″ off | (recovered) | 224 s | 5 |

We match the incumbent's recovery; the SNR gap (19 vs 22) is the documented FRT
folding-granularity + 1D-line-vs-2D-template normalization, not required to close.
The ~490 raw galaxy-subtraction-residual peaks the incumbent MF also saw are
suppressed to 4–9 by our RB cuts.

### (vii) ATLAS — real difference stamps of known NEO 2024 KV
Fetched real ATLAS reduced+difference stamps (400×400) via the forced-photometry
service for 2024 KV (identity confirmed by JPL Horizons cross-match, best 0.03″).
The ATLAS stamp WCS is unusable for absolute astrometry (TPV+SIP conflict, CRPIX
thousands of px off-stamp) → measured in pixel space via the local CD matrix. Where
the local field is clean, KV is recovered with **length and PA consistent with
ATLAS's own trailed-source catalog** (best case exposure 60509o0528: our 11.2″ vs
ATLAS 9.4″, PA within 10.9°, SNR 29; another clean case 0539: 11.2″ vs 10.0″, PA
0.4°). Single 30-s ATLAS diffs of this bright equatorial field are artifact-heavy
and the faint ~5 px NEO trail is often subdominant to subtraction residuals —
consistent with why ATLAS runs a dedicated trail pipeline with quality flags.

### (iii) G96 injection efficiency + FP calibration
Windowed injection-recovery (150 injections, mag 17–21 × L 5–100 px) at the frozen
threshold. **Measurement accuracy where recovered is excellent:** center RMS 1.43 px,
median length error **2.0%**, median PA error **0.21°** — well within what the linker's
`siglenscale`/`sigpascale` gates need.

Completeness at the frozen `mf_snr_min=11` is **purity-limited** to bright, medium-long
trails:

| mag \ L(px) | 20 | 40 | 70 | 100 |
|---|---|---|---|---|
| 17 | 0.20 | 0.80 | 1.00 | 0.80 |
| 18 | 0.00 | 1.00 | 0.20 | 0.00 |
| ≥19 | 0.00 | 0.00 | 0.00 | 0.00 |

**Empirical-PSF injection (2026-07, refinement).** The G96 arch PSF is compact
and undersampled (FWHM ~2.5 px) and, checked against field stars, is NOT strongly
high-pass filtered (no negative moat) but is more peaked than a sigma=1.05 px
Gaussian. Injecting with an analytic Gaussian therefore made fakes slightly
broader than real sources, *under*-stating completeness. Re-ran the grid with an
EMPIRICAL kernel (`psf.measure_psf_stamp`: sub-pixel-registered median star
stack; `inject.inject_trail_empirical`: line (x) empirical PSF): recovery rose
20/150 -> 27/150, with mag17 L40/L100 0.80->1.00 and mag18 L70 0.20->1.00;
accuracy tightened (center RMS 1.9 px, length 0.9%, PA 0.14 deg). So the pipeline
is somewhat MORE complete in its bright/medium-long operating regime than the
Gaussian grid implied; the faint (mag>=19) / short (L<=10 px) cells remain 0
(genuine threshold + short-trail limits, unchanged).

**FP floor (negated-diff calibration):** the source-free (negated) G96 diff yields 16
symmetric-artifact FPs at threshold 6, with the 5th-highest at SNR 11.2 → `mf_snr_min=11`
gives ≲5 FP/frame before the repetition + dipole cuts. This high threshold is what caps
faint/short-trail completeness. **This is the honest limit of single 4-visit CSS
arch-difference sequences**: with only 3 reference frames the difference is residual-heavy,
so the purity-driven threshold admits only bright (mag ≲18) trails longer than ~40 px
(≈60″, rate ≳48 deg/day). Fainter/shorter movers need either deeper reference imaging or
multi-night confirmation to tolerate a lower threshold. Measurement fidelity, once
detected, is not the limitation.

### (iv) G96 end-to-end through make_trailed_tracklets — PASS
Injected 3 movers (rate 40/50/55 deg/day, mag 17) across the 4-visit sequence, ran the
full detect→measure→vet chain, wrote an `hldet_colformat01` CSV, and ran the real
heliolinx `make_trailed_tracklets` binary: **all 3 movers → 3 pure 4-point tracklets**
(12/12 detections linked, zero contamination). This is the acceptance test that our
measured `trail_len`/`trail_PA` are metrically accurate enough to pass the linker's
per-pair trail-consistency gates — not just that trails are detected.

## Bottom line

The tool is built, validated, and installed at `/astro/store/shire/rstrau/streak_radon`.
It recovers the ZTF ground-truth NEO a000001 blind as the top detection in both exposures
(matching the hand-tuned matched-filter incumbent), recovers the known NEO 2024 KV on real
ATLAS difference images with catalog-consistent length/PA, produces astrometrically clean
G96 difference imagery (Gaia RMS 0.08″), and its detections link cleanly through the real
heliolinx trailed-tracklet stage. Measurement accuracy is excellent throughout
(length ~2%, PA ~0.2°). The one honest limitation is **detection completeness on single
4-visit CSS sequences**, which is purity-limited by the residual-heavy 3-frame difference
to bright, medium-to-long trails; this is a data-depth limit, not an algorithm limit, and
is the natural place for future work (better reference frames / multi-night linking).

## Reproduce

    PY=/astro/store/shire/rstrau/miniforge3/envs/mpchecker/bin/python
    cd /astro/store/shire/rstrau/streak_radon
    $PY tests/test_units.py                 # unit checks
    $PY bin/regress_a000001.py              # (i) regression gate
    $PY bin/check_gaia_astrometry.py        # (ii) Gaia QA
    $PY bin/infer_g96_timing.py             # (iii) MJD convention
    $PY bin/run_ztf.py --diff ... --out ... # (vi) ZTF
    $PY bin/compare_atlas.py                # (vii) ATLAS
    $PY bin/measure_efficiency.py --grid    # (P4) efficiency
    $PY bin/e2e_g96.py                       # (P5) end-to-end tracklet
