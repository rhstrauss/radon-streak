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

### (iii-G96) / (iv) / (P4) — [efficiency + end-to-end: pending completion]

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
