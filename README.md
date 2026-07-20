# streak_radon — FRT-based trailed-source detection for heliolinx

State-of-the-art trailed-source (streak) detection for solar-system fast movers,
built on the Fast Radon Transform (pyradon, Nir & Ofek 2018) with the validated
Veres (2012) forward-model measurement stage from ztf_streak. Produces
`hldet_colformat01` detection catalogs for heliolinx `make_trailed_tracklets`.

**Architecture: FRT finds, Veres fit measures.** The FRT is a near-optimal blind
locator over all lengths/angles, but its length/SNR measurements are unreliable
on real survey diffs (benchmarked 2026-07 on a000001 ground truth). So pyradon
runs permissively as a candidate generator on pre-whitened tiles; every
candidate is then re-measured by the forward fit and re-scored with our
validated integrated matched-filter SNR. See PATCHES.md for the pyradon
gotchas and how each is handled (no source patches).

## Environment

    PY=/astro/store/shire/rstrau/miniforge3/envs/mpchecker/bin/python
    cd /astro/store/shire/rstrau/streak_radon
    $PY tests/test_units.py            # unit checks
    $PY bin/regress_a000001.py         # P1 regression gate (ZTF ground truth)

scipy/astropy-only by design (no reproject/photutils/sep needed).

## Surveys

| adapter | input | status |
|---|---|---|
| g96 | CSS `arch.fz` 4-visit sequences (within-sequence leave-one-out median differencing) | primary target |
| ztf | IRSA `scimrefdiffimg` + `mskimg` | validation (a000001 truth; ztf_streak MF = incumbent) |
| atlas | reduced/diff frames via public forced-photometry service | comparison vs `trailmain` catalogs |

## Layout

    vendor/pyradon/       pinned clone (SHA in PATCHES.md)
    streakradon/          library: frt_driver, mf_snr, trail_fit, rb, psf,
                          register, template, varmap, inject, hldet_io, adapters/
    bin/                  orchestrators + regression/efficiency/e2e scripts
    config/               per-survey yaml + hldet_colformat01.txt
    qa/                   QA plots from preprocessing runs

## Output contract

15-column `hldet_colformat01` CSV; trail_len (arcsec) and trail_PA (deg, motion
convention) are load-bearing: make_trailed_tracklets keeps a pair only when both
endpoints' measured trail geometry matches the pair-implied motion within
`-siglenscale`/`-sigpascale`. RA/Dec/MJD = trail center at mid-exposure.

Provenance: ztf_streak (validated 2026-06, a000001), pyradon benchmark 2026-07.
G96 example images: /astro/store/shire/aheinze/Shared/G96_images (A. Heinze).
