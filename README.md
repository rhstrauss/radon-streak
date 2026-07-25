# streak_radon

**FRT-based trailed-source (streak) detection for solar-system fast movers.**

`streak_radon` turns difference images from *any* optical survey into a
`hldet_colformat01` detection catalog of **trailed sources** — the streaked
detections that fast-moving near-Earth objects leave in a single exposure —
ready for heliolinx [`make_trailed_tracklets`](https://github.com/lsst-dm/heliolinc2).

It pairs a near-optimal blind detector with a validated forward-model measurer:

> **FRT finds, Veres fit measures.** Our own clean-room Fast Radon Transform
> (`streakradon/frt.py`; algorithm of Nir & Ofek 2018, Brady 1998) scans every
> length and angle to *locate* candidate streaks; each candidate is then
> re-measured by a Veres et al. (2012) line⊗Gaussian forward-fit and re-scored
> with an integrated matched-filter SNR. The FRT is a great *detector* but an
> unreliable *measurer* on real survey diffs — so we use each stage for what it
> is good at.

Validated 2026-07 on ZTF ground-truth NEO **a000001** (recovered blind as the #1
detection), real ATLAS difference stamps of **2024 KV**, and CSS/G96 imagery; its
detections link cleanly through the real heliolinx trailed-tracklet stage. Full
numbers in [`REPORT.md`](REPORT.md).

---

## Install

The detector is our own clean-room Fast Radon Transform (`streakradon/frt.py`) —
no third-party detection code is required at runtime. Install from a clone in
editable mode (the survey configs under `config/` resolve relative to the repo):

```bash
git clone <your-fork-url> streak_radon
cd streak_radon
pip install -e .            # numpy, scipy, astropy, pyyaml, matplotlib
```

Licensing: GPL-3.0-or-later while the optional `vendor/pyradon` (GPLv3; Guy Nir)
is present. It is retained only as an A/B backend (`detect.backend: pyradon`);
the default `native` backend does not use it. See [`LICENSING.md`](LICENSING.md)
and [`NOTICE`](NOTICE).

That puts a `streak-radon` command on your PATH. No compiled extensions; pure
Python. (Tested on numpy 1.24–2.4 / scipy 1.10–1.17 / astropy 6–7.)

Verify:

```bash
streak-radon selftest      # unit checks: hldet columns, tiling, dedup, MF SNR
```

---

## Quickstart — synthetic data (no downloads)

The fastest way to see the whole chain work is the built-in synthetic demo. It
injects one linearly-moving trailed source across a short rapid-cadence sequence
of synthetic difference images, detects and measures it, and writes an hldet CSV:

```bash
streak-radon demo --out-dir demo_out --mosaic
```

Expected output (abridged):

```
synthetic sequence: 4 epochs, 800x800px, mover mag 18.5, trail 50.0" (50px), rate 40 deg/day
  epoch 0: 1 pass vetting; mover -> SNR 58.0  len 49.4" (truth 50.0)  center off 0.1px  chi2r 0.95
  epoch 1: 1 pass vetting; mover -> SNR 58.9  len 49.8" (truth 50.0)  center off 0.2px  chi2r 0.99
  epoch 2: 1 pass vetting; mover -> SNR 59.6  len 49.7" (truth 50.0)  center off 0.0px  chi2r 0.95
  epoch 3: 1 pass vetting; mover -> SNR 57.2  len 49.9" (truth 50.0)  center off 0.2px  chi2r 0.96

4 detections -> demo_out/demo_trails.csv
QA mosaic -> demo_out/demo_mosaic.png
```

Every epoch recovers the mover with sub-pixel centering and ~1 % length error —
the metric fidelity the linker's trail-consistency gates need.

`demo_out/demo_mosaic.png` shows the whitened diff for each epoch with the
injected mover circled. To also run the real linker (if you have a heliolinx
build):

```bash
streak-radon demo --out-dir demo_out --link \
    --mtt   /path/to/heliolinx/bin/make_trailed_tracklets \
    --earth /path/to/Earth1day2020s_02a.csv \
    --obscodes /path/to/ObsCodes.html
# or set env MAKE_TRAILED_TRACKLETS / HELIO_EARTH / HELIO_OBSCODES
```

---

## Real data — the plug-and-play path (any survey)

If your pipeline already produces **reference-subtracted (difference) FITS
images** with a WCS and a magnitude zero point, you do **not** need a bespoke
adapter. Describe your header keywords once in a config and run:

```bash
cp config/generic.yaml my_survey.yaml
$EDITOR my_survey.yaml           # point mjd_key/magzp_key/psf_key/... at your headers

streak-radon detect --survey generic --config my_survey.yaml \
    --diff  night1/diff_0001.fits \
    --mask  night1/mask_0001.fits \        # optional (nonzero = bad pixel)
    --out   night1/trails_0001.csv \
    --imgs  night1/imgs.txt                 # optional image log for the linker
```

The only thing that changes between surveys is the header map in the YAML — the
detect → measure → vet → output stages are identical. The knobs you will most
likely tune are `detect.mf_snr_min` (detection threshold) and `vet.axis_blanket`
(reject all detector-aligned trails); see **Tuning** below.

### Built-in survey adapters

Some inputs need survey-specific handling, so they get dedicated adapters:

| survey | input | why not generic |
|---|---|---|
| **`generic`** | any differenced FITS + optional mask | — (this is the default) |
| **`ztf`** | IRSA `scimrefdiffimg` (+ `mskimg`, + sci header) | exact shutter-midpoint timing |
| **`atlas`** | forced-photometry `_diff` stamps | broken stamp WCS (TPV+SIP) → linear-TAN fallback |
| **`g96`** (CSS) | `arch.fz` 4-visit sequences (**not** differenced) | within-sequence leave-one-out differencing |

```bash
# ZTF quadrant difference (science header gives the validated timing)
streak-radon detect --survey ztf \
    --diff diff.fits.fz --mask mask.fits --sci sci.fits --out ztf_trails.csv

# ATLAS difference stamp
streak-radon detect --survey atlas --diff 60509o0528_diff.fits --out klv.csv
```

### Un-differenced sequences (CSS / G96)

CSS `arch.fz` frames are not differenced, so a whole visit sequence is
registered and leave-one-out median-differenced first. That is a different entry
point (it consumes N frames and applies a cross-exposure repetition filter):

```bash
streak-radon detect-sequence --config g96 \
    --frames 'G96_20240501_2B_N25057_01_000?.arch.fz' \
    --out g96_trails.csv --imgs g96_imgs.txt --qa qa/
```

---

## Output contract

A 15-column `hldet_colformat01` CSV (see [`config/hldet_colformat01.txt`](config/hldet_colformat01.txt)):

```
#MJD,RA,Dec,mag,trail_len,trail_PA,sigmag,sig_across,sig_along,image,idstring,band,obscode,known_obj,det_qual
```

- **RA/Dec/MJD** = trail **center at mid-exposure** (the convention
  `make_trailed_tracklets` expects).
- **`trail_len`** (arcsec) and **`trail_PA`** (deg, motion convention) are
  load-bearing: the linker keeps a pair only when both endpoints' measured trail
  geometry matches the pair-implied motion within `-siglenscale`/`-sigpascale`.
  These come from the Veres fit and are accurate to ~2 % in length, ~0.2° in PA
  (validated; see `REPORT.md`).

Feed it straight to the linker:

```bash
make_trailed_tracklets -dets trails.csv -imgs imgs.txt \
    -colformat config/hldet_colformat01.txt \
    -earth Earth1day2020s_02a.csv -obscode ObsCodes.html \
    -tracklets tracklets.csv -trk2det trk2det.csv -pairdets pairdets.csv ...
```

---

## Tuning for a new survey

The one parameter that always needs calibration is the **detection threshold**
`detect.mf_snr_min` — the integrated matched-filter SNR floor. Set it from a
**negated-difference false-positive calibration**: run detection on the negated
(source-free) diff and pick the threshold that gives an acceptable FP rate. On
single 4-visit CSS diffs that lands at `mf_snr_min = 11`; on cleaner multi-epoch
references it can be lower. Other useful knobs, all in the survey YAML:

| key | meaning |
|---|---|
| `detect.mf_snr_min` | integrated MF SNR floor (the sensitivity/purity dial) |
| `detect.min_length` | physical minimum trail length in px (keep at 8; the pyradon default 32 is a trap that halves short-trail SNR) |
| `detect.frt_threshold` | permissive FRT candidate threshold (significance comes from the MF re-score, so keep this low) |
| `vet.axis_blanket` | `true` → reject **all** near-0°/90° trails (use only if your diffs are detector-artifact dominated, like ZTF); `false` → reject only trails adjacent to masked bleed/saturation |
| `header.mjd_convention` | `start` (shutter-open → mid = MJD + exptime/2) or `mid` — get this wrong and your astrometry epoch is off by half an exposure |

See [`PATCHES.md`](PATCHES.md) for the pyradon gotchas and how each is handled
(all by configuration; no source patches).

---

## How it works

```
difference image
  → whiten           (img - local_bg)/sqrt(var)   → unit-noise detection image
  → FRT tiles        native clean-room FRT on square tiles, min_length=8, permissive
  → dedup            union-find across overlapping tiles / foldings
  → MF grid-refine   maximize integrated matched-filter SNR over (PA, L, center)
  → Veres fit        line⊗Gaussian forward model → RA/Dec, mag, trail_len, PA, errors
  → real/bogus vet   length/χ²/edge/dipole-flank cuts (+ cross-exposure repetition)
  → hldet_colformat01 CSV
```

The MF grid-refinement is the key step: the raw FRT PA is quantized and its
length lands on a power-of-2 folding ladder, so a few-degree PA error
decorrelates a long thin template. Re-fitting against the statistic that matters
took a000001 from FRT SNR 6.3 → 19.4 at the correct length.

---

## Layout

```
streakradon/            importable library (survey-agnostic core + adapters/)
  frt.py                clean-room Fast Radon Transform (detector; our own code)
  frt_driver.py         run the FRT over a whole frame (tiling, dedup)
  mf_snr.py             integrated matched-filter SNR + grid re-measurement
  trail_fit.py          Veres line⊗Gaussian forward fit
  rb.py                 real/bogus vetting + cross-exposure repetition test
  varmap.py             variance maps, masks, pre-whitening
  register.py template.py psf.py    preprocessing (registration, differencing, PSF)
  inject.py             synthetic trail injection (analytic + empirical-PSF)
  synthetic.py          self-contained demo sequence generator
  hldet_io.py           hldet_colformat01 writer
  config.py             survey-config loader
  cli.py                the `streak-radon` command
  adapters/             generic, ztf, atlas, g96  → DiffExposure
bin/                    validation / reproduce harnesses (regression, efficiency, e2e)
config/                 per-survey YAML + hldet_colformat01.txt
tests/test_units.py     unit tests (incl. native-FRT correctness + detection)
REPORT.md               full validation report
```

## License & citation

Currently **GPL-3.0-or-later** (see [`LICENSE`](LICENSE), [`NOTICE`](NOTICE)). The
detector is our own clean-room FRT (`streakradon/frt.py`); no third-party
copyleft code remains, so the project may be relicensed at the authors'
discretion — see [`LICENSING.md`](LICENSING.md). If you use this tool, please cite
**Nir et al. 2018, AJ 156, 229** (the Fast Radon Transform algorithm) and
**Veres et al. 2012, PASP 124, 1197** (the trailed-source model).
