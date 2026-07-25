# pyradon deviations

Vendored clone: github.com/guynir42/pyradon @ 6a1a43096b30ba99cdfea5593dc5cad0f35d36a3
(2024-09-14, "Add a demo notebook using real images (#15)").

**Vendor source patches: NONE** — the vendored tree stays byte-identical to
upstream. Gotchas are handled by configuration, by driving the `Finder` manually
(`streakradon/frt_driver.py`), and by ONE *runtime* monkey-patch that lives in
our package (`streakradon/fastmask.py`, see "Performance patch" below):

| Gotcha (2026-07 benchmark) | Handling |
|---|---|
| `min_length=32` default gates out the folding where short-trail SNR peaks (a000001: SNR 6.3 vs 19.0) | `pars.min_length = 8` (physical: 8-16 px trails = the >=6 deg/day G96 regime) |
| Erratic absolute SNR near bright sources (scalar-variance normalization) | pre-whiten input (`image/sqrt(varmap)`), `data.variance = 1.0`; pyradon SNR treated as a hint only — significance comes from `mf_snr.py` |
| Non-square images crash the FRT variance broadcast | we tile to 1024x1024 squares ourselves (`tile_grid`) |
| Length over-integration (250-345 px for ~30-70 px trails) | `mf_snr.refine_length` re-estimates L from the along-line profile before the Veres fit; final length always from the fit |
| `input(variance=<ndarray>)` crash (older clone) | not exercised — manual drive never calls `input()`; this clone's variance setter is ndarray-safe anyway |

## Performance patch: `Streak.subtract_streak` (runtime, `streakradon/fastmask.py`)

**What upstream does.** After each streak is found, `subtract_streak` blanks its
pixels so the next iteration finds a different one:

```python
mask = model(im.shape, x1, x2, y1, y2, psf_sigma) > 0   # 4x-oversampled full-tile
im[mask] = replace_value                                #  model + convolve2d
```

`model()` builds a 4x-oversampled, PSF-convolved photometric model of the whole
tile (4096^2 for a 1024^2 tile) **purely to threshold it into a boolean
footprint** — every model VALUE is discarded. Profiling (2026-07) put this at
**46% of total pipeline runtime**: ~1.2 s per streak, ~201 s per 5280^2 exposure.

**What we substitute.** `model()>0` produces a CAPSULE — pixels within a fixed
perpendicular distance of the line *segment*. Measured against upstream, the
halfwidth is `~1.69 + 6.38*psf_sigma` px; we use a slightly generous
`2.0 + 6.5*psf_sigma` so we can never *under*-mask (under-masking would let the
next iteration re-find the same streak and change detection behaviour).

**Validation.** Over 40 random (geometry, psf_sigma) combinations, **zero**
upstream footprint pixels are missed (full coverage; ~25% extra, the safe
direction). On a real 1024^2 tile the candidate set is unchanged except for
occasional *extra* weak candidates (snr_frt ~10 against 900+ for real trails),
which the downstream `mf_snr_min` + RB vetting removes by design — no candidate
is ever lost. Measured **5.5x** on the detect stage.

Disable with `make_finder(..., fast_suppress=False)` /
`detect: {fast_suppress: false}`, or `fastmask.uninstall_fast_suppression()`.

**Gotcha for future patchers:** `finder.py` does `from streak import Streak`
while our package does `from src.streak import Streak`. Python treats `streak`
and `src.streak` as *separate module objects for the same file*, each with its
own class — patching one silently leaves the finder's copy untouched (this
produced a 1.0x "speedup" until profiling caught it). `fastmask` patches every
loaded instance.
