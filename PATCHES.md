# pyradon deviations

Vendored clone: github.com/guynir42/pyradon @ 6a1a43096b30ba99cdfea5593dc5cad0f35d36a3
(2024-09-14, "Add a demo notebook using real images (#15)").

**Source patches: NONE.** All known pyradon gotchas are handled by configuration
and by driving the `Finder` manually (see `streakradon/frt_driver.py`):

| Gotcha (2026-07 benchmark) | Handling |
|---|---|
| `min_length=32` default gates out the folding where short-trail SNR peaks (a000001: SNR 6.3 vs 19.0) | `pars.min_length = 8` (physical: 8-16 px trails = the >=6 deg/day G96 regime) |
| Erratic absolute SNR near bright sources (scalar-variance normalization) | pre-whiten input (`image/sqrt(varmap)`), `data.variance = 1.0`; pyradon SNR treated as a hint only — significance comes from `mf_snr.py` |
| Non-square images crash the FRT variance broadcast | we tile to 1024x1024 squares ourselves (`tile_grid`) |
| Length over-integration (250-345 px for ~30-70 px trails) | `mf_snr.refine_length` re-estimates L from the along-line profile before the Veres fit; final length always from the fit |
| `input(variance=<ndarray>)` crash (older clone) | not exercised — manual drive never calls `input()`; this clone's variance setter is ndarray-safe anyway |
