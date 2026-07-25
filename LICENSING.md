# Licensing & attribution

## Summary

`streak_radon` is currently licensed **GPL-3.0-or-later** (see `LICENSE`,
`NOTICE`). The detection engine is our **own clean-room Fast Radon Transform**
(`streakradon/frt.py`). The previously vendored GPLv3 `pyradon` has been
**removed** (`git rm -r vendor/pyradon`), so the distribution now contains **no
third-party copyleft source**. The project is therefore free to be **relicensed**
(e.g. BSD-3-Clause) at the authors' discretion — see "Relicensing" below. Until
that decision is made it remains GPLv3, which is a safe default.

## What is whose

| Component | Author / copyright | Status |
|---|---|---|
| Fast Radon Transform **algorithm** | Götz-Druckmüller 1996; Brady 1998; Nir-Ofek 2018 | Published mathematics — not copyrightable; free to implement |
| `streakradon/frt.py` (our FRT **code**) | © 2026 R. H. Strauss | Independent clean-room implementation; ours to license |
| `streakradon/*` (mf_snr, trail_fit, rb, varmap, register, template, adapters, cli, …) | © 2026 R. H. Strauss | Ours |
| `vendor/pyradon/` (FRT **code**) | © 2018 Guy Nir, GPLv3 | Third-party; optional backend; the *only* reason the combined work is GPLv3 |

The distinction that matters legally: the *Radon transform* and the
*recursive-doubling FRT algorithm* are published ideas that carry no copyright,
so anyone may reimplement them. Only a specific *source expression* (Guy Nir's
`pyradon` code) is copyrighted. `frt.py` is an independent expression.

## Clean-room provenance of `streakradon/frt.py`

`frt.py` was written from the equations in the published algorithm references
(Götz-Druckmüller 1996, *Pattern Recognition* 29:711; Brady 1998, *SIAM J.
Comput.* 27:107; Nir, Ofek, Ben-Ami & Zackay 2018, *AJ* 156:229) and **not** by
reading, translating, or adapting the `pyradon` (or any other) source. pyradon
was used only as an *independent numerical oracle* for validation (does our
transform localize the same streaks?), never as a structural template.

Correctness is established by in-file self-tests (`python -m streakradon.frt` /
`streakradon/frt.py`):
1. **Exact identity** — `radon_pos` equals a direct sum over the recursively
   constructed digital line (`brady_line`) for multiple image shapes. This is a
   proof the recursion computes the intended line integrals, independent of any
   reference implementation.
2. **Linearity.**
3. **Streak recovery** — injected unit-flux lines produce peaks of the correct
   amplitude at the correct `(offset, slope)`, with `endpoints_for` recovering
   the true image-frame endpoints, in all four octants.
4. **Detector** — the multi-length folding path finds short (~30 px) trails in
   unit-variance noise at arbitrary orientation.

End-to-end, the native backend recovers injected movers through the full
pipeline (`streak-radon demo`): 4/4 epochs, SNR ~60, length within 0.3″ of truth,
center within ~0.1 px, χ²ᵣ~0.96 — with pyradon never imported.

## Done (2026-07)

1. ✅ Clean-room `streakradon/frt.py` written from the algorithm and validated
   (self-tests + unit tests + end-to-end `demo`).
2. ✅ **a000001 ground-truth regression** reproduced on the native backend
   (`bin/regress_a000001.py`): both ZTF exposures PASS — MF SNR 19.5 / 16.5,
   Veres fit center off 0.00″, length/PA within gate. Matches the pyradon-era
   numbers (19.4 / 16.4).
3. ✅ `git rm -r vendor/pyradon`; removed `streakradon/fastmask.py`, the A/B
   `bin/ab_phase1.py` + `bin/ab_diagnose.py`, and `PATCHES.md`; dropped the
   `import_pyradon` / `make_finder` code paths. No pyradon references remain.

## Relicensing (open decision)

The package may now be relicensed (e.g. BSD-3-Clause) by updating `LICENSE`,
`NOTICE`, `CITATION.cff`, and `pyproject.toml:license`. **This is a
project/institutional decision** — for a permissive public release tied to the
grant, clear it with UW tech-transfer / OSS counsel first. This document is
engineering guidance, not legal advice. Scientific-citation courtesy is
independent of the license: keep citing Nir et al. (2018) for the FRT method and
Vereš et al. (2012) for the trail fit.

Scientific-citation courtesy (independent of license): please continue to cite
Nir et al. (2018) for the FRT method and Vereš et al. (2012) for the trail fit.
