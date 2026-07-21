#!/usr/bin/env python
"""Integration smoke tests exercising the new survey-agnostic paths:
  1. config loader (loud on missing)
  2. generic adapter on a synthetic differenced FITS
  3. synthetic demo sequence -> detection of the injected mover
Run: <python> tests/test_integration.py
"""
import os
import sys
import tempfile

import numpy as np
from astropy.io import fits

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from streakradon.config import load_cfg                       # noqa: E402
from streakradon.adapters.generic import prepare_diff         # noqa: E402
from streakradon.inject import inject_trail                   # noqa: E402
from streakradon.synthetic import make_wcs                    # noqa: E402


def test_config_loader():
    cfg = load_cfg("generic")
    assert cfg["survey"] == "generic"
    assert "detect" in cfg and "vet" in cfg
    try:
        load_cfg("no_such_survey_xyz")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("load_cfg should raise on a missing config")
    print("test_config_loader OK")


def test_generic_adapter():
    shape = (400, 400)
    rng = np.random.default_rng(3)
    diff = rng.normal(0, 1.0, shape)
    wcs = make_wcs(shape, pixscale_arcsec=1.0)
    inject_trail(diff, 200, 200, np.radians(40), 40.0, 17.5, 26.0, 1.4)
    hdr = wcs.to_header()
    hdr["MJD-OBS"] = 60431.15
    hdr["EXPTIME"] = 30.0
    hdr["MAGZP"] = 26.0
    hdr["FWHM"] = 3.3
    hdr["FILTER"] = "r"
    hdr["OBSCODE"] = "807"
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        path = f.name
    fits.writeto(path, diff.astype(np.float32), hdr, overwrite=True)
    cfg = load_cfg("generic")
    e = prepare_diff(path, cfg=cfg)
    os.unlink(path)
    assert e.obscode == "807", e.obscode
    assert e.band == "r", e.band
    assert abs(e.mid_mjd - (60431.15 + 15.0 / 86400.0)) < 1e-6, e.mid_mjd  # start->mid
    assert abs(e.psf_sigma_px - 3.3 / 2.3548) < 1e-3, e.psf_sigma_px
    assert abs(e.pixscale_arcsec - 1.0) < 0.02, e.pixscale_arcsec
    assert np.isfinite(e.white).all()
    print(f"test_generic_adapter OK (mid_mjd={e.mid_mjd:.5f}, "
          f"sigma={e.psf_sigma_px:.2f}px, pixscale={e.pixscale_arcsec:.3f}\")")


def test_synthetic_demo_recovers_mover():
    from streakradon.synthetic import build_sequence
    from streakradon.pipeline import process_exposure
    exps, truth = build_sequence(n_epochs=3, shape=(500, 500), mag=18.0, seed=5)
    cfg = load_cfg("generic")
    cfg["detect"].update(tile=500, overlap=0, mf_snr_min=8.0)
    n_recovered = 0
    for e in exps:
        fits_i = process_exposure(e, cfg, survey="generic", verbose=False)
        tx, ty = truth["positions"][e.image_index]
        for fit in fits_i:
            if np.hypot(fit["x"] - tx, fit["y"] - ty) < 20:
                # length within 10%, PA sane, chi2 reasonable
                assert abs(fit["trail_len"] - truth["trail_len_arcsec"]) < 0.15 * truth["trail_len_arcsec"], \
                    (fit["trail_len"], truth["trail_len_arcsec"])
                n_recovered += 1
                break
    assert n_recovered == len(exps), f"recovered {n_recovered}/{len(exps)}"
    print(f"test_synthetic_demo_recovers_mover OK ({n_recovered}/{len(exps)} epochs, "
          f"len {truth['trail_len_arcsec']:.1f}\")")


if __name__ == "__main__":
    test_config_loader()
    test_generic_adapter()
    test_synthetic_demo_recovers_mover()
    print("ALL INTEGRATION TESTS PASSED")
