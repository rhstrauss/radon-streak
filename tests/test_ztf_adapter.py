#!/usr/bin/env python
"""Regression tests for the two ZTF-specific defects fixed in 2026-07.

Both were silent: one destroyed signal, the other corrupted identifiers
downstream. Neither would show up as an error, only as missing science.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "bin"))

from streakradon.adapters.ztf import (            # noqa: E402
    DEFAULT_BAD_BITS, SOURCE_BITS, bad_pixel_mask, mask_path_for,
    mid_mjd_from_hdr)
from run_ztf_sequence import MAX_IDSTRING, idstring_for   # noqa: E402


def test_source_bits_are_not_masked():
    """ZTF mskimg bit 1 = CONTAINS SEXTRACTOR DETECTION, bit 11 = PSF-EXTRACTED
    SOURCE POSITION. These mark REAL SOURCES. The original adapter used
    `mask != 0`, which NaN'd them -- blanking the trails we are hunting whenever
    the object was bright enough to be catalogued in the science frame."""
    mask = np.zeros((4, 4), np.int32)
    mask[0, 0] = 1 << 1            # sextractor detection  -> KEEP
    mask[0, 1] = 1 << 11           # psf-extracted source  -> KEEP
    mask[1, 0] = 1 << 12           # halo from bright star -> mask
    mask[1, 1] = 1 << 8            # saturated             -> mask
    mask[2, 0] = 1 << 0            # aircraft/satellite    -> mask (streak FP)
    m = bad_pixel_mask(mask, DEFAULT_BAD_BITS)

    assert m[0, 0] == 0, "sextractor-detection pixel must NOT be masked"
    assert m[0, 1] == 0, "psf-source pixel must NOT be masked"
    assert m[1, 0] == 1 and m[1, 1] == 1
    assert m[2, 0] == 1, "satellite tracks should be masked in a streak detector"

    # And the naive version really would have thrown the sources away.
    assert (mask != 0)[0, 0] and (mask != 0)[0, 1]


def test_source_bits_disjoint_from_bad_bits():
    assert not (set(SOURCE_BITS) & set(DEFAULT_BAD_BITS))


def test_mask_bits_are_configurable():
    mask = np.array([[1 << 1]], np.int32)
    assert bad_pixel_mask(mask, (1,))[0, 0] == 1     # opt in explicitly
    assert bad_pixel_mask(mask, DEFAULT_BAD_BITS)[0, 0] == 0


def test_shutter_midpoint_preferred_over_obsmjd():
    """The diff header carries SHUTOPEN/SHUTCLSD, so no sciimg download is needed.
    OBSJD/OBSMJD are shutter-open ROUNDED ~0.7 s early, which is ~1.5" of
    along-track error at 134"/min."""
    hdr = {"SHUTOPEN": "2024-09-01T03:16:33.430",
           "SHUTCLSD": "2024-09-01T03:17:03.436",
           "OBSMJD": 60554.1364815, "EXPTIME": 30.0}
    mid = mid_mjd_from_hdr(hdr)
    naive = hdr["OBSMJD"] + hdr["EXPTIME"] / 2 / 86400.0
    assert mid != naive
    # Measured on this real header: the true midpoint is ~1.43 s later.
    assert 1.0 < (mid - naive) * 86400.0 < 2.0

    # Fallback when the shutter keywords are absent.
    assert mid_mjd_from_hdr({"OBSMJD": 60554.0, "EXPTIME": 30.0}) == \
        pytest.approx(60554.0 + 15.0 / 86400.0)


def test_idstrings_fit_heliolinx_shortstringlen():
    """heliolinx hldet.idstring is char[20] and its reader
    `stringncopy01(idstring, s, SHORTSTRINGLEN)` TRUNCATES SILENTLY. A natural ZTF
    id is 56 chars, which would collapse to a single value per exposure and
    destroy idstring-based dedup and any join back to the catalog."""
    p = ("/x/ztf_20240901136065_000577_zr_c01_o_q1_scimrefdiffimg.fits.fz")
    ids = [idstring_for(p, i) for i in (0, 1, 42, 999, 50000)]
    for s in ids:
        assert len(s) <= MAX_IDSTRING, (s, len(s))
    assert len(set(ids)) == len(ids), "candidate index must vary the id"


def test_idstrings_unique_across_quadrants_and_epochs():
    base = "/x/ztf_%s_000577_zr_c%02d_o_q%d_scimrefdiffimg.fits.fz"
    seen = set()
    for ffd in ("20240901136065", "20240901139317"):
        for ccd in (1, 2, 16):
            for q in (1, 2, 3, 4):
                s = idstring_for(base % (ffd, ccd, q), 0)
                assert len(s) <= MAX_IDSTRING
                seen.add(s)
    assert len(seen) == 2 * 3 * 4, "readout channel + epoch must both vary the id"


def test_unparseable_name_still_fits():
    s = idstring_for("/x/some_hand_staged_file_with_a_very_long_name.fits", 7)
    assert len(s) <= MAX_IDSTRING


def test_mask_path_derivation():
    assert mask_path_for("/a/b_scimrefdiffimg.fits.fz") == "/a/b_mskimg.fits"
    assert mask_path_for("/a/b_scimrefdiffimg.fits") == "/a/b_mskimg.fits"
    with pytest.raises(ValueError):
        mask_path_for("/a/b_sciimg.fits")
