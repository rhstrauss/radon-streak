#!/usr/bin/env python
"""Adapter contract: every survey adapter produces DiffExposure objects; the
detection/measurement/vetting/output stages are survey-agnostic."""
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class DiffExposure:
    diff: np.ndarray              # difference image (ADU), NaN = bad
    white: np.ndarray             # pre-whitened diff (unit noise), bad px = 0
    var: np.ndarray               # variance map used for whitening
    mask: np.ndarray              # uint8, nonzero = bad
    wcs: Any                      # astropy WCS valid for `diff` pixel grid
    mid_mjd: float                # MID-exposure epoch (heliolinx convention)
    magzp: float
    psf_sigma_px: float
    exptime_s: float
    pixscale_arcsec: float
    obscode: str
    band: str
    idbase: str                   # detection idstring prefix
    image_index: int = -1
    meta: dict = field(default_factory=dict)
