#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ryder H. Strauss
"""Config loading for streak_radon.

Survey parameters live in YAML under the repo `config/` directory (shipped with
the package). `load_cfg` accepts either an explicit path or a bare survey name
("g96", "ztf", "atlas", "generic") which resolves to `config/<name>.yaml`.

Loud by design: a missing/broken config raises rather than silently falling
back to library defaults. (The old per-script loaders swallowed a missing PyYAML
and reverted `mf_snr_min` from the frozen 11.0 to the permissive default 6.0 --
a silent sensitivity/purity change. That footgun is removed here.)
"""
import os

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PKG_ROOT, "config")


def config_path(survey_or_path):
    """Resolve a survey name or path to a concrete YAML path."""
    if survey_or_path is None:
        raise ValueError("no config specified")
    if os.path.sep in survey_or_path or survey_or_path.endswith((".yaml", ".yml")):
        p = survey_or_path
    else:
        p = os.path.join(CONFIG_DIR, survey_or_path + ".yaml")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"config not found: {p!r} "
            f"(known surveys in {CONFIG_DIR}: "
            f"{', '.join(sorted(n[:-5] for n in os.listdir(CONFIG_DIR) if n.endswith('.yaml')))})"
        )
    return p


def load_cfg(survey_or_path):
    """Load a survey config dict. Raises on missing PyYAML or missing file."""
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "PyYAML is required to load survey configs; `pip install pyyaml` "
            "(or install streak_radon, which depends on it)."
        ) from e
    with open(config_path(survey_or_path)) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {survey_or_path!r} did not parse to a mapping")
    return cfg
