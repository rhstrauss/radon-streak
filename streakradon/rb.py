#!/usr/bin/env python
"""Real/bogus vetting for streak candidates.

Base cuts lifted from ztf_streak/bin/rb_features.py (validated on ZTF), extended
with CSS/G96-specific tests:
  * conditional axis cut -- G96 within-sequence diffs are not dominated by
    detector-aligned artifacts the way ZTF diffs are, so the blanket 0/90 deg
    rejection only applies near bleed/saturated columns (survey='ztf' keeps it
    blanket, as validated there);
  * edge / satellite cuts (over-long trails, both endpoints near frame edge);
  * dipole-flank feature (negative flanking corridors = subtraction residual);
  * cross-exposure repetition test (static artifacts repeat at fixed sky
    position across a CSS 4-visit sequence; real movers displace >=13 px even
    at 1 deg/day over the ~8 min visit spacing).
"""
import numpy as np

DEFAULTS = dict(
    axis_tol_deg=4.0,        # detector-axis rejection tolerance (pixel frame)
    len_min_arcsec=6.0,      # plausible trail length window
    len_max_arcsec=300.0,    # beyond this = satellite/plane (288 deg/day at 30 s)
    chi2r_lo=0.3, chi2r_hi=3.0,
    mf_snr_min=6.0,          # integrated matched-filter SNR (NOT per-pixel)
    sig_along_max_arcsec=3.0,
    edge_px=30,              # both endpoints within this of an edge -> unbounded streak
    flank_ratio_max=-0.35,   # mean flank flux / on-line flux below this -> dipole
    repeat_rad_arcsec=5.0,   # cross-exposure repetition match radius
    repeat_pa_deg=10.0,
)


def _near_axis(pa_deg, tol):
    a = pa_deg % 180.0
    return min(a, abs(a - 90.0), abs(a - 180.0)) < tol


def _blanket_axis(cfg, survey):
    """Whether to reject ALL near-detector-axis trails (not just those adjacent
    to masked bleed/saturation). Config `vet.axis_blanket` wins when set; else
    the historical per-survey default (ZTF diffs are detector-artifact dominated;
    within-sequence CSS/generic diffs are not)."""
    b = cfg.get('axis_blanket') if cfg else None
    return (survey == 'ztf') if b is None else bool(b)


def prefit_pass(cand, cfg=None, survey='g96'):
    """Cheap cuts needing only detector-stage info; run before the expensive fit.

    cand: dict with keys snr (integrated MF SNR), pa_px_deg, near_bad_col (bool,
    optional; only used when the axis cut is conditional, not blanket).
    """
    c = {**DEFAULTS, **(cfg or {})}
    if cand['snr'] < c['mf_snr_min']:
        return False, "low_mf_snr"
    if _near_axis(cand['pa_px_deg'], c['axis_tol_deg']):
        if _blanket_axis(cfg, survey) or cand.get('near_bad_col', False):
            return False, "detector_axis"
    return True, "ok"


def passes_rb(fit, cand, imshape=None, cfg=None, survey='g96'):
    """fit: dict from trail_fit.fit_trail; cand: candidate dict (snr, pa_px_deg...)."""
    c = {**DEFAULTS, **(cfg or {})}
    if fit is None:
        return False, "fit_failed"
    if _near_axis(cand['pa_px_deg'], c['axis_tol_deg']):
        if _blanket_axis(cfg, survey) or cand.get('near_bad_col', False):
            return False, "detector_axis"
    if not (c['len_min_arcsec'] <= fit['trail_len'] <= c['len_max_arcsec']):
        return False, "length_oob"
    if not (c['chi2r_lo'] <= fit['chi2r'] <= c['chi2r_hi']):
        return False, "bad_chi2"
    if cand['snr'] < c['mf_snr_min']:
        return False, "low_mf_snr"
    if fit['sig_along'] > c['sig_along_max_arcsec']:
        return False, "poor_localization"
    if imshape is not None:
        th, h = fit['theta_px'], fit['h_px']
        ex1 = (fit['x'] - h * np.cos(th), fit['y'] - h * np.sin(th))
        ex2 = (fit['x'] + h * np.cos(th), fit['y'] + h * np.sin(th))
        e = c['edge_px']
        near = [min(p[0], p[1], imshape[1] - p[0], imshape[0] - p[1]) < e for p in (ex1, ex2)]
        if all(near):
            return False, "edge_to_edge"
    return True, "ok"


def flank_ratio(img, fit, sigma, mask=None):
    """Dipole feature: mean flux in two corridors offset +-3 sigma cross-track,
    relative to the on-line corridor mean. Strongly negative -> subtraction
    residual (star dipole). Returns ratio (0 if undetermined)."""
    th, h = fit['theta_px'], max(fit['h_px'], 3.0)
    ct, st = np.cos(th), np.sin(th)
    tt = np.arange(-int(h), int(h) + 1, dtype=float)
    hh, ww = img.shape

    def corridor_mean(off):
        xs = np.round(fit['x'] + tt * ct - off * st).astype(int)
        ys = np.round(fit['y'] + tt * st + off * ct).astype(int)
        ok = (xs >= 0) & (xs < ww) & (ys >= 0) & (ys < hh)
        if mask is not None:
            ok[ok] = mask[ys[ok], xs[ok]] == 0
        if ok.sum() < 3:
            return np.nan
        v = img[ys[ok], xs[ok]]
        v = v[np.isfinite(v)]
        return v.mean() if v.size else np.nan

    on = corridor_mean(0.0)
    f1 = corridor_mean(+3.0 * sigma)
    f2 = corridor_mean(-3.0 * sigma)
    if not np.isfinite(on) or on <= 0:
        return 0.0
    fl = np.nanmean([f1, f2])
    return float(fl / on) if np.isfinite(fl) else 0.0


def repetition_filter(per_exp_fits, mjds, rad_arcsec=None, pa_deg=None, cfg=None):
    """Cross-exposure repetition test over a sequence.

    per_exp_fits: list (one per exposure) of lists of fit dicts (need ra, dec,
    trail_PA). Detections recurring at the same sky position+PA in ANOTHER
    exposure are static artifacts -> flagged.

    Returns a parallel structure of booleans: True = KEEP (not repeated).
    """
    c = {**DEFAULTS, **(cfg or {})}
    rad = (rad_arcsec or c['repeat_rad_arcsec']) / 3600.0
    patol = pa_deg or c['repeat_pa_deg']
    keep = [[True] * len(fits) for fits in per_exp_fits]
    for i, fits_i in enumerate(per_exp_fits):
        for a, fi in enumerate(fits_i):
            for j, fits_j in enumerate(per_exp_fits):
                if j == i:
                    continue
                for fj in fits_j:
                    dra = (fi['ra'] - fj['ra']) * np.cos(np.radians(fi['dec']))
                    ddec = fi['dec'] - fj['dec']
                    if np.hypot(dra, ddec) > rad:
                        continue
                    dpa = abs((fi['trail_PA'] - fj['trail_PA']) % 180.0)
                    dpa = min(dpa, 180.0 - dpa)
                    if dpa <= patol:
                        keep[i][a] = False
                        break
                if not keep[i][a]:
                    break
    return keep


def det_qual(mf_snr_val, chi2r):
    """Quality in (0,1] for the hldet det_qual column: SNR-driven, chi2-penalized."""
    q = min(mf_snr_val / 20.0, 1.0)
    if not (0.7 <= chi2r <= 1.5):
        q *= 0.5
    return max(q, 0.01)
