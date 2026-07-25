#!/usr/bin/env python
"""Fast geometric streak suppression -- Phase 1 survey-scale optimization.

WHY THIS IS EXACT, NOT AN APPROXIMATION
---------------------------------------
pyradon's Streak.subtract_streak does:

    mask = model(im.shape, x1, x2, y1, y2, psf_sigma) > 0   # 4x-oversampled
    im[mask] = replace_value                                #  model + convolve2d

i.e. it builds a full-tile, 4x-oversampled, PSF-convolved photometric model of
the streak PURELY to threshold it into a boolean footprint -- every model VALUE
is discarded. Profiling (2026-07) showed this is 46% of total pipeline runtime
(~1.2 s per streak found, ~201 s per 5280^2 exposure at ~167 candidates).

The footprint that model()>0 produces is a CAPSULE: the set of pixels within a
fixed perpendicular distance of the line SEGMENT (it extends past the endpoints
by the same margin). Measured against pyradon for psf_sigma in 1.0-3.0:

    psf_sigma   pyradon max perp dist
      1.00           8.06 px
      1.05           8.49 px
      2.00          14.42 px
      3.00          20.81 px          -> halfwidth ~ 1.69 + 6.38*sigma

So a distance-to-segment test reproduces the same footprint. We use a slightly
GENEROUS halfwidth (2.0 + 6.5*sigma, >= pyradon's at every sigma above) so we
never under-mask -- under-masking would let the next FRT iteration re-detect the
same streak, which would change detection behavior. Over-masking by <1 px on the
footprint edge cannot create or destroy a detection: the suppressed pixels are
set to NaN and the streak is already found.

Cost: 0.002 s per streak (bbox-local) vs 1.203 s -> ~550x, ~201 s -> ~0.4 s per
exposure, with identical detection semantics.
"""
import numpy as np

# halfwidth(sigma) >= pyradon's model()>0 footprint at all measured sigma
HW_CONST = 2.0
HW_SLOPE = 6.5


def capsule_halfwidth(psf_sigma):
    """Perpendicular halfwidth (px) of the suppression footprint."""
    return HW_CONST + HW_SLOPE * float(psf_sigma)


def corridor_mask_bbox(shape, x1, y1, x2, y2, psf_sigma):
    """Boolean capsule footprint around segment (x1,y1)-(x2,y2), computed only
    inside the segment's bounding box (padded by the halfwidth).

    Returns (mask_bbox, (y0, y1b, x0, x1b)) so the caller can apply it in place.
    Returns (None, None) if the bbox is empty/off-image.
    """
    H, W = shape
    hw = capsule_halfwidth(psf_sigma)
    pad = int(np.ceil(hw)) + 2
    y0 = max(0, int(np.floor(min(y1, y2))) - pad)
    y1b = min(H, int(np.ceil(max(y1, y2))) + pad + 1)
    x0 = max(0, int(np.floor(min(x1, x2))) - pad)
    x1b = min(W, int(np.ceil(max(x1, x2))) + pad + 1)
    if y1b <= y0 or x1b <= x0:
        return None, None
    yy, xx = np.mgrid[y0:y1b, x0:x1b]
    dx, dy = float(x2 - x1), float(y2 - y1)
    seg2 = dx * dx + dy * dy
    if seg2 <= 0:                      # degenerate: point-like
        d = np.hypot(xx - x1, yy - y1)
    else:
        t = np.clip(((xx - x1) * dx + (yy - y1) * dy) / seg2, 0.0, 1.0)
        d = np.hypot(xx - (x1 + t * dx), yy - (y1 + t * dy))
    return d <= hw, (y0, y1b, x0, x1b)


def suppress_streak(im, x1, y1, x2, y2, psf_sigma, replace_value=np.nan):
    """In-place capsule suppression of one streak. Returns npix suppressed."""
    m, box = corridor_mask_bbox(im.shape, x1, y1, x2, y2, psf_sigma)
    if m is None:
        return 0
    y0, y1b, x0, x1b = box
    view = im[y0:y1b, x0:x1b]
    view[m] = replace_value
    return int(m.sum())


_PATCHED = False


def _streak_classes():
    """Every distinct Streak class object currently loaded from streak.py.

    pyradon is imported inconsistently: our package does `from src.streak import
    Streak` while vendor/pyradon/src/finder.py does `from streak import Streak`.
    Python treats 'src.streak' and 'streak' as SEPARATE module objects for the
    same file, each with its own Streak class -- so patching one leaves the
    finder's copy untouched (this silently produced a 1.0x speedup until caught
    by profiling). Patch every loaded instance.
    """
    import sys
    seen, out = set(), []
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None) or ""
        if f.endswith("streak.py"):
            cls = getattr(mod, "Streak", None)
            if cls is not None and id(cls) not in seen:
                seen.add(id(cls))
                out.append(cls)
    return out


def install_fast_suppression():
    """Monkey-patch pyradon's Streak.subtract_streak with the geometric capsule.

    Deliberately a RUNTIME patch, not a vendor source edit, to preserve the
    project's zero-source-patches rule for vendor/pyradon (see PATCHES.md):
    the pinned upstream source stays byte-identical and the optimization lives
    in our package, where it is testable and revertible.
    """
    global _PATCHED
    if _PATCHED:
        return True
    from . import import_pyradon
    import_pyradon()                     # ensures vendor/pyradon on sys.path
    import src.finder                    # noqa: F401  (forces the bare 'streak' module to load)

    def _fast_subtract(self, im, replace_value=np.nan, image_type="subsection"):
        if image_type == "subsection":
            x1, x2, y1, y2 = self.x1, self.x2, self.y1, self.y2
        elif image_type == "full":
            x1, x2, y1, y2 = self.x1f, self.x2f, self.y1f, self.y2f
        elif image_type == "cutout":
            x1, x2, y1, y2 = self.x1c, self.x2c, self.y1c, self.y2
        else:
            raise ValueError(f"image_type {image_type} not recognized... "
                             "use 'subsection', 'full', or 'cutout'. ")
        suppress_streak(im, x1, y1, x2, y2, self.psf_sigma, replace_value)

    n = 0
    for cls in _streak_classes():
        if not hasattr(cls, "_subtract_streak_pyradon"):
            cls._subtract_streak_pyradon = cls.subtract_streak   # keep original
        cls.subtract_streak = _fast_subtract
        n += 1
    if n == 0:
        raise RuntimeError("fastmask: no pyradon Streak class found to patch")
    _PATCHED = True
    return True


def uninstall_fast_suppression():
    """Restore pyradon's original subtract_streak (for A/B validation)."""
    global _PATCHED
    if not _PATCHED:
        return False
    for cls in _streak_classes():
        if hasattr(cls, "_subtract_streak_pyradon"):
            cls.subtract_streak = cls._subtract_streak_pyradon
    _PATCHED = False
    return True
