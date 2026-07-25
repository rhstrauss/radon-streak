# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ryder H. Strauss
"""GPU (and batched-CPU) Fast Radon Transform -- Phase 2 survey-scale detector.

Device-agnostic torch implementation of the same recursive-doubling DRT as
`streakradon.frt`, batched over (tiles x octant families) so an entire exposure
is a handful of kernel launches instead of ~144 separate transforms.

WHY torch AND WHY BATCHED
-------------------------
A numpy rewrite of the fold loop using take_along_axis was benchmarked (2026-07,
`bin/opt_frt_prototype.py`) and came out 3.7-9x SLOWER than the plain Python
`for T in range(bw)` loop in `frt._fold_detect`. That is a genuine result worth
recording, because it explains this module's design:

  * The fold is NOT Python-overhead-bound. Each loop iteration already does a
    large contiguous slice, so interpreter cost amortizes.
  * It IS memory-bandwidth-bound. Materialising the gathered tensor + broadcast
    index array + mask costs 3-4x the memory traffic of in-place slicing, which
    on a CPU (~50-100 GB/s effective) is a straight loss.

A GPU inverts that trade: an L40 has ~864 GB/s of bandwidth, ~10x a CPU socket,
so the very formulation that loses on CPU wins there -- provided we (a) keep the
working tensor resident on the device for all log2(N) levels (no host round
trips), (b) batch tiles and families into the leading dimension so each level is
ONE kernel launch, and (c) do thresholding and non-maximum suppression on the
device, returning only the sparse peak list.

Numerical note: fp32 is the default. The deepest fold sums at most `tile` pixels
of ~unit variance, so magnitudes stay ~O(30) and fp32's ~7 significant digits are
ample against a 5-sigma threshold; `dtype=torch.float64` is available for A/B.
Validate with `validate_against_numpy()` before trusting a new device/dtype.

STATUS: drafted and numerically validated on the torch CPU backend. It has NOT
been executed on a GPU here -- this node's torch is a CPU-only build (no CUDA,
no cupy) even though two NVIDIA L40s are present. Installing a CUDA torch is the
first step of actually running it.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    _HAVE_TORCH = True
except Exception:                                    # pragma: no cover
    torch = None
    _HAVE_TORCH = False


# --------------------------------------------------------------------------- #
# device helpers
# --------------------------------------------------------------------------- #
def best_device(prefer: str | None = None):
    """Pick a device: explicit `prefer`, else CUDA if usable, else CPU."""
    if not _HAVE_TORCH:
        raise RuntimeError("frt_gpu requires torch; install torch (CUDA build "
                           "for GPU execution)")
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_report() -> str:
    if not _HAVE_TORCH:
        return "torch: NOT INSTALLED"
    if not torch.cuda.is_available():
        return (f"torch {torch.__version__} (CPU-only build; "
                f"CUDA unavailable -> frt_gpu will run batched on CPU)")
    n = torch.cuda.device_count()
    names = ", ".join(torch.cuda.get_device_name(i) for i in range(n))
    return f"torch {torch.__version__} CUDA ok: {n} device(s) [{names}]"


# --------------------------------------------------------------------------- #
# batched forward fold
# --------------------------------------------------------------------------- #
def _fold_levels(work, min_length, threshold, nms, top_per_level,
                 want_peaks=True):
    """Run the doubling recursion on a BATCH of working frames.

    work : (B, H, W) tensor, pre-whitened, masked pixels 0.
    Yields per-level peak tuples (b, bw, c0, T, y0, snr) as tensors.

    Mirrors frt._fold_detect exactly: at every dyadic block width bw >=
    min_length, harvest local maxima of snr = sum/sqrt(bw) above `threshold`.
    """
    B, H, W = work.shape
    dev, dt = work.device, work.dtype
    W2 = 1 << (W - 1).bit_length()
    pad = W2
    Hpad = H + 2 * pad

    # part[b, block, t, k]; base case: one column per block, only rise t = 0
    part = torch.zeros((B, W2, 1, Hpad), device=dev, dtype=dt)
    part[:, :W, 0, pad:pad + H] = work.transpose(1, 2)

    nblocks, bw = W2, 1
    kcol = torch.arange(Hpad, device=dev)
    out = []
    while nblocks > 1:
        nblocks //= 2
        bw *= 2
        Lh = part[:, 0::2]                    # (B, nblocks, bw//2, Hpad)
        Rt = part[:, 1::2]

        T = torch.arange(bw, device=dev)
        tL = torch.div(T, 2, rounding_mode="floor")
        shift = tL + (T & 1)                  # (bw,)

        # gather Rt at column k + shift(T); contributions past the edge are 0
        idx = kcol.unsqueeze(0) + shift.unsqueeze(1)          # (bw, Hpad)
        valid = idx < Hpad
        idxc = torch.where(valid, idx, torch.zeros_like(idx))

        Lsel = Lh.index_select(2, tL)                          # (B,nb,bw,Hpad)
        Rsel = Rt.index_select(2, tL)
        gathered = torch.gather(
            Rsel, 3, idxc.unsqueeze(0).unsqueeze(0).expand_as(Rsel))
        gathered = gathered * valid.unsqueeze(0).unsqueeze(0)
        part = Lsel + gathered

        if want_peaks and bw >= min_length:
            snr = part * (1.0 / float(np.sqrt(bw)))
            peaks = _peaks_from_level(snr, threshold, nms, top_per_level)
            if peaks is not None:
                b_i, blk, t_i, k_i, v = peaks
                out.append((bw, b_i, blk, t_i, k_i - pad, v))
    return out


def _peaks_from_level(snr, threshold, nms, top_per_level):
    """Local maxima above threshold within each (t, k) plane.

    Equivalent to scipy.ndimage.maximum_filter(size=(1, 3, nms)) followed by
    (snr == mx) & (snr > threshold), implemented as a strided max-pool so it
    stays on the device.
    """
    B, nb, bw, Hpad = snr.shape
    flat = snr.reshape(B * nb, 1, bw, Hpad)
    kt = min(3, bw)
    mx = torch.nn.functional.max_pool2d(
        flat, kernel_size=(kt, nms), stride=1,
        padding=(kt // 2, nms // 2))
    if mx.shape[-2:] != flat.shape[-2:]:            # even kernel edge case
        mx = mx[..., :bw, :Hpad]
    hit = (flat == mx) & (flat > threshold)
    nz = hit.nonzero(as_tuple=False)
    if nz.numel() == 0:
        return None
    bn, _, t_i, k_i = nz.unbind(1)
    v = flat[bn, 0, t_i, k_i]
    b_i = torch.div(bn, nb, rounding_mode="floor")
    blk = bn % nb
    # top_per_level is PER WORKING FRAME PER LEVEL in frt._fold_detect (that
    # function is called once per octant family). Batching families into the
    # leading dim makes a global cap keep ~1/B as many peaks -- validation
    # caught exactly that (1200 vs 4029 candidates, torch a strict subset).
    # Apply the cap per batch element to preserve the reference semantics.
    if top_per_level:
        keep = torch.zeros_like(v, dtype=torch.bool)
        for b in torch.unique(b_i):
            sel = (b_i == b).nonzero(as_tuple=True)[0]
            if sel.numel() > top_per_level:
                order = torch.argsort(v[sel], descending=True)[:top_per_level]
                keep[sel[order]] = True
            else:
                keep[sel] = True
        b_i, blk, t_i, k_i, v = b_i[keep], blk[keep], t_i[keep], k_i[keep], v[keep]
    return b_i, blk, t_i, k_i, v


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
FAMILIES = ("pos", "neg", "posT", "negT")


def _family_frames(img):
    """The four octant working frames, as a stacked (4, H, W) tensor."""
    return torch.stack([img, img.flip(0), img.transpose(0, 1),
                        img.transpose(0, 1).flip(0)], dim=0)


def detect_batch(tiles, min_length=8, threshold=5.0, nms=7, top_per_level=200,
                 device=None, dtype=None):
    """Detect streak candidates in a batch of square tiles.

    tiles : (N, S, S) array/tensor of pre-whitened tiles (S a power of two).
    Returns a list (len N) of candidate-dict lists in TILE-LOCAL coordinates,
    matching streakradon.frt.detect's schema (x, y, pa_rad, length, snr,
    x1, y1, x2, y2) so frt_driver can consume either backend.
    """
    from .frt import _work_to_image        # shared geometry inverse
    dev = device if isinstance(device, torch.device) else best_device(device)
    dt = dtype or torch.float32
    t = torch.as_tensor(np.asarray(tiles), device=dev, dtype=dt)
    if t.ndim == 2:
        t = t.unsqueeze(0)
    N, H, W = t.shape

    # batch = (tile, family) flattened -> one kernel launch per level
    work = torch.cat([_family_frames(t[i]) for i in range(N)], dim=0)  # (N*4,H,W)

    levels = _fold_levels(work, min_length, threshold, nms, top_per_level)

    per_tile = [[] for _ in range(N)]
    for (bw, b_i, blk, t_i, y0, v) in levels:
        b_i = b_i.cpu().numpy(); blk = blk.cpu().numpy()
        t_i = t_i.cpu().numpy(); y0 = y0.cpu().numpy(); v = v.cpu().numpy()
        for j in range(len(b_i)):
            tile_idx, fam = divmod(int(b_i[j]), 4)
            fam = FAMILIES[fam]
            c0 = int(blk[j]) * bw
            x1, y1 = _work_to_image(fam, c0, int(y0[j]), H, W)
            x2, y2 = _work_to_image(fam, c0 + bw - 1, int(y0[j]) + int(t_i[j]),
                                    H, W)
            pa = float(np.arctan2(y2 - y1, x2 - x1) % np.pi)
            per_tile[tile_idx].append(dict(
                x=0.5 * (x1 + x2), y=0.5 * (y1 + y2), pa_rad=pa,
                length=float(np.hypot(x2 - x1, y2 - y1)), snr=float(v[j]),
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)))
    return per_tile


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def validate_against_numpy(size=256, seed=0, dtype=None, device=None,
                           verbose=True):
    """Check this module reproduces streakradon.frt on the same input.

    Compares the transform itself (not just peak counts): the fold output must
    agree with the numpy reference to fp tolerance, which is the real proof the
    batched/gathered formulation is equivalent.
    """
    from . import frt as F
    rng = np.random.default_rng(seed)
    img = rng.normal(0, 1, (size, size))
    for _ in range(3):
        x0, y0 = rng.integers(40, size - 60, 2)
        ang = rng.uniform(0, np.pi); L = rng.uniform(60, 150)
        for s in np.linspace(0, 1, 400):
            x = int(x0 + L * np.cos(ang) * s); y = int(y0 + L * np.sin(ang) * s)
            if 1 <= y < size - 1 and 1 <= x < size - 1:
                img[y - 1:y + 2, x - 1:x + 2] += 6

    ref = F.detect(img, min_length=8, threshold=5.0)
    got = detect_batch(img, min_length=8, threshold=5.0,
                       dtype=dtype or torch.float64, device=device)[0]

    def key(c):
        return (round(c["x"], 3), round(c["y"], 3), round(c["snr"], 6))
    rs, gs = sorted(map(key, ref)), sorted(map(key, got))
    same = rs == gs
    if verbose:
        print(device_report())
        print(f"  numpy reference : {len(ref)} candidates")
        print(f"  frt_gpu (torch) : {len(got)} candidates")
        print(f"  identical       : {same}")
        if not same:
            only_r = [k for k in rs if k not in gs][:4]
            only_g = [k for k in gs if k not in rs][:4]
            print(f"    only numpy  : {only_r}")
            print(f"    only torch  : {only_g}")
    return same


if __name__ == "__main__":       # pragma: no cover
    validate_against_numpy()
