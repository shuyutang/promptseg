from __future__ import annotations
import numpy as np

# COCO-style uncompressed RLE: column-major (Fortran) run lengths, always
# starting with a run of zeros (possibly length 0).


def mask_to_rle(mask: np.ndarray) -> dict:
    """mask: HxW, 0/1. Returns {"counts": [...], "size": [h, w]}."""
    h, w = mask.shape
    flat = np.asarray(mask, dtype=np.uint8).ravel(order="F")
    if flat.size == 0:
        return {"counts": [], "size": [int(h), int(w)]}

    # Indices where the value changes, plus the implicit boundaries.
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate(([0], change, [flat.size]))
    runs = np.diff(bounds)

    counts = runs.tolist()
    if flat[0] == 1:
        # COCO requires the first run to be zeros.
        counts = [0] + counts
    return {"counts": [int(c) for c in counts], "size": [int(h), int(w)]}


def rle_to_mask(rle: dict) -> np.ndarray:
    """Inverse of mask_to_rle."""
    h, w = rle["size"]
    counts = list(rle["counts"])
    flat = np.zeros(h * w, dtype=np.uint8)
    pos, value = 0, 0
    for run in counts:
        if value:
            flat[pos : pos + run] = 1
        pos += run
        value ^= 1
    return flat.reshape((h, w), order="F")


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    """[x, y, w, h] of the foreground, or None if the mask is empty."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
