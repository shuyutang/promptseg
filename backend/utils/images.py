from __future__ import annotations
import io

import numpy as np
from PIL import Image


def _png(arr: np.ndarray, mode: str) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    return buf.getvalue()


def array_to_png(arr: np.ndarray) -> bytes:
    """8-bit HxW grayscale or HxWx3 RGB -> PNG."""
    if arr.ndim == 2:
        return _png(arr.astype(np.uint8), "L")
    return _png(arr.astype(np.uint8), "RGB")


def mask_to_png(mask01: np.ndarray) -> bytes:
    """Binary mask -> 8-bit grayscale PNG (0 or 255)."""
    m = np.asarray(mask01)
    if m.dtype != np.uint8:
        m = (np.clip(m, 0, 1) * 255).astype(np.uint8)
    else:
        m = (m > 0).astype(np.uint8) * 255
    return _png(m, "L")


def _outline(mask: np.ndarray, width: int = 1) -> np.ndarray:
    """Boundary pixels, via erosion by 4-connected shifts. Avoids a scipy dep."""
    m = mask.astype(bool)
    eroded = m.copy()
    for _ in range(max(1, width)):
        e = eroded.copy()
        e[1:, :] &= eroded[:-1, :]
        e[:-1, :] &= eroded[1:, :]
        e[:, 1:] &= eroded[:, :-1]
        e[:, :-1] &= eroded[:, 1:]
        eroded = e
    return m & ~eroded


def overlay_png(shape: tuple[int, int], items: list[dict], alpha: int = 110) -> bytes:
    """Composite many labeled masks into one RGBA layer.

    items: [{"mask": HxW bool/uint8, "color": (r,g,b), "selected": bool}]

    Later items paint over earlier ones. Every mask also gets a fully opaque
    outline, which is what keeps two adjacent instances of the same label -- and
    therefore the same colour -- visually separable.
    """
    h, w = shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    for item in items:
        m = np.asarray(item["mask"]).astype(bool)
        if m.shape != (h, w) or not m.any():
            continue
        r, g, b = item["color"]
        a = int(item.get("alpha", alpha))

        rgba[m] = (r, g, b, a)
        edge = _outline(m, 2 if item.get("selected") else 1)
        rgba[edge] = (255, 255, 255, 255) if item.get("selected") else (r, g, b, 255)

    return _png(rgba, "RGBA")


def colored_overlay_png(mask01: np.ndarray, color=(232, 69, 60), alpha: int = 110) -> bytes:
    """Single-mask RGBA overlay -- used for the live preview before committing."""
    m = np.asarray(mask01).astype(bool)
    return overlay_png(m.shape, [{"mask": m, "color": tuple(color), "alpha": alpha}], alpha)
