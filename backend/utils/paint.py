from __future__ import annotations

import numpy as np

MAX_RADIUS = 256


def _disc(radius: int) -> np.ndarray:
    r = int(radius)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def _stamp(mask: np.ndarray, cx: int, cy: int, disc: np.ndarray, value: bool) -> None:
    r = disc.shape[0] // 2
    h, w = mask.shape
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    if y0 >= y1 or x0 >= x1:
        return
    sub = disc[y0 - (cy - r):y1 - (cy - r), x0 - (cx - r):x1 - (cx - r)]
    region = mask[y0:y1, x0:x1]
    if value:
        region |= sub
    else:
        region &= ~sub


def _walk(x0: int, y0: int, x1: int, y1: int, step: float) -> list[tuple[int, int]]:
    """Points along a segment, spaced closely enough that discs overlap.

    The browser only reports pointer positions every few pixels, so a fast drag
    would otherwise paint a dotted line instead of a stroke.
    """
    dx, dy = x1 - x0, y1 - y0
    dist = float(np.hypot(dx, dy))
    n = max(1, int(dist / max(step, 0.5)))
    return [(int(round(x0 + dx * i / n)), int(round(y0 + dy * i / n))) for i in range(n + 1)]


def apply_strokes(mask: np.ndarray, strokes: list[dict]) -> np.ndarray:
    """Paint brush strokes onto a model mask, in order.

    Strokes are absolute image coordinates, so the final mask is always
    reproducible as model(prompts) -> strokes. That is what makes a hand
    correction survive a later change to the prompts.
    """
    out = np.asarray(mask).astype(bool).copy()
    if not strokes:
        return out

    for s in strokes:
        pts = s.get("points") or []
        if not pts:
            continue
        radius = int(max(1, min(int(s.get("radius", 6)), MAX_RADIUS)))
        add = str(s.get("mode", "add")) != "erase"
        disc = _disc(radius)
        step = max(1.0, radius * 0.5)

        prev = None
        for p in pts:
            x, y = int(p[0]), int(p[1])
            if prev is None:
                _stamp(out, x, y, disc, add)
            else:
                for px, py in _walk(prev[0], prev[1], x, y, step):
                    _stamp(out, px, py, disc, add)
            prev = (x, y)

    return out
