from __future__ import annotations
import hashlib

# Saturated, mutually distinguishable hues that stay legible over grayscale
# medical images. Deliberately excludes near-black/near-white/low-chroma tones,
# which would vanish against the underlying scan.
PALETTE: list[str] = [
    "#E8453C",  # red
    "#3B82F6",  # blue
    "#22C55E",  # green
    "#F59E0B",  # amber
    "#A855F7",  # purple
    "#06B6D4",  # cyan
    "#EC4899",  # pink
    "#84CC16",  # lime
    "#F97316",  # orange
    "#14B8A6",  # teal
    "#6366F1",  # indigo
    "#EAB308",  # yellow
]


def canonical(label: str) -> str:
    """Labels are matched case- and whitespace-insensitively."""
    return " ".join(label.strip().split()).lower()


def default_color(label: str) -> str:
    """Deterministic color for a label, stable across sessions and machines."""
    h = hashlib.md5(canonical(label).encode("utf-8")).digest()
    return PALETTE[h[0] % len(PALETTE)]


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))


def _to_lab(color: str) -> tuple[float, float, float]:
    """sRGB -> CIE L*a*b* (D65), so colours can be compared the way eyes do."""
    def lin(v: float) -> float:
        v /= 255.0
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(v) for v in hex_to_rgb(color))
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 1.00000
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t + 16 / 116)

    fx, fy, fz = f(x), f(y), f(z)
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def color_distance(a: str, b: str) -> float:
    """CIE76 delta-E. Roughly: <25 reads as 'the same colour' at swatch size."""
    la, lb = _to_lab(a), _to_lab(b)
    return sum((x - y) ** 2 for x, y in zip(la, lb)) ** 0.5


# Two labels whose colours are closer than this are treated as colliding.
MIN_DISTANCE = 28.0


class LabelRegistry:
    """Per-image label -> color assignment.

    The color is derived from the label name so the same label looks the same
    everywhere by default. If two distinct labels in one image happen to hash to
    the same slot, the later one steps to the next free slot so the user can
    still tell them apart. The resolved mapping is written into the export, so a
    consumer never has to re-derive it.
    """

    def __init__(self) -> None:
        self._colors: dict[str, str] = {}

    def color_for(self, label: str) -> str:
        key = canonical(label)
        if key in self._colors:
            return self._colors[key]

        preferred = default_color(key)
        taken = list(self._colors.values())
        color = preferred

        def clashes(c: str) -> bool:
            return any(color_distance(c, t) < MIN_DISTANCE for t in taken)

        if clashes(color):
            start = PALETTE.index(preferred)
            rotated = [PALETTE[(start + i) % len(PALETTE)] for i in range(1, len(PALETTE))]
            # Prefer a clearly distinct slot; fall back to the most distant one
            # available once the palette is exhausted.
            free = [c for c in rotated if not clashes(c)]
            if free:
                color = free[0]
            elif taken:
                color = max(rotated, key=lambda c: min(color_distance(c, t) for t in taken))

        self._colors[key] = color
        return color

    def as_dict(self) -> dict[str, str]:
        return dict(self._colors)
