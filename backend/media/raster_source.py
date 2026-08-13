from __future__ import annotations
import io

import numpy as np
from PIL import Image, ImageSequence

from media.base import ImageSource

# Ordinary pictures, so a user can label a screenshot or an exported slice
# without converting it to DICOM first.
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}

# 16-bit PNG/TIFF still carries a real intensity range worth windowing.
_DEEP_MODES = {"I", "I;16", "I;16B", "I;16L", "F"}


class RasterSource(ImageSource):
    """PNG / JPEG / TIFF / BMP. Multi-page TIFF and GIF become multiple frames."""

    kind = "raster"

    def __init__(self, data: bytes, filename: str = "") -> None:
        im = Image.open(io.BytesIO(data))
        self._pages: list[np.ndarray] = []
        for page in ImageSequence.Iterator(im):
            self._pages.append(self._page_array(page))
        if not self._pages:
            raise ValueError("Image contains no frames.")

        self.frames = len(self._pages)
        first = self._pages[0]
        h, w = first.shape[:2]
        self._deep = first.dtype != np.uint8
        self.meta = {
            "modality": "",
            "rows": int(h),
            "columns": int(w),
            "frames": self.frames,
            "photometric_interpretation": "RGB" if first.ndim == 3 else "MONOCHROME2",
            "sop_class_uid": "",
            "series_instance_uid": "",
            "study_instance_uid": "",
            "series_description": "",
            "pixel_spacing": [],
            "window_center": None,
            "window_width": None,
            "source_format": (im.format or "").upper(),
            "source_mode": im.mode,
        }

    @staticmethod
    def _page_array(page: Image.Image) -> np.ndarray:
        if page.mode == "P":
            page = page.convert("RGB")
        elif page.mode == "RGBA":
            page = page.convert("RGB")
        elif page.mode == "LA":
            page = page.convert("L")
        elif page.mode in ("1",):
            page = page.convert("L")
        arr = np.asarray(page)
        if arr.ndim == 3 and arr.shape[-1] > 3:
            arr = arr[..., :3]
        if page.mode in _DEEP_MODES:
            return arr.astype(np.float32)
        return np.ascontiguousarray(arr)

    # ---- ImageSource -------------------------------------------------

    def frame_shape(self, frame: int) -> tuple[int, int]:
        a = self._pages[self.check_frame(frame)]
        return int(a.shape[0]), int(a.shape[1])

    def default_window(self, frame: int) -> tuple[float, float]:
        a = self._pages[self.check_frame(frame)]
        if a.dtype == np.uint8:
            return 127.5, 255.0
        lo, hi = np.percentile(a, (1, 99))
        if hi <= lo:
            lo, hi = float(a.min()), float(a.max()) or 1.0
        return float((lo + hi) / 2.0), float(max(hi - lo, 1e-6))

    def frame_uint8(self, frame: int, wc: float | None = None,
                    ww: float | None = None) -> np.ndarray:
        a = self._pages[self.check_frame(frame)]
        if a.dtype == np.uint8:
            return a  # already displayable; windowing an 8-bit picture only hurts
        if wc is None or ww is None:
            wc, ww = self.default_window(frame)
        ww = max(float(ww), 1e-6)
        lo = float(wc) - ww / 2.0
        return np.clip((a - lo) / ww * 255.0 + 0.5, 0, 255).astype(np.uint8)

    @property
    def windowing(self) -> bool:
        return self._deep
