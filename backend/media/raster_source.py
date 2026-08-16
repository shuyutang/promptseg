"""Ordinary pictures as :class:`~media.base.ImageSource`.

PNG, JPEG, TIFF, BMP, WEBP and GIF, so a user can label a screenshot or an
exported slice without converting it to DICOM first. 8-bit images are displayed
as they are; 16-bit and float ones still carry a real intensity range, so they
keep working window/level controls.
"""
from __future__ import annotations
import io

import numpy as np
from PIL import Image, ImageSequence

from media.base import ImageSource

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
"""Extensions routed to this reader before DICOM is considered."""

_DEEP_MODES = {"I", "I;16", "I;16B", "I;16L", "F"}
"""Pillow modes whose pixels are worth windowing rather than showing raw."""


class RasterSource(ImageSource):
    """PNG / JPEG / TIFF / BMP. Multi-page TIFF and GIF become multiple frames."""

    kind = "raster"

    def __init__(self, data: bytes, filename: str = "") -> None:
        """Decode every page of a picture file.

        Args:
            data: Raw file bytes.
            filename: Original name, kept for error messages only; the format is
                decided by content.

        Raises:
            ValueError: If the file decodes to no frames.
        """
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
        """Convert one Pillow page to an array the rest of the class can use.

        Palette, alpha and bilevel modes are flattened to RGB or grayscale;
        deep modes are promoted to float32 so windowing has room to work.

        Args:
            page: One frame from the file.

        Returns:
            HxW or HxWx3 array, uint8 for ordinary pictures and float32 for
            16-bit or float ones.
        """
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
        """Get ``(rows, columns)`` for a page.

        Args:
            frame: Page index.

        Returns:
            ``(rows, columns)``.

        Raises:
            IndexError: If the page index is out of range.
        """
        a = self._pages[self.check_frame(frame)]
        return int(a.shape[0]), int(a.shape[1])

    def default_window(self, frame: int) -> tuple[float, float]:
        """Get the window the viewer opens with.

        Args:
            frame: Page index.

        Returns:
            ``(center, width)``: the full 8-bit range for ordinary pictures, or
            the 1st-99th percentile span for deep ones.

        Raises:
            IndexError: If the page index is out of range.
        """
        a = self._pages[self.check_frame(frame)]
        if a.dtype == np.uint8:
            return 127.5, 255.0
        lo, hi = np.percentile(a, (1, 99))
        if hi <= lo:
            lo, hi = float(a.min()), float(a.max()) or 1.0
        return float((lo + hi) / 2.0), float(max(hi - lo, 1e-6))

    def frame_uint8(self, frame: int, wc: float | None = None,
                    ww: float | None = None) -> np.ndarray:
        """Render a page for display.

        Args:
            frame: Page index.
            wc: Window centre; ignored for 8-bit pictures.
            ww: Window width; ignored for 8-bit pictures.

        Returns:
            8-bit HxW grayscale or HxWx3 RGB.

        Raises:
            IndexError: If the page index is out of range.
        """
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
        """Whether window/level applies.

        Returns:
            True only for 16-bit or float files, which still have an intensity
            range to explore.
        """
        return self._deep
