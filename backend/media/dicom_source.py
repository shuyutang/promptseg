from __future__ import annotations

import numpy as np

from dicom.io import DicomImage, default_window, frame_shape, frame_uint8, load_single
from media.base import ImageSource


class DicomSource(ImageSource):
    """A single DICOM file, possibly multi-frame."""

    kind = "dicom"

    def __init__(self, image: DicomImage) -> None:
        self._img = image
        self.frames = image.frames
        self.meta = dict(image.meta)

    @classmethod
    def from_bytes(cls, data: bytes) -> "DicomSource":
        return cls(load_single(data))

    def frame_shape(self, frame: int) -> tuple[int, int]:
        return frame_shape(self._img, self.check_frame(frame))

    def default_window(self, frame: int) -> tuple[float, float]:
        return default_window(self._img, self.check_frame(frame))

    def frame_uint8(self, frame: int, wc: float | None = None,
                    ww: float | None = None) -> np.ndarray:
        return frame_uint8(self._img, self.check_frame(frame), wc, ww)

    @property
    def windowing(self) -> bool:
        # A colour ultrasound frame is already 8-bit RGB; there is no window to turn.
        return str(self.meta.get("photometric_interpretation", "")).upper().startswith("MONO")

    @property
    def dataset(self):
        return self._img.datasets[0]
