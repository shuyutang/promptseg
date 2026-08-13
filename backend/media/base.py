from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np


class ImageSource(ABC):
    """One loaded file.

    A file is the unit the user navigates in the file list. Most files hold a
    single frame; multi-frame DICOM and multi-page TIFF hold several, and those
    stay inside one list entry with a frame slider.
    """

    kind: str = "image"      # "dicom" | "raster"
    frames: int = 1
    meta: dict

    @abstractmethod
    def frame_shape(self, frame: int) -> tuple[int, int]:
        """(rows, columns) without decoding pixels where possible."""

    @abstractmethod
    def default_window(self, frame: int) -> tuple[float, float]:
        """(center, width) the viewer opens with."""

    @abstractmethod
    def frame_uint8(self, frame: int, wc: float | None = None,
                    ww: float | None = None) -> np.ndarray:
        """Displayable 8-bit frame: HxW grayscale or HxWx3 RGB."""

    @property
    def windowing(self) -> bool:
        """False for images with no meaningful intensity range (8-bit colour)."""
        return True

    def frame_rgb(self, frame: int, wc: float | None = None,
                  ww: float | None = None) -> np.ndarray:
        """HxWx3 uint8 for the model -- exactly the pixels the user is looking at."""
        g = self.frame_uint8(frame, wc, ww)
        return g if g.ndim == 3 else np.stack([g, g, g], axis=-1)

    def check_frame(self, frame: int) -> int:
        if frame < 0 or frame >= self.frames:
            raise IndexError(f"frame {frame} out of range (0..{self.frames - 1})")
        return frame
