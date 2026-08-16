"""The interface every loaded file presents to the rest of the app.

One file is one :class:`ImageSource`, whatever its format. Everything above this
layer -- the store, the routes, the model runner -- works in terms of frames,
windows and 8-bit pixels, and never asks whether it is looking at a DICOM or a
PNG.
"""
from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np


class ImageSource(ABC):
    """One loaded file.

    A file is the unit the user navigates in the file list. Most files hold a
    single frame; multi-frame DICOM and multi-page TIFF hold several, and those
    stay inside one list entry with a frame slider.

    Attributes:
        kind: ``"dicom"`` or ``"raster"``; shown in the file list and the export.
        frames: Number of frames in this file, at least 1.
        meta: Non-identifying technical metadata (modality, geometry, UIDs,
            pixel spacing, default window), carried into the export.
    """

    kind: str = "image"      # "dicom" | "raster"
    frames: int = 1
    meta: dict

    @abstractmethod
    def frame_shape(self, frame: int) -> tuple[int, int]:
        """Get a frame's geometry, without decoding pixels where possible.

        Args:
            frame: Frame index.

        Returns:
            ``(rows, columns)``.
        """

    @abstractmethod
    def default_window(self, frame: int) -> tuple[float, float]:
        """Get the window the viewer opens with.

        Args:
            frame: Frame index.

        Returns:
            ``(center, width)`` in the image's own intensity units.
        """

    @abstractmethod
    def frame_uint8(self, frame: int, wc: float | None = None,
                    ww: float | None = None) -> np.ndarray:
        """Render a frame for display.

        Args:
            frame: Frame index.
            wc: Window centre. None uses the file's own default.
            ww: Window width. None uses the file's own default.

        Returns:
            8-bit HxW grayscale or HxWx3 RGB.
        """

    @property
    def windowing(self) -> bool:
        """Whether window/level controls mean anything for this file.

        Returns:
            False for images with no meaningful intensity range, such as 8-bit
            colour, where the UI hides the sliders.
        """
        return True

    def frame_rgb(self, frame: int, wc: float | None = None,
                  ww: float | None = None) -> np.ndarray:
        """Render a frame for the model.

        This is exactly the pixels the user is looking at, which is what makes a
        click mean the same thing to the user and to SAM.

        Args:
            frame: Frame index.
            wc: Window centre. None uses the file's own default.
            ww: Window width. None uses the file's own default.

        Returns:
            HxWx3 uint8, grayscale replicated across channels where needed.
        """
        g = self.frame_uint8(frame, wc, ww)
        return g if g.ndim == 3 else np.stack([g, g, g], axis=-1)

    def check_frame(self, frame: int) -> int:
        """Validate a frame index.

        Args:
            frame: Frame index to check.

        Returns:
            The same index, for use inline.

        Raises:
            IndexError: If the index is outside this file's frames.
        """
        if frame < 0 or frame >= self.frames:
            raise IndexError(f"frame {frame} out of range (0..{self.frames - 1})")
        return frame
