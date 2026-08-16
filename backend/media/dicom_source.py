"""DICOM files as :class:`~media.base.ImageSource`.

A thin adapter: the pixel work -- decoding, modality and VOI LUTs, palette
colour, MONOCHROME1 inversion -- lives in :mod:`dicom.io`, and this class only
binds it to the interface the rest of the app uses.
"""
from __future__ import annotations

import numpy as np

from dicom.io import DicomImage, default_window, frame_shape, frame_uint8, load_single
from media.base import ImageSource


class DicomSource(ImageSource):
    """A single DICOM file, possibly multi-frame."""

    kind = "dicom"

    def __init__(self, image: DicomImage) -> None:
        """Wrap an already-parsed DICOM.

        Args:
            image: Parsed file from :func:`dicom.io.load_single`.
        """
        self._img = image
        self.frames = image.frames
        self.meta = dict(image.meta)

    @classmethod
    def from_bytes(cls, data: bytes) -> "DicomSource":
        """Parse a DICOM file.

        Args:
            data: Raw file bytes.

        Returns:
            The wrapped source.

        Raises:
            ValueError: If the file carries no pixel data.
        """
        return cls(load_single(data))

    def frame_shape(self, frame: int) -> tuple[int, int]:
        """Get ``(rows, columns)`` for a frame, from the header.

        Args:
            frame: Frame index.

        Returns:
            ``(rows, columns)``.

        Raises:
            IndexError: If the frame index is out of range.
        """
        return frame_shape(self._img, self.check_frame(frame))

    def default_window(self, frame: int) -> tuple[float, float]:
        """Get the file's own window, or a percentile fallback.

        Args:
            frame: Frame index.

        Returns:
            ``(center, width)``.

        Raises:
            IndexError: If the frame index is out of range.
        """
        return default_window(self._img, self.check_frame(frame))

    def frame_uint8(self, frame: int, wc: float | None = None,
                    ww: float | None = None) -> np.ndarray:
        """Render a frame for display.

        Args:
            frame: Frame index.
            wc: Window centre; None applies the file's VOI LUT instead.
            ww: Window width; None applies the file's VOI LUT instead.

        Returns:
            8-bit HxW grayscale or HxWx3 RGB, MONOCHROME1 already inverted.

        Raises:
            IndexError: If the frame index is out of range.
        """
        return frame_uint8(self._img, self.check_frame(frame), wc, ww)

    @property
    def windowing(self) -> bool:
        """Whether window/level applies.

        Returns:
            True for monochrome images. A colour ultrasound frame is already
            8-bit RGB; there is no window to turn.
        """
        return str(self.meta.get("photometric_interpretation", "")).upper().startswith("MONO")

    @property
    def dataset(self):
        """The first pydicom dataset, for callers that need raw tags.

        Returns:
            pydicom.Dataset: Used by the loader to read series UID and slice
            position for ordering.
        """
        return self._img.datasets[0]
