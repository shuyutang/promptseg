"""Pydantic request and response models for the HTTP API.

These types are also the reproducibility contract: an annotation carries the
prompts, brush strokes, display window, threshold and candidate index it was
produced under, so it can be re-derived rather than merely read back.
"""
from __future__ import annotations
from typing import Literal, Optional

from pydantic import BaseModel, Field

from config import settings


class Point(BaseModel):
    """One click, in native image pixels.

    Attributes:
        x: Column, 0-based.
        y: Row, 0-based.
        label: 1 to include this region, 0 or -1 to exclude it.
    """
    x: int
    y: int
    label: Literal[1, 0, -1] = 1  # 1 = include, 0/-1 = exclude


class Window(BaseModel):
    """Display window (level and width).

    Also fixes what the model sees, so it is part of a prompt's reproducible
    state and part of the embedding cache key.

    Attributes:
        center: Window centre, in the image's own intensity units.
        width: Window width, in the same units.
    """
    center: float
    width: float


class Stroke(BaseModel):
    """One brush drag, in native image pixels.

    Strokes are stored alongside the prompts rather than baked into the mask, so
    the final mask is always ``model(prompts)`` with these replayed on top.

    Attributes:
        mode: ``"add"`` to paint pixels in, ``"erase"`` to take them out.
        radius: Brush radius in pixels, 1-256.
        points: ``[[x, y], ...]`` pointer positions along the drag.
    """
    mode: Literal["add", "erase"] = "add"
    radius: int = Field(default=6, ge=1, le=256)
    points: list[list[int]] = Field(default_factory=list)  # [[x, y], ...]


class Prompts(BaseModel):
    """What the user pointed at.

    Attributes:
        points: Include/exclude clicks.
        boxes: ``[[x1, y1, x2, y2], ...]`` drawn boxes.
    """
    points: list[Point] = Field(default_factory=list)
    boxes: list[list[int]] = Field(default_factory=list)  # [x1, y1, x2, y2]

    def is_empty(self) -> bool:
        """Report whether there is anything for the model to act on.

        Returns:
            True when neither points nor boxes were given, in which case a mask
            can still be produced by hand with brush strokes alone.
        """
        return not self.points and not self.boxes


class PreviewRequest(BaseModel):
    """Transient segmentation -- what these prompts would produce, uncommitted.

    Attributes:
        image_id: Image to segment.
        frame: Frame index within that image.
        prompts: Points and boxes placed so far.
        window: Display window; the image's default is used when omitted.
        threshold: Probability above which a pixel is foreground.
        mask_index: Which candidate to take. Candidates are ranked by predicted
            IoU, so 0 is the best-scoring one.
        strokes: Brush strokes to replay on top of the model output.
    """
    image_id: str
    frame: int = 0
    prompts: Prompts = Field(default_factory=Prompts)
    window: Optional[Window] = None
    threshold: float = settings.default_threshold
    mask_index: int = 0
    strokes: list[Stroke] = Field(default_factory=list)


class AnnotationCreate(BaseModel):
    """Commit a mask as a labelled instance.

    Attributes:
        image_id: Image the mask belongs to.
        frame: Frame index within that image.
        label: Free text. Reusing one adds another instance in the same colour;
            matching is case- and whitespace-insensitive.
        prompts: Points and boxes that produced the mask.
        window: Display window the mask was drawn under.
        threshold: Probability above which a pixel is foreground.
        mask_index: Which ranked candidate was taken.
        strokes: Brush strokes replayed on top of the model output.
    """
    image_id: str
    frame: int = 0
    label: str
    prompts: Prompts
    window: Optional[Window] = None
    threshold: float = settings.default_threshold
    mask_index: int = 0
    strokes: list[Stroke] = Field(default_factory=list)


class AnnotationUpdate(BaseModel):
    """Edit an existing annotation. Unset fields are left alone.

    Anything that changes the mask re-runs the decoder against the cached image
    embedding.

    Attributes:
        label: New label. Moving to a different label takes a new instance
            number under it.
        prompts: Replacement points and boxes.
        window: Replacement display window.
        threshold: Replacement foreground threshold.
        mask_index: Replacement candidate index.
        strokes: Replaces the stroke list wholesale -- that is how undo is
            expressed.
    """
    label: Optional[str] = None
    prompts: Optional[Prompts] = None
    window: Optional[Window] = None
    threshold: Optional[float] = None
    mask_index: Optional[int] = None
    strokes: Optional[list[Stroke]] = None


class AnnotationOut(BaseModel):
    """An annotation as the API returns it.

    Attributes:
        id: Annotation identifier.
        image_id: Image it belongs to.
        frame: Frame index within that image.
        label: Display form of the label, as first typed by the user.
        instance: 1-based instance number within this image and label.
        color: Hex colour for the label, resolved on the workspace so it is the
            same in every file of the folder.
        area: Foreground pixel count.
        bbox: ``[x, y, width, height]``, or None for an empty mask.
        prompts: Points and boxes that produced the mask.
        window: Display window the mask was produced under.
        threshold: Foreground threshold used.
        mask_index: Which ranked candidate was taken.
        strokes: Brush strokes replayed on top of the model output.
        score: Model's predicted IoU for the chosen candidate, or None for a
            purely hand-drawn mask.
        created_at: UTC ISO-8601 timestamp, seconds resolution.
        updated_at: UTC ISO-8601 timestamp of the last edit.
    """
    id: str
    image_id: str
    frame: int
    label: str
    instance: int
    color: str
    area: int
    bbox: Optional[list[int]]
    prompts: Prompts
    window: Optional[Window]
    threshold: float
    mask_index: int
    strokes: list[Stroke] = Field(default_factory=list)
    score: Optional[float]
    created_at: str
    updated_at: str


class WorkspaceCreate(BaseModel):
    """Create an empty workspace.

    Attributes:
        name: Display name, usually the picked folder's name.
    """
    name: str = ""
