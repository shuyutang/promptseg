from __future__ import annotations
from typing import Literal, Optional

from pydantic import BaseModel, Field

from config import settings


class Point(BaseModel):
    x: int
    y: int
    label: Literal[1, 0, -1] = 1  # 1 = include, 0/-1 = exclude


class Window(BaseModel):
    """Display window. Also fixes what the model sees, so it is part of a
    prompt's reproducible state."""
    center: float
    width: float


class Prompts(BaseModel):
    points: list[Point] = Field(default_factory=list)
    boxes: list[list[int]] = Field(default_factory=list)  # [x1, y1, x2, y2]

    def is_empty(self) -> bool:
        return not self.points and not self.boxes


class PreviewRequest(BaseModel):
    """Transient segmentation -- what these prompts would produce, uncommitted."""
    image_id: str
    frame: int = 0
    prompts: Prompts = Field(default_factory=Prompts)
    window: Optional[Window] = None
    threshold: float = settings.default_threshold
    # Candidates are ranked by predicted IoU, so 0 is the best-scoring one.
    mask_index: int = 0


class AnnotationCreate(BaseModel):
    image_id: str
    frame: int = 0
    label: str
    prompts: Prompts
    window: Optional[Window] = None
    threshold: float = settings.default_threshold
    mask_index: int = 0


class AnnotationUpdate(BaseModel):
    """Unset fields are left alone. Anything that changes the mask re-runs the
    decoder against the cached image embedding."""
    label: Optional[str] = None
    prompts: Optional[Prompts] = None
    window: Optional[Window] = None
    threshold: Optional[float] = None
    mask_index: Optional[int] = None


class AnnotationOut(BaseModel):
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
    score: Optional[float]
    created_at: str
    updated_at: str
