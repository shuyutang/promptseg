from __future__ import annotations
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from config import settings
from dicom.io import DicomImage
from labels import LabelRegistry, canonical


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Annotation:
    id: str
    image_id: str
    frame: int
    label: str          # display form, as first typed by the user
    instance: int       # 1-based, unique per (image_id, label)
    prompts: dict       # {"points": [{x,y,label}], "boxes": [[x1,y1,x2,y2]]}
    window: dict | None  # {"center": c, "width": w} the mask was produced under
    threshold: float
    mask_index: int
    rle: dict
    area: int
    bbox: list[int] | None
    score: float | None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)


@dataclass
class ImageRecord:
    image_id: str
    image: DicomImage
    filename: str
    labels: LabelRegistry = field(default_factory=LabelRegistry)
    annotations: "OrderedDict[str, Annotation]" = field(default_factory=OrderedDict)
    # canonical label -> highest instance number handed out so far. Never
    # decremented, so instance numbers stay stable when an annotation is deleted.
    _instance_counter: dict[str, int] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)

    def next_instance(self, label: str) -> int:
        key = canonical(label)
        n = self._instance_counter.get(key, 0) + 1
        self._instance_counter[key] = n
        return n

    def display_label(self, label: str) -> str:
        """Reuse the casing of the first occurrence so 'Spine' and 'spine' merge."""
        key = canonical(label)
        for a in self.annotations.values():
            if canonical(a.label) == key:
                return a.label
        return label.strip()

    def label_summary(self) -> list[dict]:
        out: dict[str, dict] = {}
        for a in self.annotations.values():
            key = canonical(a.label)
            entry = out.setdefault(key, {"name": a.label, "color": self.labels.color_for(key), "count": 0})
            entry["count"] += 1
        return sorted(out.values(), key=lambda e: e["name"].lower())


class Store:
    """In-memory session state. Bounded so a long-lived server cannot grow forever."""

    def __init__(self, max_images: int | None = None) -> None:
        self._images: OrderedDict[str, ImageRecord] = OrderedDict()
        self._max = max_images or settings.max_images
        self._lock = threading.RLock()
        self.on_evict = None  # set by the app to drop cached embeddings too

    def add(self, image: DicomImage, filename: str) -> ImageRecord:
        with self._lock:
            image_id = str(uuid.uuid4())
            rec = ImageRecord(image_id=image_id, image=image, filename=filename)
            self._images[image_id] = rec
            self._images.move_to_end(image_id)
            while len(self._images) > self._max:
                old_id, _ = self._images.popitem(last=False)
                if self.on_evict:
                    self.on_evict(old_id)
            return rec

    def get(self, image_id: str) -> ImageRecord:
        with self._lock:
            if image_id not in self._images:
                raise KeyError(image_id)
            self._images.move_to_end(image_id)
            return self._images[image_id]

    def find_annotation(self, ann_id: str) -> tuple[ImageRecord, Annotation]:
        with self._lock:
            for rec in self._images.values():
                ann = rec.annotations.get(ann_id)
                if ann is not None:
                    return rec, ann
            raise KeyError(ann_id)

    # ---- annotations -------------------------------------------------

    def add_annotation(self, rec: ImageRecord, *, frame: int, label: str, prompts: dict,
                       window: dict | None, threshold: float, mask_index: int,
                       mask: np.ndarray, score: float | None) -> Annotation:
        from utils.rle import mask_bbox, mask_to_rle

        with self._lock:
            display = rec.display_label(label)
            rec.labels.color_for(display)
            ann = Annotation(
                id=str(uuid.uuid4()),
                image_id=rec.image_id,
                frame=frame,
                label=display,
                instance=rec.next_instance(display),
                prompts=prompts,
                window=window,
                threshold=threshold,
                mask_index=mask_index,
                rle=mask_to_rle(mask),
                area=int(mask.sum()),
                bbox=mask_bbox(mask),
                score=score,
            )
            rec.annotations[ann.id] = ann
            return ann

    def update_annotation(self, rec: ImageRecord, ann: Annotation, *,
                          mask: np.ndarray | None = None, score: float | None = None,
                          **fields) -> Annotation:
        from utils.rle import mask_bbox, mask_to_rle

        with self._lock:
            new_label = fields.pop("label", None)
            if new_label is not None and canonical(new_label) != canonical(ann.label):
                display = rec.display_label(new_label) if any(
                    canonical(a.label) == canonical(new_label)
                    for a in rec.annotations.values() if a.id != ann.id
                ) else new_label.strip()
                ann.label = display
                # Moving to a different label means a new instance number there.
                ann.instance = rec.next_instance(display)
                rec.labels.color_for(display)

            for k, v in fields.items():
                if v is not None:
                    setattr(ann, k, v)

            if mask is not None:
                ann.rle = mask_to_rle(mask)
                ann.area = int(mask.sum())
                ann.bbox = mask_bbox(mask)
                ann.score = score

            ann.updated_at = _now()
            return ann

    def delete_annotation(self, rec: ImageRecord, ann_id: str) -> bool:
        with self._lock:
            return rec.annotations.pop(ann_id, None) is not None
