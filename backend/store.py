from __future__ import annotations
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from config import settings
from labels import LabelRegistry, canonical
from media.base import ImageSource


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
    # Hand corrections, replayed on top of the model mask in order. Keeping them
    # separate from the prompts is what lets a brush fix survive re-prompting.
    strokes: list[dict] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)


@dataclass
class ImageRecord:
    image_id: str
    source: ImageSource
    filename: str
    workspace_id: str = ""
    index: int = 0                      # position in the workspace file list
    labels: LabelRegistry = field(default_factory=LabelRegistry)
    annotations: "OrderedDict[str, Annotation]" = field(default_factory=OrderedDict)
    # canonical label -> highest instance number handed out so far. Never
    # decremented, so instance numbers stay stable when an annotation is deleted.
    _instance_counter: dict[str, int] = field(default_factory=dict)
    reviewed: bool = False
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
        return _summarise([self])

    def to_listing(self) -> dict:
        rows, cols = self.source.frame_shape(0)
        return {
            "image_id": self.image_id,
            "filename": self.filename,
            "index": self.index,
            "kind": self.source.kind,
            "frames": self.source.frames,
            "rows": rows,
            "columns": cols,
            "modality": self.source.meta.get("modality", ""),
            "windowing": self.source.windowing,
            "annotation_count": len(self.annotations),
            "labels": sorted({a.label for a in self.annotations.values()}, key=str.lower),
            "reviewed": self.reviewed,
        }


def _summarise(records: list[ImageRecord]) -> list[dict]:
    out: dict[str, dict] = {}
    for rec in records:
        for a in rec.annotations.values():
            key = canonical(a.label)
            entry = out.setdefault(key, {"name": a.label, "color": rec.labels.color_for(key), "count": 0})
            entry["count"] += 1
    return sorted(out.values(), key=lambda e: e["name"].lower())


@dataclass
class Workspace:
    """One folder-load: many files, one shared label vocabulary.

    Colours live here rather than on the image so that 'liver' is the same
    colour in every file of a folder -- the whole point of labelling a batch.
    """
    workspace_id: str
    name: str = ""
    labels: LabelRegistry = field(default_factory=LabelRegistry)
    images: "OrderedDict[str, ImageRecord]" = field(default_factory=OrderedDict)
    created_at: str = field(default_factory=_now)

    @property
    def records(self) -> list[ImageRecord]:
        return list(self.images.values())

    def label_summary(self) -> list[dict]:
        return _summarise(self.records)

    def to_listing(self) -> dict:
        return {
            "workspace_id": self.workspace_id,
            "name": self.name,
            "created_at": self.created_at,
            "image_count": len(self.images),
            "annotation_count": sum(len(r.annotations) for r in self.images.values()),
            "labels": self.label_summary(),
            "images": [r.to_listing() for r in self.images.values()],
        }


class Store:
    """In-memory session state. Bounded so a long-lived server cannot grow forever."""

    def __init__(self, max_workspaces: int | None = None) -> None:
        self._workspaces: OrderedDict[str, Workspace] = OrderedDict()
        self._index: dict[str, ImageRecord] = {}     # image_id -> record
        self._max = max_workspaces or settings.max_workspaces
        self._lock = threading.RLock()
        self.on_evict = None  # set by the app to drop cached embeddings too

    # ---- workspaces ---------------------------------------------------

    def create_workspace(self, name: str = "") -> Workspace:
        with self._lock:
            ws = Workspace(workspace_id=str(uuid.uuid4()), name=name)
            self._workspaces[ws.workspace_id] = ws
            self._evict_locked()
            return ws

    def get_workspace(self, workspace_id: str) -> Workspace:
        with self._lock:
            if workspace_id not in self._workspaces:
                raise KeyError(workspace_id)
            self._workspaces.move_to_end(workspace_id)
            return self._workspaces[workspace_id]

    def workspaces(self) -> list[Workspace]:
        with self._lock:
            return list(self._workspaces.values())

    def _evict_locked(self) -> None:
        while len(self._workspaces) > self._max:
            _, old = self._workspaces.popitem(last=False)
            for image_id in old.images:
                self._index.pop(image_id, None)
                if self.on_evict:
                    self.on_evict(image_id)

    # ---- images -------------------------------------------------------

    def add_image(self, ws: Workspace, source: ImageSource, filename: str) -> ImageRecord:
        with self._lock:
            if len(ws.images) >= settings.max_files:
                raise ValueError(
                    f"Workspace is limited to {settings.max_files} files "
                    f"(raise SAM2_MAX_FILES to load more)."
                )
            rec = ImageRecord(
                image_id=str(uuid.uuid4()), source=source, filename=filename,
                workspace_id=ws.workspace_id, index=len(ws.images), labels=ws.labels,
            )
            ws.images[rec.image_id] = rec
            self._index[rec.image_id] = rec
            self._workspaces.move_to_end(ws.workspace_id)
            return rec

    def get(self, image_id: str) -> ImageRecord:
        with self._lock:
            rec = self._index.get(image_id)
            if rec is None:
                raise KeyError(image_id)
            self._workspaces.move_to_end(rec.workspace_id)
            return rec

    def delete_image(self, image_id: str) -> bool:
        with self._lock:
            rec = self._index.pop(image_id, None)
            if rec is None:
                return False
            ws = self._workspaces.get(rec.workspace_id)
            if ws:
                ws.images.pop(image_id, None)
                for i, r in enumerate(ws.images.values()):
                    r.index = i
            if self.on_evict:
                self.on_evict(image_id)
            return True

    def find_annotation(self, ann_id: str) -> tuple[ImageRecord, Annotation]:
        with self._lock:
            for rec in self._index.values():
                ann = rec.annotations.get(ann_id)
                if ann is not None:
                    return rec, ann
            raise KeyError(ann_id)

    # ---- annotations -------------------------------------------------

    def add_annotation(self, rec: ImageRecord, *, frame: int, label: str, prompts: dict,
                       window: dict | None, threshold: float, mask_index: int,
                       mask: np.ndarray, score: float | None,
                       strokes: list[dict] | None = None) -> Annotation:
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
                strokes=list(strokes or []),
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
