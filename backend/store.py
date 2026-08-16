"""In-memory session state: workspaces, images and annotations.

The data model is workspace -> image -> frame -> annotation. Two things it is
worth being explicit about, because both are load-bearing:

Geometry is per image, never per workspace. A picked folder routinely mixes
64-square, 512-square and 1955x1841 files, and assuming one size for the batch
sends clicks to the wrong coordinates with no error at all.

Colours are per workspace, never per image, which is what makes "liver" the same
colour in every file of a folder -- the whole point of labelling a batch.

Nothing here is persisted. Eviction is per workspace, never per image, so a
folder cannot lose files mid-annotation.
"""
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
    """Get the current time for a timestamp field.

    Returns:
        UTC ISO-8601, seconds resolution.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Annotation:
    """One labelled instance on one frame.

    Everything needed to re-derive the mask is stored, not just the mask, which
    is what makes an exported annotation reproducible rather than merely
    readable.

    Attributes:
        id: Annotation identifier.
        image_id: Image it belongs to.
        frame: Frame index within that image.
        label: Display form, as first typed by the user.
        instance: 1-based, unique per (image_id, label).
        prompts: ``{"points": [{x, y, label}], "boxes": [[x1, y1, x2, y2]]}``.
        window: ``{"center": c, "width": w}`` the mask was produced under, or
            None.
        threshold: Foreground probability threshold used.
        mask_index: Which ranked candidate was taken.
        rle: The mask, as column-major COCO run lengths.
        area: Foreground pixel count.
        bbox: ``[x, y, width, height]``, or None if empty.
        score: Predicted IoU, or None for a purely hand-drawn mask.
        strokes: Hand corrections, replayed on top of the model mask in order.
            Keeping them separate from the prompts is what lets a brush fix
            survive re-prompting.
        created_at: UTC ISO-8601 timestamp.
        updated_at: UTC ISO-8601 timestamp of the last edit.
    """
    id: str
    image_id: str
    frame: int
    label: str
    instance: int
    prompts: dict
    window: dict | None
    threshold: float
    mask_index: int
    rle: dict
    area: int
    bbox: list[int] | None
    score: float | None
    strokes: list[dict] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)


@dataclass
class ImageRecord:
    """One file in a workspace, with its own geometry and its annotations.

    Attributes:
        image_id: Image identifier.
        source: The loaded file.
        filename: Path within the picked folder, as uploaded.
        workspace_id: Owning workspace.
        index: Position in the workspace file list.
        labels: The workspace's registry, shared so a label keeps its colour
            across the folder.
        annotations: Annotation id to annotation, in creation order.
        _instance_counter: Canonical label to highest instance number handed
            out. Never decremented, so instance numbers stay stable when an
            annotation is deleted.
        reviewed: Whether the user marked this file done.
        created_at: UTC ISO-8601 timestamp.
    """
    image_id: str
    source: ImageSource
    filename: str
    workspace_id: str = ""
    index: int = 0
    labels: LabelRegistry = field(default_factory=LabelRegistry)
    annotations: "OrderedDict[str, Annotation]" = field(default_factory=OrderedDict)
    _instance_counter: dict[str, int] = field(default_factory=dict)
    reviewed: bool = False
    created_at: str = field(default_factory=_now)

    def next_instance(self, label: str) -> int:
        """Hand out the next instance number for a label on this image.

        Args:
            label: Label as typed; matched canonically.

        Returns:
            A 1-based number, unique within this image and label for the
            lifetime of the record.
        """
        key = canonical(label)
        n = self._instance_counter.get(key, 0) + 1
        self._instance_counter[key] = n
        return n

    def display_label(self, label: str) -> str:
        """Resolve how a label should be spelled on this image.

        Reuses the casing of the first occurrence, so ``Spine`` and ``spine``
        merge into one label rather than two.

        Args:
            label: Label as typed.

        Returns:
            The existing display form if this label is already used here,
            otherwise the input stripped of surrounding whitespace.
        """
        key = canonical(label)
        for a in self.annotations.values():
            if canonical(a.label) == key:
                return a.label
        return label.strip()

    def label_summary(self) -> list[dict]:
        """Summarise the labels used on this image.

        Returns:
            ``[{"name", "color", "count"}]``, sorted by name.
        """
        return _summarise([self])

    def to_listing(self) -> dict:
        """Render the row the file list shows for this image.

        Returns:
            Identity, geometry, modality, whether windowing applies, annotation
            count, the labels present and the reviewed flag.
        """
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
    """Count annotations per label across images.

    Args:
        records: The images to summarise; one for an image summary, all of them
            for a workspace summary.

    Returns:
        ``[{"name", "color", "count"}]``, sorted case-insensitively by name.
    """
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

    Colours live here rather than on the image so that "liver" is the same
    colour in every file of a folder -- the whole point of labelling a batch.

    Attributes:
        workspace_id: Workspace identifier.
        name: Display name, usually the picked folder.
        labels: Label to colour assignment, shared by every image.
        images: Image id to record, in file-list order.
        created_at: UTC ISO-8601 timestamp.
    """
    workspace_id: str
    name: str = ""
    labels: LabelRegistry = field(default_factory=LabelRegistry)
    images: "OrderedDict[str, ImageRecord]" = field(default_factory=OrderedDict)
    created_at: str = field(default_factory=_now)

    @property
    def records(self) -> list[ImageRecord]:
        """The images, in file-list order.

        Returns:
            list[ImageRecord]: A new list; mutating it does not touch the
            workspace.
        """
        return list(self.images.values())

    def label_summary(self) -> list[dict]:
        """Summarise the labels used anywhere in the workspace.

        Returns:
            ``[{"name", "color", "count"}]``, sorted by name.
        """
        return _summarise(self.records)

    def to_listing(self) -> dict:
        """Render the workspace as the UI consumes it.

        Returns:
            Identity, counts, the label summary, and one listing per image.
        """
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
    """In-memory session state. Bounded so a long-lived server cannot grow forever.

    Workspaces are held in LRU order and evicted whole. Every mutating method
    takes a re-entrant lock, since FastAPI serves requests from a thread pool.

    Attributes:
        on_evict: Optional callback invoked with each ``image_id`` as it leaves
            the store. The app wires this to the runner so cached embeddings go
            with the images they belong to.
    """

    def __init__(self, max_workspaces: int | None = None) -> None:
        """Create an empty store.

        Args:
            max_workspaces: How many workspaces stay resident. Defaults to
                ``settings.max_workspaces``.
        """
        self._workspaces: OrderedDict[str, Workspace] = OrderedDict()
        self._index: dict[str, ImageRecord] = {}     # image_id -> record
        self._max = max_workspaces or settings.max_workspaces
        self._lock = threading.RLock()
        self.on_evict = None

    # ---- workspaces ---------------------------------------------------

    def create_workspace(self, name: str = "") -> Workspace:
        """Create a workspace, evicting the oldest if the cap is exceeded.

        Args:
            name: Display name, usually the picked folder.

        Returns:
            The new workspace.
        """
        with self._lock:
            ws = Workspace(workspace_id=str(uuid.uuid4()), name=name)
            self._workspaces[ws.workspace_id] = ws
            self._evict_locked()
            return ws

    def get_workspace(self, workspace_id: str) -> Workspace:
        """Fetch a workspace and mark it recently used.

        Args:
            workspace_id: Workspace identifier.

        Returns:
            The workspace.

        Raises:
            KeyError: If it is unknown or has been evicted.
        """
        with self._lock:
            if workspace_id not in self._workspaces:
                raise KeyError(workspace_id)
            self._workspaces.move_to_end(workspace_id)
            return self._workspaces[workspace_id]

    def workspaces(self) -> list[Workspace]:
        """List resident workspaces.

        Returns:
            list[Workspace]: Least recently used first.
        """
        with self._lock:
            return list(self._workspaces.values())

    def _evict_locked(self) -> None:
        """Drop least-recently-used workspaces until the cap is met.

        Whole workspaces go at once, never individual images, so a folder cannot
        lose files mid-annotation. Each departing image is passed to
        :attr:`on_evict`. The caller must hold the lock.
        """
        while len(self._workspaces) > self._max:
            _, old = self._workspaces.popitem(last=False)
            for image_id in old.images:
                self._index.pop(image_id, None)
                if self.on_evict:
                    self.on_evict(image_id)

    # ---- images -------------------------------------------------------

    def add_image(self, ws: Workspace, source: ImageSource, filename: str) -> ImageRecord:
        """Append a file to a workspace.

        Args:
            ws: Workspace to add to.
            source: The loaded file.
            filename: Path within the picked folder.

        Returns:
            The new record, sharing the workspace's label registry.

        Raises:
            ValueError: If the workspace already holds ``settings.max_files``
                files.
        """
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
        """Fetch an image and mark its workspace recently used.

        Args:
            image_id: Image identifier.

        Returns:
            The record.

        Raises:
            KeyError: If it is unknown or its workspace has been evicted.
        """
        with self._lock:
            rec = self._index.get(image_id)
            if rec is None:
                raise KeyError(image_id)
            self._workspaces.move_to_end(rec.workspace_id)
            return rec

    def delete_image(self, image_id: str) -> bool:
        """Remove one file from its workspace, renumbering the ones after it.

        Args:
            image_id: Image identifier.

        Returns:
            True if it was there, False if it was not.
        """
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
        """Locate an annotation without knowing which image holds it.

        Args:
            ann_id: Annotation identifier.

        Returns:
            ``(record, annotation)``.

        Raises:
            KeyError: If no resident image holds it.
        """
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
        """Commit a mask as a labelled instance.

        Args:
            rec: Image the mask belongs to.
            frame: Frame index within that image.
            label: Label as typed; the display form and instance number are
                resolved here.
            prompts: Points and boxes that produced the mask.
            window: ``{"center", "width"}`` the mask was produced under.
            threshold: Foreground probability threshold used.
            mask_index: Which ranked candidate was taken.
            mask: HxW binary mask; encoded to RLE, and its area and bounding box
                measured.
            score: Predicted IoU, or None for a hand-drawn mask.
            strokes: Brush strokes to store beside the prompts.

        Returns:
            The stored annotation.
        """
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
        """Apply an edit to an existing annotation, in place.

        Args:
            rec: Image the annotation belongs to.
            ann: The annotation to edit.
            mask: New HxW binary mask, or None to leave the geometry alone. When
                given, the RLE, area and bounding box are recomputed.
            score: Predicted IoU that goes with ``mask``.
            **fields: Any other attribute to set, e.g. ``label``, ``prompts``,
                ``window``, ``threshold``. None values are ignored, so a partial
                update only touches what it names. Renaming to a label that is
                already used on this image adopts its display form, and either
                way the annotation takes a fresh instance number under the new
                label.

        Returns:
            The same annotation, with ``updated_at`` refreshed.
        """
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
        """Remove one annotation.

        Instance numbers are not reused afterwards, so the remaining masks keep
        the names the user learned.

        Args:
            rec: Image the annotation belongs to.
            ann_id: Annotation identifier.

        Returns:
            True if it was there, False if it was not.
        """
        with self._lock:
            return rec.annotations.pop(ann_id, None) is not None
