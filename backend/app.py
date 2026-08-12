from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from config import settings
from dicom.io import (
    default_window, frame_rgb, frame_shape, frame_shapes, frame_uint8,
    has_uniform_geometry, load_single, load_zip_series,
)
from labels import canonical, hex_to_rgb
from models.sam2_runner import Sam2Runner
from schemas import AnnotationCreate, AnnotationOut, AnnotationUpdate, PreviewRequest, Window
from store import Annotation, ImageRecord, Store
from utils.images import array_to_png, mask_to_png, overlay_png
from utils.rle import rle_to_mask

app = FastAPI(title="sam2web -- DICOM segmentation")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

STATIC = Path(__file__).parent / "static"
store = Store()
runner = Sam2Runner()
store.on_evict = runner.drop_image


# ---- helpers ----------------------------------------------------------

def _rec(image_id: str) -> ImageRecord:
    try:
        return store.get(image_id)
    except KeyError:
        raise HTTPException(404, f"Unknown image_id {image_id!r}. It may have been evicted; re-upload.")


def _win(rec: ImageRecord, frame: int, window: Optional[Window]) -> tuple[float, float]:
    if window is not None:
        return float(window.center), float(window.width)
    return default_window(rec.image, frame)


def _embed_key(image_id: str, frame: int, wc: float, ww: float) -> str:
    # Windowing changes the pixels handed to the encoder, so it has to be part
    # of the cache key -- otherwise re-windowing would silently reuse stale
    # embeddings computed from a different-looking image.
    return f"{image_id}:{frame}:{round(wc, 2)}:{round(ww, 2)}"


def _prompts_dict(prompts) -> dict:
    return {
        "points": [[p.x, p.y, p.label] for p in prompts.points],
        "boxes": [list(map(int, b)) for b in prompts.boxes],
    }


def _run(rec: ImageRecord, frame: int, prompts, window: Optional[Window],
         threshold: float, mask_index: int):
    if prompts.is_empty():
        raise HTTPException(400, "At least one point or box prompt is required.")
    if frame < 0 or frame >= rec.image.frames:
        raise HTTPException(400, f"frame {frame} out of range (0..{rec.image.frames - 1})")

    wc, ww = _win(rec, frame, window)
    rgb = frame_rgb(rec.image, frame, wc, ww)
    try:
        return runner.segment(
            _embed_key(rec.image_id, frame, wc, ww), rgb,
            _prompts_dict(prompts), threshold, mask_index,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


def _out(rec: ImageRecord, ann: Annotation) -> AnnotationOut:
    return AnnotationOut(
        id=ann.id, image_id=ann.image_id, frame=ann.frame, label=ann.label,
        instance=ann.instance, color=rec.labels.color_for(ann.label),
        area=ann.area, bbox=ann.bbox, prompts=ann.prompts,
        window=Window(**ann.window) if ann.window else None,
        threshold=ann.threshold, mask_index=ann.mask_index, score=ann.score,
        created_at=ann.created_at, updated_at=ann.updated_at,
    )


# ---- pages ------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": "stub" if runner.stub else runner.model_id,
        "device": str(runner.device),
        "cached_embeddings": len(runner._cache),
    }


# ---- images -----------------------------------------------------------

@app.post("/dicom/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    name = (file.filename or "upload").strip()
    try:
        image = load_zip_series(raw) if name.lower().endswith(".zip") else load_single(raw)
    except Exception as e:
        raise HTTPException(400, f"Failed to read DICOM: {e}")

    rec = store.add(image, name)
    wc, ww = default_window(image, 0)
    return {
        "image_id": rec.image_id,
        "filename": name,
        "frames": image.frames,
        "meta": image.meta,
        "default_window": {"center": wc, "width": ww},
        # Per-frame geometry: a zipped folder can mix image sizes, and the
        # viewer must resize per frame or clicks land at the wrong coordinates.
        "frame_shapes": frame_shapes(image),
        "uniform_geometry": has_uniform_geometry(image),
    }


@app.get("/frame_info")
def frame_info(image_id: str = Query(...), frame: int = Query(0)):
    rec = _rec(image_id)
    try:
        rows, cols = frame_shape(rec.image, frame)
    except IndexError as e:
        raise HTTPException(400, str(e))
    wc, ww = default_window(rec.image, frame)
    return {"frame": frame, "rows": rows, "columns": cols,
            "default_window": {"center": wc, "width": ww}}


@app.get("/frame.png")
def frame_png(image_id: str = Query(...), frame: int = Query(0),
              wc: Optional[float] = None, ww: Optional[float] = None):
    rec = _rec(image_id)
    try:
        arr = frame_uint8(rec.image, frame, wc, ww)
    except IndexError as e:
        raise HTTPException(400, str(e))
    return Response(array_to_png(arr), media_type="image/png",
                    headers={"Cache-Control": "no-store"})


# ---- transient preview -------------------------------------------------

@app.post("/segment/preview.png")
def preview_png(req: PreviewRequest, color: str = Query("#E8453C"), alpha: int = Query(110)):
    """Uncommitted mask for the prompts currently being placed."""
    rec = _rec(req.image_id)
    res = _run(rec, req.frame, req.prompts, req.window, req.threshold, req.mask_index)
    png = overlay_png(res.mask.shape, [{"mask": res.mask, "color": hex_to_rgb(color)}], alpha)
    return Response(png, media_type="image/png", headers={
        "X-Mask-Score": f"{res.score:.4f}",
        "X-Mask-Area": str(int(res.mask.sum())),
        "X-Mask-Candidates": str(res.num_candidates),
        "Cache-Control": "no-store",
    })


# ---- annotations -------------------------------------------------------

@app.post("/annotations", response_model=AnnotationOut)
def create_annotation(req: AnnotationCreate):
    rec = _rec(req.image_id)
    if not req.label.strip():
        raise HTTPException(400, "label must not be empty.")
    res = _run(rec, req.frame, req.prompts, req.window, req.threshold, req.mask_index)
    if res.mask.sum() == 0:
        raise HTTPException(422, "Prompts produced an empty mask; nothing to save.")

    wc, ww = _win(rec, req.frame, req.window)
    ann = store.add_annotation(
        rec, frame=req.frame, label=req.label, prompts=req.prompts.model_dump(),
        window={"center": wc, "width": ww}, threshold=req.threshold,
        mask_index=req.mask_index, mask=res.mask, score=res.score,
    )
    return _out(rec, ann)


@app.get("/annotations", response_model=list[AnnotationOut])
def list_annotations(image_id: str = Query(...), frame: Optional[int] = None):
    rec = _rec(image_id)
    anns = [a for a in rec.annotations.values() if frame is None or a.frame == frame]
    return [_out(rec, a) for a in anns]


@app.patch("/annotations/{ann_id}", response_model=AnnotationOut)
def update_annotation(ann_id: str, req: AnnotationUpdate):
    try:
        rec, ann = store.find_annotation(ann_id)
    except KeyError:
        raise HTTPException(404, f"Unknown annotation {ann_id!r}")

    from schemas import Prompts

    # Re-segment only if something that shapes the mask actually changed.
    mask_fields = (req.prompts, req.window, req.threshold, req.mask_index)
    mask = score = None
    if any(f is not None for f in mask_fields):
        prompts = req.prompts if req.prompts is not None else Prompts(**ann.prompts)
        window = req.window if req.window is not None else (Window(**ann.window) if ann.window else None)
        threshold = req.threshold if req.threshold is not None else ann.threshold
        mask_index = req.mask_index if req.mask_index is not None else ann.mask_index

        res = _run(rec, ann.frame, prompts, window, threshold, mask_index)
        if res.mask.sum() == 0:
            raise HTTPException(422, "Edit produced an empty mask; the previous mask was kept.")
        mask, score = res.mask, res.score

        wc, ww = _win(rec, ann.frame, window)
        ann.prompts = prompts.model_dump()
        ann.window = {"center": wc, "width": ww}
        ann.threshold = threshold
        ann.mask_index = mask_index

    store.update_annotation(rec, ann, mask=mask, score=score, label=req.label)
    return _out(rec, ann)


@app.delete("/annotations/{ann_id}")
def delete_annotation(ann_id: str):
    try:
        rec, _ = store.find_annotation(ann_id)
    except KeyError:
        raise HTTPException(404, f"Unknown annotation {ann_id!r}")
    store.delete_annotation(rec, ann_id)
    return {"deleted": ann_id}


@app.get("/annotations/overlay.png")
def annotations_overlay(image_id: str = Query(...), frame: int = Query(0),
                        selected: Optional[str] = None, alpha: int = Query(110)):
    """All committed masks on this frame, composited with per-label colours."""
    rec = _rec(image_id)
    try:
        h, w = frame_shape(rec.image, frame)
    except IndexError as e:
        raise HTTPException(400, str(e))

    items = []
    for a in rec.annotations.values():
        if a.frame != frame:
            continue
        h, w = a.rle["size"]
        items.append({
            "mask": rle_to_mask(a.rle),
            "color": hex_to_rgb(rec.labels.color_for(a.label)),
            "selected": a.id == selected,
        })
    if not items and (h == 0 or w == 0):
        raise HTTPException(400, "Unknown frame size for an image with no annotations.")

    return Response(overlay_png((h, w), items, alpha), media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@app.get("/annotations/{ann_id}/mask.png")
def annotation_mask_png(ann_id: str):
    try:
        _, ann = store.find_annotation(ann_id)
    except KeyError:
        raise HTTPException(404, f"Unknown annotation {ann_id!r}")
    return Response(mask_to_png(rle_to_mask(ann.rle)), media_type="image/png")


@app.get("/labels")
def labels(image_id: str = Query(...)):
    return {"labels": _rec(image_id).label_summary()}


# ---- export ------------------------------------------------------------

@app.get("/export.json")
def export_json(image_id: str = Query(...), include_masks: bool = Query(True)):
    rec = _rec(image_id)
    anns = sorted(rec.annotations.values(), key=lambda a: (a.frame, canonical(a.label), a.instance))

    payload = {
        "schema_version": "1.0",
        "generator": "sam2web",
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {
            "id": "stub" if runner.stub else runner.model_id,
            "candidates_ranked_by": "predicted_iou",
        },
        "image": {
            "image_id": rec.image_id,
            "filename": rec.filename,
            "kind": rec.image.kind,
            "frames": rec.image.frames,
            **rec.image.meta,
        },
        "labels": rec.label_summary(),
        "annotations": [
            {
                "id": a.id,
                "label": a.label,
                "instance": a.instance,
                "color": rec.labels.color_for(a.label),
                "frame": a.frame,
                "area": a.area,
                "bbox": a.bbox,  # [x, y, w, h]
                "score": a.score,
                "prompts": a.prompts,
                "window": a.window,
                "threshold": a.threshold,
                "mask_index": a.mask_index,
                "created_at": a.created_at,
                "updated_at": a.updated_at,
                **({"mask": {"format": "coco_rle_uncompressed", "order": "fortran",
                             "size": a.rle["size"], "counts": a.rle["counts"]}}
                   if include_masks else {}),
            }
            for a in anns
        ],
    }
    return payload
