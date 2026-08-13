from __future__ import annotations
import io
import posixpath
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import settings
from labels import canonical, hex_to_rgb
from media.loader import load_batch
from models.sam2_runner import Sam2Runner
from schemas import (
    AnnotationCreate, AnnotationOut, AnnotationUpdate, PreviewRequest,
    Stroke, Window, WorkspaceCreate,
)
from store import Annotation, ImageRecord, Store, Workspace
from utils.images import array_to_png, mask_to_png, overlay_png
from utils.paint import apply_strokes
from utils.rle import rle_to_mask

app = FastAPI(title="sam2web -- DICOM & image segmentation")
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


def _ws(workspace_id: str) -> Workspace:
    try:
        return store.get_workspace(workspace_id)
    except KeyError:
        raise HTTPException(404, f"Unknown workspace {workspace_id!r}. It may have been evicted; re-upload.")


def _win(rec: ImageRecord, frame: int, window: Optional[Window]) -> tuple[float, float]:
    if window is not None:
        return float(window.center), float(window.width)
    return rec.source.default_window(frame)


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


@dataclass
class RunResult:
    mask: np.ndarray
    score: float | None
    num_candidates: int


def _run(rec: ImageRecord, frame: int, prompts, window: Optional[Window],
         threshold: float, mask_index: int,
         strokes: Optional[list[Stroke]] = None) -> RunResult:
    strokes = list(strokes or [])
    if prompts.is_empty() and not strokes:
        raise HTTPException(400, "At least one point, box, or brush stroke is required.")
    if frame < 0 or frame >= rec.source.frames:
        raise HTTPException(400, f"frame {frame} out of range (0..{rec.source.frames - 1})")

    wc, ww = _win(rec, frame, window)

    if prompts.is_empty():
        # Pure hand-drawn mask: no model call at all.
        h, w = rec.source.frame_shape(frame)
        base, score, cands = np.zeros((h, w), dtype=bool), None, 0
    else:
        rgb = rec.source.frame_rgb(frame, wc, ww)
        try:
            res = runner.segment(
                _embed_key(rec.image_id, frame, wc, ww), rgb,
                _prompts_dict(prompts), threshold, mask_index,
            )
        except ValueError as e:
            raise HTTPException(400, str(e))
        base, score, cands = res.mask, res.score, res.num_candidates

    mask = apply_strokes(base, [s.model_dump() for s in strokes]) if strokes else base
    return RunResult(mask=mask, score=score, num_candidates=cands)


def _out(rec: ImageRecord, ann: Annotation) -> AnnotationOut:
    return AnnotationOut(
        id=ann.id, image_id=ann.image_id, frame=ann.frame, label=ann.label,
        instance=ann.instance, color=rec.labels.color_for(ann.label),
        area=ann.area, bbox=ann.bbox, prompts=ann.prompts,
        window=Window(**ann.window) if ann.window else None,
        threshold=ann.threshold, mask_index=ann.mask_index,
        strokes=[Stroke(**s) for s in ann.strokes], score=ann.score,
        created_at=ann.created_at, updated_at=ann.updated_at,
    )


# ---- pages ------------------------------------------------------------

if (STATIC / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")


@app.get("/")
def index():
    page = STATIC / "index.html"
    if not page.exists():
        raise HTTPException(
            503, "Frontend not built. Run `npm --prefix frontend install && npm --prefix frontend run build`."
        )
    return FileResponse(page, headers={"Cache-Control": "no-store"})


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": "stub" if runner.stub else runner.model_id,
        "device": str(runner.device),
        "cached_embeddings": len(runner._cache),
        "workspaces": len(store.workspaces()),
    }


# ---- workspaces & files -----------------------------------------------

@app.post("/workspaces")
def create_workspace(req: WorkspaceCreate = WorkspaceCreate()):
    return store.create_workspace(req.name).to_listing()


@app.get("/workspaces")
def list_workspaces():
    return [{"workspace_id": w.workspace_id, "name": w.name,
             "image_count": len(w.images), "created_at": w.created_at}
            for w in store.workspaces()]


@app.get("/workspaces/{workspace_id}")
def get_workspace(workspace_id: str):
    return _ws(workspace_id).to_listing()


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...),
                 workspace_id: Optional[str] = Form(None),
                 name: Optional[str] = Form(None)):
    """Add files to a workspace. A folder pick arrives here as many parts, and a
    .zip is expanded into its members -- one list entry per image either way."""
    payloads = [((f.filename or "upload").strip(), await f.read()) for f in files]
    if not payloads:
        raise HTTPException(400, "No files uploaded.")

    loaded, errors = load_batch(payloads)
    if not loaded and errors:
        raise HTTPException(400, "No readable images. " + "; ".join(errors[:5]))

    ws = _ws(workspace_id) if workspace_id else store.create_workspace(
        name or posixpath.dirname(payloads[0][0]) or "workspace"
    )

    added = []
    for filename, source in loaded:
        try:
            rec = store.add_image(ws, source, filename)
        except ValueError as e:
            errors.append(str(e))
            break
        added.append(rec.to_listing())

    return {"workspace_id": ws.workspace_id, "added": len(added),
            "errors": errors, "images": added, "workspace": ws.to_listing()}


@app.post("/dicom/upload")
async def upload_single(file: UploadFile = File(...),
                        workspace_id: Optional[str] = Form(None)):
    """Single-file upload. Kept because it is the smallest possible client."""
    raw = await file.read()
    name = (file.filename or "upload").strip()
    loaded, errors = load_batch([(name, raw)])
    if not loaded:
        raise HTTPException(400, "Failed to read image. " + "; ".join(errors[:3]))

    ws = _ws(workspace_id) if workspace_id else store.create_workspace(name)
    recs = [store.add_image(ws, src, fname) for fname, src in loaded]
    rec = recs[0]
    wc, ww = rec.source.default_window(0)
    return {
        "workspace_id": ws.workspace_id,
        "image_id": rec.image_id,
        "filename": rec.filename,
        "kind": rec.source.kind,
        "frames": rec.source.frames,
        "meta": rec.source.meta,
        "default_window": {"center": wc, "width": ww},
        "images": [r.to_listing() for r in recs],
        "errors": errors,
    }


class ImagePatch(BaseModel):
    reviewed: Optional[bool] = None


@app.patch("/images/{image_id}")
def patch_image(image_id: str, req: ImagePatch):
    rec = _rec(image_id)
    if req.reviewed is not None:
        rec.reviewed = bool(req.reviewed)
    return rec.to_listing()


@app.delete("/images/{image_id}")
def delete_image(image_id: str):
    if not store.delete_image(image_id):
        raise HTTPException(404, f"Unknown image_id {image_id!r}")
    return {"deleted": image_id}


@app.get("/frame_info")
def frame_info(image_id: str = Query(...), frame: int = Query(0)):
    rec = _rec(image_id)
    try:
        rows, cols = rec.source.frame_shape(frame)
    except IndexError as e:
        raise HTTPException(400, str(e))
    wc, ww = rec.source.default_window(frame)
    return {"frame": frame, "rows": rows, "columns": cols,
            "windowing": rec.source.windowing,
            "default_window": {"center": wc, "width": ww}}


@app.get("/frame.png")
def frame_png(image_id: str = Query(...), frame: int = Query(0),
              wc: Optional[float] = None, ww: Optional[float] = None):
    rec = _rec(image_id)
    try:
        arr = rec.source.frame_uint8(frame, wc, ww)
    except IndexError as e:
        raise HTTPException(400, str(e))
    return Response(array_to_png(arr), media_type="image/png",
                    headers={"Cache-Control": "no-store"})


# ---- transient preview -------------------------------------------------

@app.post("/segment/preview.png")
def preview_png(req: PreviewRequest, color: str = Query("#E8453C"), alpha: int = Query(110)):
    """Uncommitted mask for the prompts currently being placed."""
    rec = _rec(req.image_id)
    res = _run(rec, req.frame, req.prompts, req.window, req.threshold,
               req.mask_index, req.strokes)
    png = overlay_png(res.mask.shape, [{"mask": res.mask, "color": hex_to_rgb(color)}], alpha)
    return Response(png, media_type="image/png", headers={
        "X-Mask-Score": f"{res.score:.4f}" if res.score is not None else "",
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
    res = _run(rec, req.frame, req.prompts, req.window, req.threshold,
               req.mask_index, req.strokes)
    if res.mask.sum() == 0:
        raise HTTPException(422, "Prompts produced an empty mask; nothing to save.")

    wc, ww = _win(rec, req.frame, req.window)
    ann = store.add_annotation(
        rec, frame=req.frame, label=req.label, prompts=req.prompts.model_dump(),
        window={"center": wc, "width": ww}, threshold=req.threshold,
        mask_index=req.mask_index, mask=res.mask, score=res.score,
        strokes=[s.model_dump() for s in req.strokes],
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
    mask_fields = (req.prompts, req.window, req.threshold, req.mask_index, req.strokes)
    mask = score = None
    if any(f is not None for f in mask_fields):
        prompts = req.prompts if req.prompts is not None else Prompts(**ann.prompts)
        window = req.window if req.window is not None else (Window(**ann.window) if ann.window else None)
        threshold = req.threshold if req.threshold is not None else ann.threshold
        mask_index = req.mask_index if req.mask_index is not None else ann.mask_index
        strokes = req.strokes if req.strokes is not None else [Stroke(**s) for s in ann.strokes]

        res = _run(rec, ann.frame, prompts, window, threshold, mask_index, strokes)
        if res.mask.sum() == 0:
            raise HTTPException(422, "Edit produced an empty mask; the previous mask was kept.")
        mask, score = res.mask, res.score

        wc, ww = _win(rec, ann.frame, window)
        ann.prompts = prompts.model_dump()
        ann.window = {"center": wc, "width": ww}
        ann.threshold = threshold
        ann.mask_index = mask_index
        ann.strokes = [s.model_dump() for s in strokes]

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
                        selected: Optional[str] = None, exclude: Optional[str] = None,
                        alpha: int = Query(110)):
    """All committed masks on this frame, composited with per-label colours.

    `exclude` drops one annotation: while it is being edited its live preview is
    drawn instead, and showing both would leave the old mask peeking out."""
    rec = _rec(image_id)
    try:
        h, w = rec.source.frame_shape(frame)
    except IndexError as e:
        raise HTTPException(400, str(e))

    items = []
    for a in rec.annotations.values():
        if a.frame != frame or a.id == exclude:
            continue
        items.append({
            "mask": rle_to_mask(a.rle),
            "color": hex_to_rgb(rec.labels.color_for(a.label)),
            "selected": a.id == selected,
        })

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
def labels(image_id: Optional[str] = None, workspace_id: Optional[str] = None):
    if workspace_id:
        return {"labels": _ws(workspace_id).label_summary()}
    if image_id:
        return {"labels": _rec(image_id).label_summary()}
    raise HTTPException(400, "Pass image_id or workspace_id.")


# ---- export ------------------------------------------------------------

def _records(image_id: Optional[str], workspace_id: Optional[str]) -> tuple[Workspace, list[ImageRecord]]:
    if workspace_id:
        ws = _ws(workspace_id)
        return ws, ws.records
    if image_id:
        rec = _rec(image_id)
        return _ws(rec.workspace_id), [rec]
    raise HTTPException(400, "Pass image_id or workspace_id.")


def _ann_dict(rec: ImageRecord, a: Annotation, include_masks: bool) -> dict:
    return {
        "id": a.id,
        "label": a.label,
        "instance": a.instance,
        "color": rec.labels.color_for(a.label),
        "frame": a.frame,
        "area": a.area,
        "bbox": a.bbox,  # [x, y, w, h]
        "score": a.score,
        "prompts": a.prompts,
        "strokes": a.strokes,
        "window": a.window,
        "threshold": a.threshold,
        "mask_index": a.mask_index,
        "created_at": a.created_at,
        "updated_at": a.updated_at,
        **({"mask": {"format": "coco_rle_uncompressed", "order": "fortran",
                     "size": a.rle["size"], "counts": a.rle["counts"]}}
           if include_masks else {}),
    }


def _image_dict(rec: ImageRecord, include_masks: bool) -> dict:
    anns = sorted(rec.annotations.values(), key=lambda a: (a.frame, canonical(a.label), a.instance))
    return {
        "image_id": rec.image_id,
        "filename": rec.filename,
        "index": rec.index,
        "kind": rec.source.kind,
        "frames": rec.source.frames,
        "reviewed": rec.reviewed,
        **rec.source.meta,
        "annotations": [_ann_dict(rec, a, include_masks) for a in anns],
    }


def _export_doc(ws: Workspace, records: list[ImageRecord], include_masks: bool) -> dict:
    from store import _summarise

    return {
        "schema_version": "2.0",
        "generator": "sam2web",
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {
            "id": "stub" if runner.stub else runner.model_id,
            "candidates_ranked_by": "predicted_iou",
            "mask_composition": "model(prompts) then strokes replayed in order",
        },
        "workspace": {
            "workspace_id": ws.workspace_id,
            "name": ws.name,
            "image_count": len(records),
            "annotation_count": sum(len(r.annotations) for r in records),
        },
        "labels": _summarise(records),
        "images": [_image_dict(r, include_masks) for r in records],
    }


@app.get("/export.json")
def export_json(image_id: Optional[str] = None, workspace_id: Optional[str] = None,
                include_masks: bool = Query(True)):
    """Every annotation in the workspace, in one document -- the whole point of
    labelling a folder is not having to export file by file."""
    ws, records = _records(image_id, workspace_id)
    return _export_doc(ws, records, include_masks)


def _safe(name: str) -> str:
    stem = posixpath.basename(name) or "image"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "image"


@app.get("/export.zip")
def export_zip(image_id: Optional[str] = None, workspace_id: Optional[str] = None):
    """annotations.json plus one PNG per mask, for tools that want pixels."""
    import json

    ws, records = _records(image_id, workspace_id)
    doc = _export_doc(ws, records, include_masks=True)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("annotations.json", json.dumps(doc, indent=2))
        for rec in records:
            folder = f"masks/{rec.index:04d}_{_safe(rec.filename)}"
            for a in rec.annotations.values():
                png = mask_to_png(rle_to_mask(a.rle))
                zf.writestr(f"{folder}/f{a.frame:03d}_{_safe(a.label)}_{a.instance:02d}.png", png)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Response(buf.getvalue(), media_type="application/zip", headers={
        "Content-Disposition": f'attachment; filename="sam2web-{stamp}.zip"',
    })
