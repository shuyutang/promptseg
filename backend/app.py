"""FastAPI application: routes, request plumbing and the export document.

Endpoint docstrings are also what FastAPI publishes at ``/docs``, so they are
written to be read there; ``docs/api.md`` lists the same routes in one table.

Three things happen at import time and are worth knowing about. The model is
constructed here, so the first server start blocks until the weights are
downloaded and loaded. The store's eviction hook is wired to the runner, so
dropping a workspace also frees the embeddings its images were holding. And the
session database is opened, so annotating writes through to disk as it happens
and a restart does not cost the user work they never exported.
"""
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
from media.loader import load_batch, load_one
from models.sam2_runner import Sam2Runner
from persistence import SessionDB
from schemas import (
    AnnotationCreate, AnnotationOut, AnnotationUpdate, PreviewRequest,
    Stroke, Window, WorkspaceCreate,
)
from store import Annotation, ImageRecord, Store, Workspace
from utils.images import array_to_png, mask_to_png, overlay_png
from utils.paint import apply_strokes
from utils.rle import rle_to_mask

app = FastAPI(title="promptseg -- DICOM & image segmentation")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

STATIC = Path(__file__).parent / "static"
"""Where the built frontend lives. Committed, so a fresh clone runs without Node."""

def _open_db() -> SessionDB | None:
    """Open the session database, or carry on without one.

    An unwritable data directory is a reason to lose persistence, not a reason
    to refuse to start -- the tool is still fully usable with export as the only
    way out, which is what it did before.

    Returns:
        The database, or None when persistence is off or unavailable.
    """
    if not settings.persist:
        return None
    try:
        db = SessionDB(settings.data_dir)
        db.prune(settings.max_saved)
        return db
    except Exception as e:                                  # noqa: BLE001
        print(f"promptseg: sessions will not be saved ({type(e).__name__}: {e})")
        return None


db = _open_db()
"""Durable session storage, or None when ``SAM2_PERSIST=0``."""

store = Store(db=db)
runner = Sam2Runner()
store.on_evict = runner.drop_image


# ---- helpers ----------------------------------------------------------

def _rec(image_id: str) -> ImageRecord:
    """Look up an image, or fail the request.

    Args:
        image_id: Image identifier from the client.

    Returns:
        The record.

    Raises:
        HTTPException: 404 if it is unknown or its workspace was evicted.
    """
    try:
        return store.get(image_id)
    except KeyError:
        raise HTTPException(404, f"Unknown image_id {image_id!r}. It may have been evicted; re-upload.")


def _ws(workspace_id: str) -> Workspace:
    """Look up a workspace, or fail the request.

    Args:
        workspace_id: Workspace identifier from the client.

    Returns:
        The workspace.

    Raises:
        HTTPException: 404 if it is unknown or was evicted.
    """
    try:
        return store.get_workspace(workspace_id)
    except KeyError:
        raise HTTPException(404, f"Unknown workspace {workspace_id!r}. It may have been evicted; re-upload.")


def _win(rec: ImageRecord, frame: int, window: Optional[Window]) -> tuple[float, float]:
    """Resolve the window a request should be rendered under.

    Args:
        rec: The image.
        frame: Frame index.
        window: Explicit window, or None to use the file's default.

    Returns:
        ``(center, width)``.
    """
    if window is not None:
        return float(window.center), float(window.width)
    return rec.source.default_window(frame)


def _embed_key(image_id: str, frame: int, wc: float, ww: float) -> str:
    """Build the embedding cache key.

    Windowing changes the pixels handed to the encoder, so it has to be part of
    the key -- otherwise re-windowing would silently reuse stale embeddings
    computed from a different-looking image.

    Args:
        image_id: Image identifier.
        frame: Frame index.
        wc: Window centre.
        ww: Window width.

    Returns:
        A key of the form ``image_id:frame:center:width``, window values rounded
        to 2 decimals so slider jitter does not miss the cache.
    """
    return f"{image_id}:{frame}:{round(wc, 2)}:{round(ww, 2)}"


def _prompts_dict(prompts) -> dict:
    """Flatten API prompts into what the runner expects.

    Args:
        prompts: A :class:`schemas.Prompts`.

    Returns:
        ``{"points": [[x, y, label]], "boxes": [[x1, y1, x2, y2]]}``.
    """
    return {
        "points": [[p.x, p.y, p.label] for p in prompts.points],
        "boxes": [list(map(int, b)) for b in prompts.boxes],
    }


@dataclass
class RunResult:
    """The outcome of one segmentation request.

    Attributes:
        mask: HxW binary mask, strokes already replayed.
        score: Predicted IoU, or None when no model call was made.
        num_candidates: How many candidates the model offered; 0 for a purely
            hand-drawn mask.
    """
    mask: np.ndarray
    score: float | None
    num_candidates: int


def _run(rec: ImageRecord, frame: int, prompts, window: Optional[Window],
         threshold: float, mask_index: int,
         strokes: Optional[list[Stroke]] = None) -> RunResult:
    """Produce a mask: segment if there are prompts, then replay the strokes.

    This is the one place a mask is composed, shared by preview, create and
    update so all three agree on what a given request means.

    Args:
        rec: The image.
        frame: Frame index within it.
        prompts: A :class:`schemas.Prompts`. May be empty if strokes are given.
        window: Display window, or None for the file's default.
        threshold: Foreground probability threshold.
        mask_index: Which ranked candidate to take.
        strokes: Brush strokes to replay on top of the model output.

    Returns:
        RunResult: The composed mask and what produced it.

    Raises:
        HTTPException: 400 if there is nothing to act on, if the frame index is
            out of range, or if the runner rejects the prompts.
    """
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
    """Render a stored annotation as the API response model.

    Args:
        rec: The image it belongs to, consulted for the label's colour.
        ann: The stored annotation.

    Returns:
        AnnotationOut: Everything except the mask pixels, which the client
        fetches as PNG.
    """
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
    """Serve the built frontend.

    Returns:
        FileResponse: ``static/index.html``, uncached so a rebuilt UI shows up
        on reload.

    Raises:
        HTTPException: 503 if the frontend has not been built.
    """
    page = STATIC / "index.html"
    if not page.exists():
        raise HTTPException(
            503, "Frontend not built. Run `npm --prefix frontend install && npm --prefix frontend run build`."
        )
    return FileResponse(page, headers={"Cache-Control": "no-store"})


@app.get("/health")
def health():
    """Report what actually came up.

    Returns:
        The model id (or ``"stub"``), the device in use, how many embeddings are
        cached, how many workspaces are resident and what the session store is
        doing. Checking ``device`` is the quickest way to catch a CPU-only torch
        on a CUDA box, and ``storage.error`` the quickest way to catch a data
        directory that cannot be written.
    """
    return {
        "ok": True,
        "model": "stub" if runner.stub else runner.model_id,
        "device": str(runner.device),
        "cached_embeddings": len(runner._cache),
        "workspaces": len(store.workspaces()),
        "persist": db is not None,
        "storage": db.stats() if db else None,
    }


# ---- workspaces & files -----------------------------------------------

@app.post("/workspaces")
def create_workspace(req: WorkspaceCreate = WorkspaceCreate()):
    """Create an empty workspace to upload into.

    Args:
        req: Carries the display name.

    Returns:
        The new workspace's listing.
    """
    return store.create_workspace(req.name).to_listing()


@app.get("/workspaces")
def list_workspaces():
    """List resident workspaces.

    Returns:
        One entry per workspace with its id, name, file count and creation time.
    """
    return [{"workspace_id": w.workspace_id, "name": w.name,
             "image_count": len(w.images), "created_at": w.created_at}
            for w in store.workspaces()]


@app.get("/workspaces/{workspace_id}")
def get_workspace(workspace_id: str):
    """Get one workspace: its file list, label summary and progress.

    Args:
        workspace_id: Workspace identifier.

    Returns:
        The workspace listing, including a row per image.

    Raises:
        HTTPException: 404 if the workspace is unknown or was evicted.
    """
    return _ws(workspace_id).to_listing()


# ---- saved sessions ---------------------------------------------------

@app.get("/sessions")
def list_sessions():
    """List the sessions on disk, so work can be picked up after a restart.

    Returns:
        ``{"persist": bool, "sessions": [...]}``. Each session carries its id,
        name, timestamps, file and annotation counts and the labels used, which
        is enough to recognise it without opening it. Most recently worked on
        first. ``persist`` is False when the server is running memory-only, and
        the list is then empty.
    """
    return {"persist": db is not None, "sessions": db.sessions() if db else []}


@app.post("/sessions/{workspace_id}/open")
def open_session(workspace_id: str):
    """Reopen a saved session, putting its images and masks back.

    The session keeps its identifiers, so this resumes the work rather than
    copying it. A session that is already resident is returned as it stands,
    which makes the call safe to repeat.

    Args:
        workspace_id: Workspace identifier, from :func:`list_sessions`.

    Returns:
        The workspace listing, in the same shape ``/workspaces/{id}`` returns,
        plus ``errors`` naming any file whose saved bytes could not be read.

    Raises:
        HTTPException: 404 if no such session was saved; 503 if the server is
            running memory-only.
    """
    if db is None:
        raise HTTPException(503, "Sessions are not being saved (SAM2_PERSIST=0).")
    try:
        return {**store.get_workspace(workspace_id).to_listing(), "errors": []}
    except KeyError:
        pass

    saved = db.load_workspace(workspace_id)
    if saved is None:
        raise HTTPException(404, f"Unknown session {workspace_id!r}")

    images, errors = [], []
    for img in saved.images:
        if not img.data:
            errors.append(f"{img.filename}: saved image data is missing")
            continue
        try:
            source = load_one(img.filename, img.data)
        except Exception as e:                              # noqa: BLE001
            errors.append(f"{img.filename}: {e}")
            continue
        images.append({"image_id": img.image_id, "source": source,
                       "filename": img.filename, "blob_sha": img.blob_sha,
                       "reviewed": img.reviewed,
                       "instances": img.instances, "created_at": img.created_at,
                       "annotations": img.annotations})

    ws = store.restore_workspace(saved.workspace_id, saved.name, saved.created_at,
                                 saved.label_colors, images)
    return {**ws.to_listing(), "errors": errors}


@app.delete("/sessions/{workspace_id}")
def delete_session(workspace_id: str):
    """Forget a saved session, on disk and in memory.

    Args:
        workspace_id: Workspace identifier.

    Returns:
        The deleted id.

    Raises:
        HTTPException: 404 if no such session was saved; 503 if the server is
            running memory-only.
    """
    if db is None:
        raise HTTPException(503, "Sessions are not being saved (SAM2_PERSIST=0).")
    store.forget_workspace(workspace_id)
    if not db.delete_workspace(workspace_id):
        raise HTTPException(404, f"Unknown session {workspace_id!r}")
    return {"deleted": workspace_id}


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...),
                 workspace_id: Optional[str] = Form(None),
                 name: Optional[str] = Form(None)):
    """Add files to a workspace.

    A folder pick arrives here as many parts, and a .zip is expanded into its
    members -- one list entry per image either way. Unreadable files are
    reported rather than failing the batch.

    Args:
        files: Uploaded parts. Each filename carries its path within the folder,
            which is what the file list and the export show.
        workspace_id: Workspace to append to. Omit to start a new one.
        name: Display name for a new workspace; defaults to the folder's name.

    Returns:
        The workspace id, how many files were added, per-file errors, the rows
        that were added and the full workspace listing.

    Raises:
        HTTPException: 400 if nothing was uploaded or nothing was readable; 404
            if ``workspace_id`` is unknown.
    """
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
    for item in loaded:
        try:
            rec = store.add_image(ws, item.source, item.name, item.data)
        except ValueError as e:
            errors.append(str(e))
            break
        added.append(rec.to_listing())

    return {"workspace_id": ws.workspace_id, "added": len(added),
            "errors": errors, "images": added, "workspace": ws.to_listing()}


@app.post("/dicom/upload")
async def upload_single(file: UploadFile = File(...),
                        workspace_id: Optional[str] = Form(None)):
    """Upload one file. Kept because it is the smallest possible client.

    Args:
        file: The uploaded file. A .zip is still expanded, so this can produce
            several images.
        workspace_id: Workspace to append to. Omit to start a new one.

    Returns:
        The first image's identity, geometry, metadata and default window, plus
        a listing for everything that loaded.

    Raises:
        HTTPException: 400 if the file could not be read; 404 if
            ``workspace_id`` is unknown.
    """
    raw = await file.read()
    name = (file.filename or "upload").strip()
    loaded, errors = load_batch([(name, raw)])
    if not loaded:
        raise HTTPException(400, "Failed to read image. " + "; ".join(errors[:3]))

    ws = _ws(workspace_id) if workspace_id else store.create_workspace(name)
    recs = [store.add_image(ws, i.source, i.name, i.data) for i in loaded]
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
    """Fields that can be changed on an image.

    Attributes:
        reviewed: Whether the user has marked this file done.
    """
    reviewed: Optional[bool] = None


@app.patch("/images/{image_id}")
def patch_image(image_id: str, req: ImagePatch):
    """Mark a file done, or not done.

    Args:
        image_id: Image identifier.
        req: The fields to change; unset ones are left alone.

    Returns:
        The image's updated listing row.

    Raises:
        HTTPException: 404 if the image is unknown or was evicted.
    """
    rec = _rec(image_id)
    if req.reviewed is not None:
        store.set_reviewed(rec, req.reviewed)
    return rec.to_listing()


@app.delete("/images/{image_id}")
def delete_image(image_id: str):
    """Drop a file from its workspace, freeing its cached embeddings.

    Args:
        image_id: Image identifier.

    Returns:
        The deleted id.

    Raises:
        HTTPException: 404 if the image is unknown.
    """
    if not store.delete_image(image_id):
        raise HTTPException(404, f"Unknown image_id {image_id!r}")
    return {"deleted": image_id}


@app.get("/frame_info")
def frame_info(image_id: str = Query(...), frame: int = Query(0)):
    """Get one frame's geometry and default window.

    Geometry is per image, so the client asks per file rather than assuming one
    size for the folder.

    Args:
        image_id: Image identifier.
        frame: Frame index within it.

    Returns:
        Rows and columns, whether window/level applies, and the window the
        viewer should open with.

    Raises:
        HTTPException: 400 if the frame index is out of range; 404 if the image
            is unknown.
    """
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
    """Render a frame for display.

    These are the exact pixels the model is given, which is what makes a click
    mean the same thing to the user and to SAM.

    Args:
        image_id: Image identifier.
        frame: Frame index within it.
        wc: Window centre. Omit both this and ``ww`` for the file's own window.
        ww: Window width.

    Returns:
        Response: An 8-bit PNG, uncached because the window can change.

    Raises:
        HTTPException: 400 if the frame index is out of range; 404 if the image
            is unknown.
    """
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
    """Segment for the prompts currently being placed, without committing.

    This is the request behind the ~10 ms click: on a cached embedding only the
    mask decoder runs.

    Args:
        req: Image, frame, prompts, window, threshold, candidate index and any
            brush strokes.
        color: Hex fill colour for the overlay.
        alpha: Fill opacity, 0-255.

    Returns:
        Response: An RGBA PNG overlay. The mask's score, pixel area and
        candidate count come back in the ``X-Mask-Score``, ``X-Mask-Area`` and
        ``X-Mask-Candidates`` headers, so the UI needs no second request.

    Raises:
        HTTPException: 400 if there is nothing to act on or the frame index is
            out of range; 404 if the image is unknown.
    """
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
    """Commit the current mask as a labelled instance.

    Reusing a label adds another instance in the same colour; the colour lives
    on the workspace, so it is the same in every file of the folder.

    Args:
        req: Image, frame, label, prompts, window, threshold, candidate index
            and any brush strokes.

    Returns:
        AnnotationOut: The stored annotation, including its instance number and
        colour.

    Raises:
        HTTPException: 400 if the label is empty or the request has nothing to
            act on; 404 if the image is unknown; 422 if the prompts produce an
            empty mask.
    """
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
    """List an image's annotations.

    Args:
        image_id: Image identifier.
        frame: Restrict to one frame. Omit for every frame of the file.

    Returns:
        list[AnnotationOut]: In creation order.

    Raises:
        HTTPException: 404 if the image is unknown.
    """
    rec = _rec(image_id)
    anns = [a for a in rec.annotations.values() if frame is None or a.frame == frame]
    return [_out(rec, a) for a in anns]


@app.patch("/annotations/{ann_id}", response_model=AnnotationOut)
def update_annotation(ann_id: str, req: AnnotationUpdate):
    """Edit an annotation: its prompts, strokes, label, window or threshold.

    The mask is only recomputed if something that shapes it actually changed,
    and the recompute reuses the cached image embedding. Passing ``strokes``
    replaces the list wholesale, which is how undo is expressed.

    Args:
        ann_id: Annotation identifier.
        req: The fields to change; unset ones are left alone.

    Returns:
        AnnotationOut: The updated annotation.

    Raises:
        HTTPException: 404 if the annotation is unknown; 422 if the edit
            produces an empty mask, in which case the previous mask is kept.
    """
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
    """Delete one annotation.

    Args:
        ann_id: Annotation identifier.

    Returns:
        The deleted id.

    Raises:
        HTTPException: 404 if the annotation is unknown.
    """
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
    """Composite every committed mask on a frame, in per-label colours.

    Args:
        image_id: Image identifier.
        frame: Frame index within it.
        selected: Draw this annotation with a thicker white outline.
        exclude: Drop one annotation: while it is being edited its live preview
            is drawn instead, and showing both would leave the old mask peeking
            out.
        alpha: Fill opacity, 0-255.

    Returns:
        Response: An RGBA PNG, uncached because it changes on every edit.

    Raises:
        HTTPException: 400 if the frame index is out of range; 404 if the image
            is unknown.
    """
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
    """Get one mask on its own.

    Args:
        ann_id: Annotation identifier.

    Returns:
        Response: An 8-bit PNG whose pixels are 0 or 255.

    Raises:
        HTTPException: 404 if the annotation is unknown.
    """
    try:
        _, ann = store.find_annotation(ann_id)
    except KeyError:
        raise HTTPException(404, f"Unknown annotation {ann_id!r}")
    return Response(mask_to_png(rle_to_mask(ann.rle)), media_type="image/png")


@app.get("/labels")
def labels(image_id: Optional[str] = None, workspace_id: Optional[str] = None):
    """Get the label vocabulary and its colours.

    Args:
        image_id: Summarise one file.
        workspace_id: Summarise the whole folder. Takes precedence.

    Returns:
        ``{"labels": [{"name", "color", "count"}]}``, sorted by name.

    Raises:
        HTTPException: 400 if neither argument was given; 404 if the image or
            workspace is unknown.
    """
    if workspace_id:
        return {"labels": _ws(workspace_id).label_summary()}
    if image_id:
        return {"labels": _rec(image_id).label_summary()}
    raise HTTPException(400, "Pass image_id or workspace_id.")


# ---- export ------------------------------------------------------------

def _records(image_id: Optional[str], workspace_id: Optional[str]) -> tuple[Workspace, list[ImageRecord]]:
    """Resolve what an export request covers.

    Args:
        image_id: Export one file.
        workspace_id: Export the whole folder. Takes precedence.

    Returns:
        ``(workspace, records)``.

    Raises:
        HTTPException: 400 if neither argument was given; 404 if either is
            unknown.
    """
    if workspace_id:
        ws = _ws(workspace_id)
        return ws, ws.records
    if image_id:
        rec = _rec(image_id)
        return _ws(rec.workspace_id), [rec]
    raise HTTPException(400, "Pass image_id or workspace_id.")


def _ann_dict(rec: ImageRecord, a: Annotation, include_masks: bool) -> dict:
    """Render one annotation for the export.

    Args:
        rec: The image it belongs to, consulted for the label's colour.
        a: The annotation.
        include_masks: Whether to embed the RLE mask. False gives a
            metadata-only export.

    Returns:
        The annotation's export entry. Prompts, strokes, window, threshold and
        candidate index are always present, so the mask can be re-derived rather
        than only read.
    """
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
    """Render one image for the export.

    Args:
        rec: The image.
        include_masks: Whether to embed RLE masks.

    Returns:
        Identity, geometry and metadata, plus the annotations sorted by frame,
        label and instance.
    """
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
    """Build the schema 2.0 export document.

    Every image appears, including ones with no annotations -- the export
    records what was looked at and found empty, not only what was marked.

    Args:
        ws: The workspace being exported.
        records: The images to include.
        include_masks: Whether to embed RLE masks.

    Returns:
        The document, described in ``docs/api.md``.
    """
    from store import _summarise

    return {
        "schema_version": "2.0",
        "generator": "promptseg",
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
    """Export every annotation in one document.

    The whole point of labelling a folder is not having to export file by file.

    Args:
        image_id: Export one file.
        workspace_id: Export the whole folder. Takes precedence.
        include_masks: Set false for a metadata-only export.

    Returns:
        The schema 2.0 document; masks are column-major COCO uncompressed RLE.

    Raises:
        HTTPException: 400 if neither identifier was given; 404 if either is
            unknown.
    """
    ws, records = _records(image_id, workspace_id)
    return _export_doc(ws, records, include_masks)


def _safe(name: str) -> str:
    """Make a filename safe to write into a zip.

    Args:
        name: A filename or label, possibly holding a path or punctuation.

    Returns:
        The basename with anything outside ``A-Za-z0-9._-`` replaced by
        underscores, never empty.
    """
    stem = posixpath.basename(name) or "image"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "image"


@app.get("/export.zip")
def export_zip(image_id: Optional[str] = None, workspace_id: Optional[str] = None):
    """Export annotations.json plus one PNG per mask, for tools that want pixels.

    Args:
        image_id: Export one file.
        workspace_id: Export the whole folder. Takes precedence.

    Returns:
        Response: A zip holding ``annotations.json`` and
        ``masks/<index>_<file>/f<frame>_<label>_<instance>.png``, offered as a
        timestamped download.

    Raises:
        HTTPException: 400 if neither identifier was given; 404 if either is
            unknown.
    """
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
        "Content-Disposition": f'attachment; filename="promptseg-{stamp}.zip"',
    })
