# Architecture

[← README](../README.md) · [Setup](setup.md) · [Usage](usage.md) · [API & export](api.md)

## The encoder/decoder split

The image encoder is the expensive part of SAM 2 and depends only on the pixels,
so it is split from the prompt decoder and cached per `(image, frame, window)`:

Measured end to end on an RTX 4090 — one `POST /segment/preview.png`, model plus
mask rendering plus PNG encoding:

| | Cold (encode + decode) | Cached edit |
| --- | --- | --- |
| 512² CT, DICOM | ~51 ms | ~13 ms |
| 1024² MR, DICOM | ~67 ms | ~28 ms |
| 1841×1955 CR, DICOM | ~184 ms | ~131 ms |
| 4928×3264 photo, JPEG (16 MP) | ~313 ms | ~269 ms |

The encoder always runs at 1024×1024, so cold cost is near-constant in image
size; what grows is upscaling the mask logits back to native resolution, then
RLE- and PNG-encoding them. By 16 megapixels that rendering *is* the cost — the
gap between cold and cached has almost closed, and the split below stops buying
much. Large radiographs and full-resolution camera images stay usable but stop
feeling instant; downscaling before annotating is the fix if it matters.

This split is the whole interactivity budget. Anything that re-encodes on every
click gives it away.

## Rules worth not breaking

**Window/level is part of the cache key.** Re-windowing changes the pixels the
encoder sees, so reusing an embedding across windows would silently return masks
for a different-looking image. The window is also stored on every annotation, so
an edit reproduces the exact image the mask was drawn on.

**The model sees exactly the 8-bit windowed frame the browser displays**, so a
click means the same thing to the user and to SAM.

**Geometry is per image, never per workspace.** A picked folder routinely mixes
64², 512² and 1955×1841 files; assuming one size for the batch sends clicks to
the wrong coordinates with no error at all.

**Brush strokes sit beside the prompts, not inside the mask.** The final mask is
always `model(prompts)` with strokes replayed in order — that is what lets a hand
correction survive re-prompting, and what makes an exported annotation
reproducible rather than merely readable.

**Eviction is per workspace, never per image**, so a folder cannot lose files
mid-annotation.

## Structure

```
workspace  ──  one folder-load; owns the label vocabulary and its colours
  └── image ──  one file; owns its own geometry and instance numbering
        └── frame ── multi-frame DICOM / multi-page TIFF only
              └── annotation ── label + instance + prompts + strokes + RLE mask
```

```
backend/
  app.py            FastAPI routes
  store.py          workspaces, images, annotations (the resident working copy)
  persistence.py    the copy on disk: SQLite + content-addressed image blobs
  media/            format handling: base.py, dicom_source.py, raster_source.py, loader.py
  dicom/io.py       pixel decoding, windowing, palette/MONOCHROME1 handling
  models/           the SAM2 runner and its embedding cache
  utils/            paint.py (brush geometry), rle.py, images.py (PNG + overlay rendering)
  labels.py         label vocabulary and colour assignment
  static/           the built frontend (committed)
frontend/src/       React 19 + TypeScript sources
```

## Frontend

React 19 + TypeScript, built with Vite into `backend/static/`. Four stacked
layers the browser composites — base frame PNG, committed-mask overlay PNG, live
preview PNG, and a canvas holding only the prompt markers and the in-flight brush
stroke. All mask rendering is server-side; the client never touches mask pixels.

Rebuilding is only needed if you are changing the UI. Node 20+:

```bash
cd frontend
npm install
npm run dev        # hot reload on :5173, proxying the API to a backend on :8000
npm run build      # -> ../backend/static, which is committed
npm run typecheck
```

`backend/static/` is checked in deliberately, so a fresh clone runs without
Node — which also means a UI change is not live until you rebuild and commit the
output alongside it.

## Tests

Fixtures are the public DICOM files bundled with pydicom plus synthesised
PNG/JPEG/TIFF — no downloads, no PHI.

```bash
cd backend
SAM2_STUB=1 ../.venv/bin/python -m pytest tests -q     # 73 passed, 5 skipped — no GPU, ~1.5 s
SAM2_STUB=0 ../.venv/bin/python -m pytest tests -q     # 78 passed — adds the real-model ones
```

The five skipped ones (`tests/test_sam2_gpu.py`) assert what only a real
checkpoint can show: plausible masks, candidates ranked by score, a negative
point shrinking the mask, box prompts, and the cached edit actually being faster
than the cold one. Everything else — loading, ordering, brush geometry, colours,
export round-trips — runs against the stub. The persistence tests restart the
server for real, by reloading the application module against a temporary data
directory, so nothing in memory survives into the assertion.

## Known limits

- **Saved sessions include the images.** Reopening one has to put the pixels
  back, so the original files are copied into `SAM2_DATA_DIR` in the clear —
  DICOM included. `SAM2_PERSIST=0` restores the memory-only behaviour.
- **Single user.** No accounts, no locking; `allow_origins=["*"]`. Run it on
  localhost or behind something that does authentication.
- **Zero-shot quality on grayscale is uneven.** SAM 2.1 is trained on natural
  images, so photographs are the easy case; a click on homogeneous soft tissue
  can instead flood a whole body region. Negative points, box prompts and the
  brush all help. `wanglab/MedSAM2` is the first thing to try if masks on scans
  are poor — and worth *not* using if the folder is ordinary images.
- **2D only.** No propagation across slices yet — the obvious next step, since
  SAM 2's video memory is built for exactly that.
- **No DICOM-SEG output.** JSON and PNG only.
