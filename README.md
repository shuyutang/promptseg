# sam2web

Browser-based interactive segmentation of medical images with SAM 2.1.

Open a **folder** of DICOM/PNG/JPEG/TIFF, work through the file list one image at
a time, click on the anatomy, and get a mask back in ~10 ms. Masks are saved as
**labelled instances** — one label can have many instances, all sharing a colour
across the whole folder — and every one stays editable: refine the prompts, or
paint the mask directly with a brush. Export is **one JSON document for the
entire folder**, not one per file.

![screenshot](docs/screenshot.png)

## Quick start

```bash
uv venv --python 3.12
uv pip install torch torchvision --torch-backend=cu128
uv pip install -r backend/requirements.txt

cd backend && ../.venv/bin/python -m uvicorn app:app --port 8000
# open http://localhost:8000
```

The frontend ships pre-built in `backend/static/`, so **Node is not needed to run
it** — only to change it (see [Frontend](#frontend)).

Or with Docker (weights persist in `~/.cache/sam2web-weights`):

```bash
docker build -t sam2web-backend backend/
./backend/start_server.sh
```

## Sample data

```bash
.venv/bin/python scripts/make_samples.py     # writes ./samples/ (~18 MB)
```

Everything is derived from the public DICOMs bundled with pydicom — no PHI, no
downloads.

| Path | What it is |
| --- | --- |
| `samples/mixed_folder/` | **Start here.** 11 files: CT (128², 512², 1955×1841 CR), MR (484², 1024²), colour + palette + grayscale ultrasound, plus PNG / JPEG / 16-bit-PNG renderings of the same pixels. Press **Open folder…** and pick it. |
| `samples/ct_series/` | 10 single-frame slices whose filenames run *backwards*, to show that ordering follows `ImagePositionPatient`, not the filename. |
| `samples/ct_series.zip` | The same series as a zip — **Add files…** expands it. |
| `samples/multiframe/` | Single files holding 10 and 120 frames, for the frame slider. |

For richer clinical data, [TCIA](https://www.cancerimagingarchive.net/) publishes
whole CC-BY studies as DICOM.

## Using it

**Load** — *Open folder…* starts a new workspace from a directory (recursively);
*Add files…* appends individual files or a `.zip` to the current one. One file =
one row in the list. A multi-frame file stays one row and gets a frame slider.

**Label** — type a label, click the target, press *Add mask*. Reusing a label
adds another instance (`vertebra #1`, `#2`, …) in the same colour; adjacent
instances are told apart by their outlines. The label keeps its colour in every
file of the folder.

**Adjust** — two independent ways, and they compose:

| | |
| --- | --- |
| **Re-prompt** | Click a mask in the sidebar, add include/exclude points or a box, *Save changes*. Re-runs the decoder against the cached embedding. |
| **Brush / Erase** | Paint pixels straight into or out of the mask. Strokes are stored *alongside* the prompts rather than baked in, so the final mask is always `model(prompts)` with the strokes replayed on top — which means a hand correction **survives a later change to the prompts**. *Undo brush* drops the last stroke. |

A mask can also be drawn purely by hand: brush with no prompt at all, and no
model call happens.

**Export** — *Export all (JSON)* writes every annotation across every image in
the workspace to one file. *+ PNGs (zip)* adds one PNG per mask.

| Action | Input |
| --- | --- |
| Tools | `1` point · `2` exclude point · `3` box · `4` brush · `5` erase |
| Exclude point | right-click (any tool) |
| Brush size | `[` / `]` |
| Frame | `←` / `→` |
| Next / previous file | `n` / `p` |
| Mark file done | `d` |
| Commit / cancel | `Enter` / `Esc` |
| Delete selected mask | `Delete` |

## How it works

The image encoder is the expensive part of SAM 2 and depends only on the pixels,
so it is split from the prompt decoder and cached per `(image, frame, window)`:

| RTX 4090 | Cold (encode + decode) | Cached edit |
| --- | --- | --- |
| 128×128 CT | ~45 ms | ~9 ms |
| 600×800 US | ~55 ms | ~20 ms |
| 1955×1841 CR | ~198 ms | ~117 ms |

The encoder always runs at 1024×1024, so cold cost is near-constant; what grows
with image size is upscaling the mask logits back to native resolution and
RLE-encoding them. Large radiographs stay usable but stop feeling instant.

The window/level setting is part of the cache key because re-windowing changes
the pixels the encoder sees. It is also stored on every annotation, so an edit
reproduces the exact image the mask was originally drawn on.

**The model sees exactly the 8-bit windowed frame the browser displays**, so a
click means the same thing to the user and to SAM.

### Structure

```
workspace  ──  one folder-load; owns the label vocabulary and its colours
  └── image ──  one file; owns its own geometry and instance numbering
        └── frame ── multi-frame DICOM / multi-page TIFF only
              └── annotation ── label + instance + prompts + strokes + RLE mask
```

Geometry is per **image**, never per workspace: a picked folder routinely mixes
64², 512² and 1955×1841 files, and assuming one size for the batch silently sends
clicks to the wrong coordinates.

### Frontend

React 19 + TypeScript, built with Vite into `backend/static/`. Four stacked
layers the browser composites — base frame PNG, committed-mask overlay PNG, live
preview PNG, and a canvas holding only the prompt markers and the in-flight brush
stroke. All mask rendering is server-side; the client never touches mask pixels.

```bash
cd frontend
npm install
npm run build      # -> ../backend/static
npm run dev        # hot reload on :5173, proxying the API to :8000
```

### Configuration

| Env var | Default | Purpose |
| --- | --- | --- |
| `SAM2_MODEL_ID` | `facebook/sam2.1-hiera-base-plus` | Any SAM2-compatible checkpoint — try `facebook/sam2.1-hiera-small` for speed, or `wanglab/MedSAM2` for a medical fine-tune |
| `SAM2_DEVICE` | `cuda` | `cpu` also works, slowly |
| `SAM2_STUB` | `0` | `1` runs the whole API with a prompt-responsive fake model — no weights, no GPU |
| `SAM2_MAX_WORKSPACES` | `4` | LRU cap on resident workspaces. Eviction is per workspace, never per image, so a folder cannot lose slices mid-annotation |
| `SAM2_MAX_FILES` | `500` | Cap on one workspace, so a mis-picked folder cannot exhaust RAM |
| `SAM2_MAX_EMBEDDINGS` | `24` | LRU cap on cached embeddings (~17 MB each) |

## Export format

`GET /export.json?workspace_id=…` (or `?image_id=…` for one file):

```jsonc
{
  "schema_version": "2.0",
  "model": { "id": "facebook/sam2.1-hiera-base-plus",
             "mask_composition": "model(prompts) then strokes replayed in order" },
  "workspace": { "name": "mixed_folder", "image_count": 11, "annotation_count": 3 },
  "labels": [ { "name": "lung", "color": "#F97316", "count": 2 } ],
  "images": [
    {
      "filename": "mixed_folder/001_ct_chest_128.dcm",
      "modality": "CT", "rows": 128, "columns": 128, "frames": 1,
      "series_instance_uid": "…", "pixel_spacing": [0.66, 0.66],
      "reviewed": true,
      "annotations": [
        {
          "id": "…", "label": "lung", "instance": 1, "color": "#F97316",
          "frame": 0, "area": 2832, "bbox": [x, y, w, h], "score": 0.55,
          "prompts": { "points": [ { "x": 40, "y": 60, "label": 1 } ], "boxes": [] },
          "strokes": [ { "mode": "add", "radius": 8, "points": [[20,20],[26,23]] } ],
          "window": { "center": -88.0, "width": 1523.0 },
          "threshold": 0.5, "mask_index": 0,
          "mask": { "format": "coco_rle_uncompressed", "order": "fortran",
                    "size": [128, 128], "counts": [ … ] }
        }
      ]
    }
  ]
}
```

Every image appears, including ones with no annotations — the export records
what was *looked at* and found empty, not only what was marked. Masks are
column-major COCO uncompressed RLE; `backend/utils/rle.py::rle_to_mask` decodes
them and the test suite asserts the round-trip. Pass `include_masks=false` for a
metadata-only export.

Because `prompts`, `strokes` and `window` are all exported, an annotation is
reproducible, not just readable.

## API

| Method | Path | |
| --- | --- | --- |
| `POST` | `/upload` | many `files` parts (+ optional `workspace_id`) → one row per image; zips are expanded |
| `GET` | `/workspaces/{id}` | file list with per-file annotation counts and progress |
| `PATCH` | `/images/{id}` | mark a file done |
| `DELETE` | `/images/{id}` | drop a file from the workspace |
| `GET` | `/frame_info` | per-image geometry, windowing support, default window |
| `GET` | `/frame.png` | `image_id`, `frame`, optional `wc`/`ww` |
| `POST` | `/segment/preview.png` | uncommitted mask for the current prompts + strokes |
| `POST` | `/annotations` | commit a labelled instance |
| `GET` | `/annotations` | list, optionally filtered by `frame` |
| `PATCH` | `/annotations/{id}` | edit prompts / strokes / label / window / threshold |
| `DELETE` | `/annotations/{id}` | |
| `GET` | `/annotations/overlay.png` | all masks on a frame, composited (`exclude=` one being edited) |
| `GET` | `/export.json` | whole workspace, or one image |
| `GET` | `/export.zip` | the same JSON plus one PNG per mask |

## Tests

Fixtures are the public DICOM files bundled with pydicom plus synthesised
PNG/JPEG/TIFF — no downloads, no PHI.

```bash
cd backend
SAM2_STUB=1 ../.venv/bin/python -m pytest tests -q          # 63 tests, no GPU
SAM2_STUB=0 ../.venv/bin/python -m pytest tests -q          # 68 incl. real-model tests
```

## Known limits

- **State is in memory.** Restarting the server drops every workspace and
  annotation. Export before you stop it.
- **Single user.** No accounts, no locking; `allow_origins=["*"]`. Run it on
  localhost or behind something that does authentication.
- **Zero-shot quality on grayscale is uneven.** SAM 2.1 is trained on natural
  images; a click on homogeneous soft tissue can flood a whole body region.
  Negative points, box prompts and the brush all help. `wanglab/MedSAM2` is the
  first thing to try if masks are poor.
- **2D only.** No propagation across slices yet — the obvious next step, since
  SAM 2's video memory is built for exactly that.
- **No DICOM-SEG output.** JSON and PNG only.
