# sam2web

Browser-based interactive segmentation of DICOM images with SAM 2.1.

Upload a DICOM (single file, multi-frame, or a zipped series), click on the
anatomy, and get a mask back in ~10 ms. Masks are saved as **labelled
instances** — one label can have many instances, all sharing a colour — and each
one stays editable, because its prompts are stored alongside the mask rather
than thrown away. Export is a single JSON document.

![screenshot](docs/screenshot.png)

## Quick start

```bash
uv venv --python 3.12
uv pip install torch torchvision --torch-backend=cu128
uv pip install -r backend/requirements.txt

cd backend && ../.venv/bin/python -m uvicorn app:app --port 8000
# open http://localhost:8000
```

Or with Docker (weights persist in `~/.cache/sam2web-weights`):

```bash
docker build -t sam2web-backend backend/
./backend/start_server.sh
```

## Using it

| Action | Input |
| --- | --- |
| Include point | left-click |
| Exclude point | right-click, or shift-click |
| Box prompt | drag |
| Cycle SAM's 3 candidates | `c` / **Next candidate** |
| Commit or update a mask | `Enter` / **Add mask** |
| Clear prompts, deselect | `Esc` |
| Change frame | `←` `→` |

Type a label before committing. Re-using a label adds another **instance** of
it (`vertebra #1`, `vertebra #2`, …) in the same colour; instances are told
apart by their outlines. Click a mask in the sidebar to reload its prompts and
refine it — editing re-runs segmentation from the adjusted prompts, so the mask
is regenerated rather than patched.

## How it works

The image encoder is the expensive part of SAM 2 and depends only on the pixels,
so it is split from the prompt decoder and cached per `(image, frame, window)`:

| | RTX 4090, 128×128 CT |
| --- | --- |
| Cold (encode + decode) | ~45 ms |
| Cached edit (decode only) | ~9 ms |

The window/level setting is part of the cache key because re-windowing changes
the pixels the encoder sees. It is also stored on every annotation, so an edit
reproduces the exact image the mask was originally drawn on.

**The model sees exactly the 8-bit windowed frame the browser displays**, so a
click means the same thing to the user and to SAM.

### Configuration

| Env var | Default | Purpose |
| --- | --- | --- |
| `SAM2_MODEL_ID` | `facebook/sam2.1-hiera-base-plus` | Any SAM2-compatible checkpoint — try `facebook/sam2.1-hiera-small` for speed, or `wanglab/MedSAM2` for a medical fine-tune |
| `SAM2_DEVICE` | `cuda` | `cpu` also works, slowly |
| `SAM2_STUB` | `0` | `1` runs the whole API with a prompt-responsive fake model — no weights, no GPU |
| `SAM2_MAX_IMAGES` | `8` | LRU cap on resident studies |
| `SAM2_MAX_EMBEDDINGS` | `24` | LRU cap on cached embeddings (~17 MB each) |

## Export format

`GET /export.json?image_id=…` returns:

```jsonc
{
  "schema_version": "1.0",
  "image":  { "modality": "CT", "rows": 128, "columns": 128,
              "series_instance_uid": "…", "pixel_spacing": [0.66, 0.66] },
  "labels": [ { "name": "lung", "color": "#F97316", "count": 2 } ],
  "annotations": [
    {
      "id": "…", "label": "lung", "instance": 1, "color": "#F97316",
      "frame": 0, "area": 2303, "bbox": [x, y, w, h], "score": 0.477,
      "prompts": { "points": [ { "x": 40, "y": 60, "label": 1 } ], "boxes": [] },
      "window": { "center": -88.0, "width": 1523.0 },
      "threshold": 0.5, "mask_index": 0,
      "mask": { "format": "coco_rle_uncompressed", "order": "fortran",
                "size": [128, 128], "counts": [ … ] }
    }
  ]
}
```

Masks are column-major COCO uncompressed RLE. `backend/utils/rle.py::rle_to_mask`
decodes them; the test suite asserts the round-trip. Pass `include_masks=false`
for a metadata-only export.

Because `prompts` and `window` are exported too, an annotation is reproducible,
not just readable.

## API

| Method | Path | |
| --- | --- | --- |
| `POST` | `/dicom/upload` | `.dcm` or `.zip` → `image_id`, frame count, metadata, default window |
| `GET` | `/frame.png` | `image_id`, `frame`, optional `wc`/`ww` |
| `POST` | `/segment/preview.png` | uncommitted mask for the current prompts |
| `POST` | `/annotations` | commit a labelled instance |
| `GET` | `/annotations` | list, optionally filtered by `frame` |
| `PATCH` | `/annotations/{id}` | edit prompts / label / window / threshold |
| `DELETE` | `/annotations/{id}` | |
| `GET` | `/annotations/overlay.png` | all masks on a frame, composited |
| `GET` | `/export.json` | full annotation document |

## Tests

Fixtures are the public DICOM files bundled with pydicom — CT, MR, multi-frame
MR, and JPEG2000 colour ultrasound. No downloads, no PHI.

```bash
cd backend
SAM2_STUB=1 ../.venv/bin/python -m pytest tests -q          # 31 tests, no GPU
SAM2_STUB=0 ../.venv/bin/python -m pytest tests -q          # + 5 real-model tests
```

## Known limits

- **State is in memory.** Restarting the server drops every study and annotation.
  Export before you stop it.
- **Single user.** No accounts, no locking; `allow_origins=["*"]`. Run it on
  localhost or behind something that does authentication.
- **Zero-shot quality on grayscale is uneven.** SAM 2.1 is trained on natural
  images; a click on homogeneous soft tissue can flood a whole body region.
  Negative points and box prompts help. `wanglab/MedSAM2` is the first thing to
  try if masks are poor.
- **2D only.** No propagation across slices yet — the obvious next step, since
  SAM 2's video memory is built for exactly that.
- **No DICOM-SEG output.** JSON only.
