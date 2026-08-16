# API & export format

[← README](../README.md) · [Setup](setup.md) · [Usage](usage.md) · [Architecture](architecture.md)

Interactive docs are at `/docs` on the running server (FastAPI generates them).

## Endpoints

| Method | Path | |
| --- | --- | --- |
| `GET` | `/health` | model id, device, cached embeddings, resident workspaces |
| `POST` | `/upload` | many `files` parts (+ optional `workspace_id`) → one row per image; zips are expanded |
| `GET` | `/workspaces` | list |
| `POST` | `/workspaces` | create an empty one |
| `GET` | `/workspaces/{id}` | file list with per-file annotation counts and progress |
| `GET` | `/sessions` | saved sessions, most recently worked on first; `persist: false` when the server is memory-only |
| `POST` | `/sessions/{id}/open` | reopen a saved session, images and masks included; same ids as before |
| `DELETE` | `/sessions/{id}` | forget a saved session, reclaiming its image bytes |
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
| `GET` | `/annotations/{id}/mask.png` | one mask alone |
| `GET` | `/labels` | vocabulary and colours for an image or a workspace |
| `GET` | `/export.json` | whole workspace, or one image |
| `GET` | `/export.zip` | the same JSON plus one PNG per mask |

## Export format

`GET /export.json?workspace_id=…` (or `?image_id=…` for one file):

```jsonc
{
  "schema_version": "2.0",
  "model": { "id": "facebook/sam2.1-hiera-base-plus",
             "mask_composition": "model(prompts) then strokes replayed in order" },
  "workspace": { "name": "mixed_folder", "image_count": 8, "annotation_count": 3 },
  "labels": [ { "name": "lung", "color": "#F97316", "count": 2 } ],
  "images": [
    {
      "filename": "mixed_folder/001_ct_512.dcm",
      "modality": "CT", "rows": 512, "columns": 512, "frames": 1,
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
what was *looked at* and found empty, not only what was marked.

The per-image block is the same shape whatever the file was. PNG, JPEG and TIFF
carry the same keys as DICOM, with the DICOM-only ones empty — `modality: ""`,
`series_instance_uid: ""`, `pixel_spacing: []`, `window_center: null` — so a
consumer never has to branch on the source format. `window` on an annotation is
likewise recorded but inert for images that have no windowing.

Masks are column-major COCO uncompressed RLE; `backend/utils/rle.py::rle_to_mask`
decodes them and the test suite asserts the round-trip. Pass
`include_masks=false` for a metadata-only export.

Because `prompts`, `strokes`, `window`, `threshold` and `mask_index` are all
exported, an annotation is reproducible, not just readable.
