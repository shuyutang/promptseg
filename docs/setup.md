# Setup

[← README](../README.md) · [Usage](usage.md) · [Architecture](architecture.md) · [API & export](api.md)

## Requirements

| | |
| --- | --- |
| Python | 3.12 (what it is developed and tested on) |
| GPU | Optional. An NVIDIA GPU is what makes clicking feel instant; with none present it falls back to CPU on its own and a click takes a few seconds |
| Disk | ~310 MB of weights, fetched from the Hugging Face Hub the first time the server starts, plus whatever the saved sessions hold — see [Saved sessions](#saved-sessions) |
| Node | **Not needed to run it** — the frontend ships pre-built in `backend/static/`. Only to change it, see [Architecture](architecture.md#frontend) |
| Browser | Chrome or Edge. Folder picking relies on `webkitdirectory`; individual files and zips work anywhere |

## Install

From the repository root, with [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12
uv pip install torch torchvision --torch-backend=cu128    # CUDA 12.8 wheels
uv pip install -r backend/requirements.txt
```

Or with stock tooling:

```bash
python3.12 -m venv .venv
.venv/bin/pip install torch torchvision      # CPU wheels; pytorch.org has the CUDA index URLs
.venv/bin/pip install -r backend/requirements.txt
```

Torch is installed on its own line in both cases because the right wheel depends
on your driver, and `requirements.txt` deliberately does not pin one.

## Run

```bash
cd backend
../.venv/bin/python -m uvicorn app:app --port 8000
```

Then open <http://localhost:8000>. **The first start is the slow one**: weights
load at import, so the server does not answer until the download finishes.
Afterwards they are cached in `~/.cache/huggingface` and startup is a few
seconds.

Check what actually came up:

```bash
curl -s localhost:8000/health
# {"ok":true,"model":"facebook/sam2.1-hiera-base-plus","device":"cuda",
#  "persist":true,"storage":{"path":"~/.local/share/promptseg","sessions":2,…}}
```

## Without a GPU

```bash
SAM2_STUB=1 ../.venv/bin/python -m uvicorn app:app --port 8000
```

Runs the entire API against a prompt-responsive fake model: no weights, no
download, no GPU. Every feature except mask quality behaves normally — useful
for looking at the UI, and it is what the test suite uses.

Real weights on CPU also work (`SAM2_DEVICE=cpu`), just slowly.

## Docker

Needs `nvidia-container-toolkit`. Weights persist in `~/.cache/promptseg-weights`,
so the image does not have to be rebuilt to change models, and saved sessions in
`~/.local/share/promptseg` on the host — the container runs with `--rm`, so a
session kept inside it would not survive the server stopping.

```bash
docker build -t promptseg-backend backend/
./backend/start_server.sh          # http://localhost:8000
```

## Sample images

```bash
.venv/bin/python scripts/make_samples.py     # writes ./samples/mixed_folder (~11 MB)
```

8 files derived from the public DICOMs bundled with pydicom — no PHI, nothing
downloaded. It deliberately mixes what a real pick throws at the app: CT, MR and
ultrasound from 128×128 to 1955×1841, colour, palette and inverted (MONOCHROME1)
images, and a PNG and a JPEG alongside the DICOMs. Each image appears once.

Press **Open folder…** and pick it.

The samples are medical because DICOM is the part with edge cases worth
exercising, not because the app needs them: **any folder of PNG, JPEG, TIFF, BMP
or WEBP works the same way** — a photo directory, a page of screenshots, a
microscopy export. Non-DICOM files simply arrive without a window to set, so the
**L**/**W** sliders do not appear. SAM 2.1 was trained on natural images, so
zero-shot quality on ordinary photos is generally *better* than on grayscale
scans.

For richer clinical data, [TCIA](https://www.cancerimagingarchive.net/)
publishes whole CC-BY studies.

## Configuration

| Env var | Default | Purpose |
| --- | --- | --- |
| `SAM2_MODEL_ID` | `facebook/sam2.1-hiera-base-plus` | Any SAM2-compatible checkpoint — try `facebook/sam2.1-hiera-small` for speed, or `wanglab/MedSAM2` for a medical fine-tune |
| `SAM2_DEVICE` | `cuda` | `cpu` also works, slowly |
| `SAM2_STUB` | `0` | `1` runs the whole API with a prompt-responsive fake model — no weights, no GPU |
| `SAM2_MAX_WORKSPACES` | `4` | LRU cap on resident workspaces. Eviction is per workspace, never per image, so a folder cannot lose slices mid-annotation |
| `SAM2_MAX_FILES` | `500` | Cap on one workspace, so a mis-picked folder cannot exhaust RAM |
| `SAM2_MAX_EMBEDDINGS` | `24` | LRU cap on cached embeddings (~17 MB each) |
| `SAM2_PERSIST` | `1` | `0` keeps every session in memory only, as older versions did: nothing is written to disk and an export is the only way to keep anything |
| `SAM2_DATA_DIR` | `~/.local/share/promptseg` | Where saved sessions live: `sessions.db` plus a `blobs/` directory of the original files |
| `SAM2_MAX_SAVED` | `20` | How many sessions to keep before the least recently worked on are deleted. `0` keeps everything |

### Saved sessions

Every mask is written to disk as it is made, so stopping the server — on purpose
or otherwise — does not cost work that was never exported. The **⟲ Saved**
button in the file list reopens a session exactly as it was left: same images,
masks, labels, colours, instance numbers and done flags.

Making that possible means the **original files are stored too**, not just the
annotations, because a mask is not much use without the pixels it was drawn on.
So a folder of DICOMs annotated here ends up copied, in the clear, under
`SAM2_DATA_DIR`. On a machine where that is not acceptable, run with
`SAM2_PERSIST=0` and export as before. Files are content-addressed, so loading
the same folder twice costs disk once, and deleting a session (the `×` beside
it) reclaims the space immediately.

`GET /health` reports what the store is doing, including a `storage.error` when
the data directory turns out not to be writable — in which case the app keeps
working and simply stops saving.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| Server takes minutes to answer on first start | It is downloading ~310 MB of weights before binding. Watch the log; later starts take seconds |
| `/health` says `"device":"cpu"` on a CUDA box | A CPU-only torch got installed. Reinstall it from the CUDA wheel index, then confirm with `python -c "import torch; print(torch.cuda.is_available())"` |
| Clicks are slow but masks look right | CPU inference, or a very large image. `SAM2_MODEL_ID=facebook/sam2.1-hiera-small` is the cheap fix |
| *Open folder…* does nothing | `webkitdirectory` is not supported by the browser. Use *Add files…* with a zip instead |
| One file missing after loading a folder | Unreadable files never fail the batch; they are listed by name under *N file(s) skipped* at the top of the file list |
| A click grabs far too much — a whole body region, a whole background | Zero-shot SAM, most often on low-contrast grayscale. Add negative points (right-click), draw a box instead, or clean up with the eraser. On DICOM, a tighter window usually helps more than extra points |
| Work missing after a restart | Press **⟲ Saved** in the file list and reopen the session. If the list is empty, check `curl -s localhost:8000/health` for `"persist": false` (running with `SAM2_PERSIST=0`) or a `storage.error` |
| `storage.error` in `/health` | `SAM2_DATA_DIR` is not writable, or the disk is full. Annotation still works; nothing is being saved until it is fixed |
| Saved sessions taking too much disk | They hold the original images. Delete the ones you are done with (`×` in the **⟲ Saved** list), or lower `SAM2_MAX_SAVED` |
| `Address already in use` | Another copy is running: `pkill -f "uvicorn app:app"`, or pass a different `--port` |
