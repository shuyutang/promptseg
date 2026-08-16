# Usage

[← README](../README.md) · [Setup](setup.md) · [Architecture](architecture.md) · [API & export](api.md)

## First five minutes

1. **Open folder…** in the left panel, and pick `samples/mixed_folder/` (see
   [Setup](setup.md#sample-images) for how to generate it). All 8 files appear
   in the list, each with its own geometry — the folder mixes a 128×128 CT with
   a 1955×1841 radiograph on purpose.
2. Click the first file. Grayscale DICOM gets **L** and **W** sliders (level and
   window width) in the toolbar; drag them until what you want is visible. What
   you see is what the model is given, so this matters *before* you click.
   `reset` goes back to the file's own window. Ordinary images — PNG, JPEG,
   TIFF — have no window to set, so the sliders are simply absent and you can
   start clicking straight away.
3. Type a label in the right-hand panel — say `lung`, or `dog` — and **click
   inside the thing you want**. A preview mask appears in ~14 ms. Add points if
   it grabbed too little, right-click to say *not this*, or switch to the box
   tool and drag.
4. **Add mask** (or `Enter`) commits it as `lung #1`. Click again elsewhere for
   `lung #2` — same label, same colour, next instance.
5. Not quite right? Pick the mask in the sidebar and either add prompts, or take
   the **brush** (`4`) / **eraser** (`5`) and paint. *Save changes* keeps it.
6. Press `d` to mark the file done, `n` for the next one. The progress bar
   counts `n/8`.
7. **Export annotations (JSON)** when finished — one document for the whole
   folder.
   Do it before stopping the server: state lives in memory only.

## Load

*Open folder…* starts a new workspace from a directory (recursively);
*Add files…* appends individual files or a `.zip` to the current one. One file =
one row in the list. A multi-frame file stays one row and gets a frame slider.

DICOM, PNG, JPEG, TIFF, BMP and WEBP all load, and nothing about the workflow
below is specific to medical images — a folder of photographs behaves exactly
like a folder of slices. What DICOM adds is metadata the app uses: a window to
set, a modality badge, pixel spacing in the export, and slice ordering. Ordinary
images skip all of that and go straight to the clicking.

Format is decided by content first and extension second, because folder exports
routinely have no extension at all. Files that cannot be read are listed by name
rather than failing the batch.

Ordering is by natural filename (`img2` before `img10`), except within one
`SeriesInstanceUID`, where slice position wins — exported slices are often named
in acquisition order, which is not anatomical order.

## Label

Type a label, click the target, press *Add mask*. Reusing a label adds another
instance (`vertebra #1`, `#2`, …) in the same colour; adjacent instances are told
apart by their outlines. **The label keeps its colour in every file of the
folder**, which is the point of colours living on the workspace rather than the
image. Instance numbering restarts per file.

## Adjust

Two independent ways, and they compose:

| | |
| --- | --- |
| **Re-prompt** | Click a mask in the sidebar, add include/exclude points or a box, *Save changes*. Re-runs the decoder against the cached embedding. |
| **Brush / Erase** | Paint pixels straight into or out of the mask. Strokes are stored *alongside* the prompts rather than baked in, so the final mask is always `model(prompts)` with the strokes replayed on top — which means a hand correction **survives a later change to the prompts**. *Undo brush* drops the last stroke. |

A mask can also be drawn purely by hand: brush with no prompt at all, and no
model call happens.

The eraser removes whatever is under it, including model pixels — erasing
exactly what you painted does not restore the original mask.

## Export

*Export annotations (JSON)* writes every annotation across every image in the
workspace to one file — labels, colours, prompts, brush strokes and outlines.

*Export annotations + mask PNGs (ZIP)* adds the same masks as images: a folder
per file, each mask a PNG named `f<frame>_<label>_<instance>.png`. Use it when
the next tool wants pixels rather than run-length encoding.

Format is in [API & export](api.md#export-format).

## Keyboard

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
