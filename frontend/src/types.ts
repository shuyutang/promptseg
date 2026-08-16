/**
 * @fileoverview Shapes shared between the client and the API.
 *
 * These mirror the pydantic models in `backend/schemas.py`; the field names are
 * the API's, in snake_case, so responses can be used without translation. Only
 * `Draft` and `Tool` are the client's own.
 */

/** One click, in native image pixels. `label` 1 includes, 0 excludes. */
export type Point = { x: number; y: number; label: 1 | 0 }

/** A drawn box as `[x1, y1, x2, y2]`, in native image pixels. */
export type Box = [number, number, number, number]

/** One brush drag: how to apply it, how wide, and the path in image pixels. */
export type Stroke = { mode: 'add' | 'erase'; radius: number; points: [number, number][] }

/** Display window (level and width). Also fixes what the model sees. */
export type Window = { center: number; width: number }

/** What the user pointed at. */
export type Prompts = { points: Point[]; boxes: Box[] }

/**
 * A committed mask, as the API returns it.
 *
 * Carries everything that produced it -- prompts, strokes, window, threshold,
 * candidate index -- so an edit resumes from exactly the state it was made in.
 */
export type Annotation = {
  id: string
  image_id: string
  frame: number
  label: string
  /** 1-based, per image and label: `vertebra #1`, `#2`, … */
  instance: number
  /** Hex colour of the label, assigned across the whole workspace. */
  color: string
  /** Foreground pixel count. */
  area: number
  /** `[x, y, width, height]`, or null for an empty mask. */
  bbox: [number, number, number, number] | null
  prompts: Prompts
  window: Window | null
  threshold: number
  /** Which ranked candidate was taken; 0 is the best-scoring one. */
  mask_index: number
  strokes: Stroke[]
  /** The model's predicted IoU, or null for a hand-drawn mask. */
  score: number | null
  created_at: string
  updated_at: string
}

/** One row of the file list. Geometry is per file, never per workspace. */
export type ImageListing = {
  image_id: string
  /** Path within the picked folder, as uploaded. */
  filename: string
  /** Position in the file list, 0-based. */
  index: number
  kind: 'dicom' | 'raster'
  frames: number
  rows: number
  columns: number
  modality: string
  /** False when window/level would do nothing, e.g. an 8-bit colour frame. */
  windowing: boolean
  annotation_count: number
  labels: string[]
  /** Whether the user marked this file done. */
  reviewed: boolean
}

/** One label and how often it is used, with the colour it is drawn in. */
export type LabelSummary = { name: string; color: string; count: number }

/** One folder-load: the file list, the label vocabulary and the totals. */
export type WorkspaceListing = {
  workspace_id: string
  name: string
  created_at: string
  image_count: number
  annotation_count: number
  labels: LabelSummary[]
  images: ImageListing[]
}

/**
 * One session saved on the server, as the resume list shows it.
 *
 * Enough to recognise a session without opening it: what folder it was, when it
 * was last worked on, how far it got and what was being labelled.
 */
export type SessionSummary = {
  workspace_id: string
  name: string
  created_at: string
  /** UTC ISO-8601; when the session was last annotated, not when it was made. */
  updated_at: string
  image_count: number
  annotation_count: number
  /** The distinct labels used anywhere in the session. */
  labels: string[]
}

/** The result of an upload. `errors` names files that could not be read. */
export type UploadResult = {
  workspace_id: string
  added: number
  errors: string[]
  images: ImageListing[]
  workspace: WorkspaceListing
}

/** One frame's geometry and the window the viewer should open with. */
export type FrameInfo = {
  frame: number
  rows: number
  columns: number
  windowing: boolean
  default_window: Window
}

/** An uncommitted mask: an object URL for the overlay, plus its measurements. */
export type PreviewResult = { url: string; area: number; score: number | null }

/** What the user is building right now: either a new mask or an edit of one. */
export type Draft = {
  editing: string | null       // annotation id when editing an existing mask
  label: string
  points: Point[]
  boxes: Box[]
  strokes: Stroke[]
  threshold: number
  maskIndex: number
}

/** The selected pointer tool. Right-click always excludes, whichever is active. */
export type Tool = 'include' | 'exclude' | 'box' | 'brush' | 'eraser'

/**
 * Starts a fresh draft.
 *
 * @param label Label to keep; the next instance usually shares it.
 * @return An empty draft at the default threshold and best candidate.
 */
export const emptyDraft = (label = ''): Draft => ({
  editing: null, label, points: [], boxes: [], strokes: [],
  threshold: 0.5, maskIndex: 0,
})

/**
 * Reports whether a draft has anything to segment or paint.
 *
 * @param d The draft to test.
 * @return True when it holds no points, boxes or strokes.
 */
export const draftIsEmpty = (d: Draft) =>
  d.points.length === 0 && d.boxes.length === 0 && d.strokes.length === 0
