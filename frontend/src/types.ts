export type Point = { x: number; y: number; label: 1 | 0 }
export type Box = [number, number, number, number]
export type Stroke = { mode: 'add' | 'erase'; radius: number; points: [number, number][] }
export type Window = { center: number; width: number }

export type Prompts = { points: Point[]; boxes: Box[] }

export type Annotation = {
  id: string
  image_id: string
  frame: number
  label: string
  instance: number
  color: string
  area: number
  bbox: [number, number, number, number] | null
  prompts: Prompts
  window: Window | null
  threshold: number
  mask_index: number
  strokes: Stroke[]
  score: number | null
  created_at: string
  updated_at: string
}

export type ImageListing = {
  image_id: string
  filename: string
  index: number
  kind: 'dicom' | 'raster'
  frames: number
  rows: number
  columns: number
  modality: string
  windowing: boolean
  annotation_count: number
  labels: string[]
  reviewed: boolean
}

export type LabelSummary = { name: string; color: string; count: number }

export type WorkspaceListing = {
  workspace_id: string
  name: string
  created_at: string
  image_count: number
  annotation_count: number
  labels: LabelSummary[]
  images: ImageListing[]
}

export type UploadResult = {
  workspace_id: string
  added: number
  errors: string[]
  images: ImageListing[]
  workspace: WorkspaceListing
}

export type FrameInfo = {
  frame: number
  rows: number
  columns: number
  windowing: boolean
  default_window: Window
}

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

export type Tool = 'include' | 'exclude' | 'box' | 'brush' | 'eraser'

export const emptyDraft = (label = ''): Draft => ({
  editing: null, label, points: [], boxes: [], strokes: [],
  threshold: 0.5, maskIndex: 0,
})

export const draftIsEmpty = (d: Draft) =>
  d.points.length === 0 && d.boxes.length === 0 && d.strokes.length === 0
