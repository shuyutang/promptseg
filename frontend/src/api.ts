import type {
  Annotation, Draft, FrameInfo, ImageListing, PreviewResult, UploadResult,
  Window, WorkspaceListing,
} from './types'

async function json<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error(await detail(r))
  return r.json() as Promise<T>
}

async function detail(r: Response): Promise<string> {
  try {
    const body = await r.json()
    if (typeof body?.detail === 'string') return body.detail
    return JSON.stringify(body?.detail ?? body)
  } catch {
    return `${r.status} ${r.statusText}`
  }
}

export const health = () => fetch('/health').then(json<{ model: string; device: string }>)

/** Files arrive as many `files` parts -- one folder pick, one request. */
export async function upload(files: File[], workspaceId: string | null,
                             name?: string): Promise<UploadResult> {
  const fd = new FormData()
  for (const f of files) {
    // webkitRelativePath keeps the folder structure that the file list shows.
    fd.append('files', f, (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name)
  }
  if (workspaceId) fd.append('workspace_id', workspaceId)
  if (name) fd.append('name', name)
  return json<UploadResult>(await fetch('/upload', { method: 'POST', body: fd }))
}

export const getWorkspace = async (id: string) =>
  json<WorkspaceListing>(await fetch(`/workspaces/${id}`))

export const frameInfo = async (imageId: string, frame: number) =>
  json<FrameInfo>(await fetch(`/frame_info?image_id=${imageId}&frame=${frame}`))

export const listAnnotations = async (imageId: string) =>
  json<Annotation[]>(await fetch(`/annotations?image_id=${imageId}`))

export const setReviewed = async (imageId: string, reviewed: boolean) =>
  json<ImageListing>(await fetch(`/images/${imageId}`, {
    method: 'PATCH', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ reviewed }),
  }))

export const deleteImage = async (imageId: string) =>
  json<unknown>(await fetch(`/images/${imageId}`, { method: 'DELETE' }))

export const frameUrl = (imageId: string, frame: number, w: Window | null) =>
  `/frame.png?image_id=${imageId}&frame=${frame}` +
  (w ? `&wc=${w.center}&ww=${w.width}` : '')

export const overlayUrl = (imageId: string, frame: number, opts: {
  selected?: string | null; exclude?: string | null; alpha?: number; bust?: number
} = {}) =>
  `/annotations/overlay.png?image_id=${imageId}&frame=${frame}` +
  (opts.selected ? `&selected=${opts.selected}` : '') +
  (opts.exclude ? `&exclude=${opts.exclude}` : '') +
  `&alpha=${opts.alpha ?? 110}&v=${opts.bust ?? 0}`

const body = (imageId: string, frame: number, d: Draft, w: Window | null) => ({
  image_id: imageId,
  frame,
  prompts: { points: d.points, boxes: d.boxes },
  strokes: d.strokes,
  window: w,
  threshold: d.threshold,
  mask_index: d.maskIndex,
})

export async function preview(imageId: string, frame: number, d: Draft,
                              w: Window | null, color: string,
                              signal?: AbortSignal): Promise<PreviewResult> {
  const r = await fetch(`/segment/preview.png?color=${encodeURIComponent(color)}`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body(imageId, frame, d, w)), signal,
  })
  if (!r.ok) throw new Error(await detail(r))
  const score = r.headers.get('X-Mask-Score')
  return {
    url: URL.createObjectURL(await r.blob()),
    area: Number(r.headers.get('X-Mask-Area') ?? 0),
    score: score ? Number(score) : null,
  }
}

export async function createAnnotation(imageId: string, frame: number, d: Draft,
                                       w: Window | null): Promise<Annotation> {
  return json<Annotation>(await fetch('/annotations', {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ ...body(imageId, frame, d, w), label: d.label }),
  }))
}

export async function patchAnnotation(id: string, patch: Record<string, unknown>) {
  return json<Annotation>(await fetch(`/annotations/${id}`, {
    method: 'PATCH', headers: { 'content-type': 'application/json' },
    body: JSON.stringify(patch),
  }))
}

export const deleteAnnotation = async (id: string) =>
  json<unknown>(await fetch(`/annotations/${id}`, { method: 'DELETE' }))
