/**
 * @fileoverview The one place the client talks to the server.
 *
 * Requests are same-origin: in production the backend serves the built app, and
 * in development Vite proxies these paths to it. Image-producing endpoints are
 * exposed as URL builders rather than fetches, so an `<img>` can load them
 * directly and the browser does the decoding.
 */
import type {
  Annotation, Draft, FrameInfo, ImageListing, PreviewResult, SessionSummary,
  UploadResult, Window, WorkspaceListing,
} from './types'

/**
 * Unwraps a JSON response, turning a failure into a readable Error.
 *
 * @param r The response.
 * @return The parsed body.
 * @throws {Error} With the server's `detail` message when the status is not ok.
 */
async function json<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error(await detail(r))
  return r.json() as Promise<T>
}

/**
 * Extracts an error message from a failed response.
 *
 * @param r The failed response.
 * @return FastAPI's `detail` string where there is one, otherwise the body or
 *     the bare status line.
 */
async function detail(r: Response): Promise<string> {
  try {
    const body = await r.json()
    if (typeof body?.detail === 'string') return body.detail
    return JSON.stringify(body?.detail ?? body)
  } catch {
    return `${r.status} ${r.statusText}`
  }
}

/** Reports which model and device the server came up with. */
export const health = () => fetch('/health').then(json<{ model: string; device: string }>)

/**
 * Uploads files. Files arrive as many `files` parts -- one folder pick, one request.
 *
 * @param files What the picker returned. A folder pick carries a relative path
 *     on each file, which becomes the name shown in the list.
 * @param workspaceId Workspace to append to, or null to start a new one.
 * @param name Display name for a new workspace.
 * @return The workspace, the rows that were added, and per-file errors.
 */
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

/**
 * Fetches a workspace: its file list, labels and progress.
 *
 * @param id Workspace id.
 * @return The listing.
 */
export const getWorkspace = async (id: string) =>
  json<WorkspaceListing>(await fetch(`/workspaces/${id}`))

/**
 * Lists the sessions the server has saved.
 *
 * @return The sessions, most recently worked on first, and whether the server
 *     is saving them at all -- `persist: false` means this build keeps state in
 *     memory only and the list will be empty.
 */
export const listSessions = async () =>
  json<{ persist: boolean; sessions: SessionSummary[] }>(await fetch('/sessions'))

/**
 * Reopens a saved session, images and masks included.
 *
 * @param id Workspace id from {@link listSessions}.
 * @return The workspace listing, plus `errors` naming any file whose saved
 *     bytes could not be read back.
 */
export const openSession = async (id: string) =>
  json<WorkspaceListing & { errors: string[] }>(
    await fetch(`/sessions/${id}/open`, { method: 'POST' }))

/**
 * Deletes a saved session from the server, for good.
 *
 * @param id Workspace id.
 * @return The server's acknowledgement.
 */
export const deleteSession = async (id: string) =>
  json<unknown>(await fetch(`/sessions/${id}`, { method: 'DELETE' }))

/**
 * Fetches one frame's geometry and default window.
 *
 * @param imageId Image id.
 * @param frame Frame index within it.
 * @return The frame info.
 */
export const frameInfo = async (imageId: string, frame: number) =>
  json<FrameInfo>(await fetch(`/frame_info?image_id=${imageId}&frame=${frame}`))

/**
 * Lists every annotation on an image, across all its frames.
 *
 * @param imageId Image id.
 * @return The annotations, in creation order.
 */
export const listAnnotations = async (imageId: string) =>
  json<Annotation[]>(await fetch(`/annotations?image_id=${imageId}`))

/**
 * Marks a file done, or not done.
 *
 * @param imageId Image id.
 * @param reviewed The new state.
 * @return The updated listing row.
 */
export const setReviewed = async (imageId: string, reviewed: boolean) =>
  json<ImageListing>(await fetch(`/images/${imageId}`, {
    method: 'PATCH', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ reviewed }),
  }))

/**
 * Drops a file from its workspace.
 *
 * @param imageId Image id.
 * @return The server's acknowledgement.
 */
export const deleteImage = async (imageId: string) =>
  json<unknown>(await fetch(`/images/${imageId}`, { method: 'DELETE' }))

/**
 * Builds the URL of a rendered frame.
 *
 * @param imageId Image id.
 * @param frame Frame index within it.
 * @param w Display window, or null for the file's own.
 * @return A URL an `<img>` can load.
 */
export const frameUrl = (imageId: string, frame: number, w: Window | null) =>
  `/frame.png?image_id=${imageId}&frame=${frame}` +
  (w ? `&wc=${w.center}&ww=${w.width}` : '')

/**
 * Builds the URL of the composited overlay of committed masks.
 *
 * @param imageId Image id.
 * @param frame Frame index within it.
 * @param opts `selected` outlines one mask; `exclude` omits the one being
 *     edited, whose live preview is drawn instead; `alpha` is the fill opacity;
 *     `bust` is a cache-busting counter bumped after every change.
 * @return A URL an `<img>` can load.
 */
export const overlayUrl = (imageId: string, frame: number, opts: {
  selected?: string | null; exclude?: string | null; alpha?: number; bust?: number
} = {}) =>
  `/annotations/overlay.png?image_id=${imageId}&frame=${frame}` +
  (opts.selected ? `&selected=${opts.selected}` : '') +
  (opts.exclude ? `&exclude=${opts.exclude}` : '') +
  `&alpha=${opts.alpha ?? 110}&v=${opts.bust ?? 0}`

/**
 * Builds the request body shared by preview and commit, so both mean the same thing.
 *
 * @param imageId Image id.
 * @param frame Frame index within it.
 * @param d The draft being worked on.
 * @param w Display window, or null for the file's own.
 * @return The payload, in the API's field names.
 */
const body = (imageId: string, frame: number, d: Draft, w: Window | null) => ({
  image_id: imageId,
  frame,
  prompts: { points: d.points, boxes: d.boxes },
  strokes: d.strokes,
  window: w,
  threshold: d.threshold,
  mask_index: d.maskIndex,
})

/**
 * Segments without committing, for the mask that follows the pointer.
 *
 * @param imageId Image id.
 * @param frame Frame index within it.
 * @param d The draft being worked on.
 * @param w Display window, or null for the file's own.
 * @param color Hex fill colour for the overlay.
 * @param signal Aborts a request the next keystroke has already replaced.
 * @return An object URL for the overlay plus the mask's area and score. The
 *     caller owns the URL and must revoke it.
 * @throws {Error} With the server's message when the request fails.
 */
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

/**
 * Commits the draft as a labelled instance.
 *
 * @param imageId Image id.
 * @param frame Frame index within it.
 * @param d The draft, whose label is used.
 * @param w Display window, or null for the file's own.
 * @return The stored annotation, with its instance number and colour.
 */
export async function createAnnotation(imageId: string, frame: number, d: Draft,
                                       w: Window | null): Promise<Annotation> {
  return json<Annotation>(await fetch('/annotations', {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ ...body(imageId, frame, d, w), label: d.label }),
  }))
}

/**
 * Edits an annotation. Fields left out are unchanged.
 *
 * @param id Annotation id.
 * @param patch Any of `label`, `prompts`, `strokes`, `window`, `threshold`,
 *     `mask_index`. Sending `strokes` replaces the list, which is how undo is
 *     expressed.
 * @return The updated annotation.
 */
export async function patchAnnotation(id: string, patch: Record<string, unknown>) {
  return json<Annotation>(await fetch(`/annotations/${id}`, {
    method: 'PATCH', headers: { 'content-type': 'application/json' },
    body: JSON.stringify(patch),
  }))
}

/**
 * Deletes an annotation.
 *
 * @param id Annotation id.
 * @return The server's acknowledgement.
 */
export const deleteAnnotation = async (id: string) =>
  json<unknown>(await fetch(`/annotations/${id}`, { method: 'DELETE' }))
