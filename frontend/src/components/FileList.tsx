/**
 * @fileoverview The left panel: load a folder, work through it, export the lot.
 *
 * The list is the unit of work -- one row per file, with its own annotation
 * count and done flag -- and the footer keeps the progress and the exports
 * where they are visible the whole time.
 */
import { useRef } from 'react'
import type { ImageListing, WorkspaceListing } from '../types'

/** Props for {@link FileList}. */
type Props = {
  /** The loaded workspace, or null before anything is picked. */
  workspace: WorkspaceListing | null
  /** Image id of the open file. */
  currentId: string | null
  /** True while an upload is in flight; the pick buttons are disabled. */
  busy: boolean
  /** Per-file load errors, shown collapsed rather than as a failure. */
  errors: string[]
  /** Called with what the picker returned. */
  onPick: (files: File[]) => void
  /** Called with the image id to open. */
  onSelect: (id: string) => void
  /** Called to mark a file done, or not done. */
  onToggleReviewed: (id: string, value: boolean) => void
  /** Called to drop a file from the workspace. */
  onRemove: (id: string) => void
  /** Called to download the whole workspace as JSON. */
  onExportJson: () => void
  /** Called to download the JSON plus one PNG per mask. */
  onExportZip: () => void
}

// A folder pick only sets webkitRelativePath, which React's types don't know about.
declare module 'react' {
  interface InputHTMLAttributes<T> { webkitdirectory?: string }
}

/**
 * Shortens a path to what is worth showing in a narrow list.
 *
 * @param path Path within the picked folder.
 * @return Its last segment, or the whole path if there is only one.
 */
function shortName(path: string) {
  const parts = path.split('/')
  return parts[parts.length - 1] || path
}

/**
 * Renders the file list panel.
 *
 * @param p Component props.
 * @return The panel: pick buttons, skipped files, the list, progress and exports.
 */
export default function FileList(p: Props) {
  const filesRef = useRef<HTMLInputElement>(null)
  const folderRef = useRef<HTMLInputElement>(null)
  const ws = p.workspace
  const images = ws?.images ?? []
  const done = images.filter((i) => i.reviewed).length

  /**
   * Hands a picked selection up, then clears the input.
   *
   * @param el The file input that fired, or null.
   */
  const pick = (el: HTMLInputElement | null) => {
    const list = el?.files
    if (!list || !list.length) return
    p.onPick(Array.from(list))
    if (el) el.value = ''   // so re-picking the same folder still fires change
  }

  return (
    <aside className="panel files">
      <div className="panel-head">
        <h1>promptseg</h1>
        <span className="muted">{ws ? `${images.length} file${images.length === 1 ? '' : 's'}` : 'no files'}</span>
      </div>

      <div className="row gap">
        <button onClick={() => folderRef.current?.click()} disabled={p.busy}>Open folder…</button>
        <button onClick={() => filesRef.current?.click()} disabled={p.busy}>Add files…</button>
      </div>
      <input ref={folderRef} type="file" webkitdirectory="" multiple hidden
             onChange={(e) => pick(e.currentTarget)} />
      <input ref={filesRef} type="file" multiple hidden
             accept=".dcm,.dicom,.ima,.png,.jpg,.jpeg,.bmp,.tif,.tiff,.webp,.zip,image/*,application/zip"
             onChange={(e) => pick(e.currentTarget)} />
      <p className="hint">DICOM, PNG, JPEG, TIFF, or a .zip</p>

      {p.errors.length > 0 && (
        <details className="errors">
          <summary>{p.errors.length} file(s) skipped</summary>
          <ul>{p.errors.slice(0, 20).map((e, i) => <li key={i}>{e}</li>)}</ul>
        </details>
      )}

      <div className="list" role="listbox" aria-label="Images">
        {images.map((im: ImageListing) => (
          <div
            key={im.image_id}
            role="option"
            aria-selected={im.image_id === p.currentId}
            className={`file ${im.image_id === p.currentId ? 'active' : ''}`}
            onClick={() => p.onSelect(im.image_id)}
            title={im.filename}
          >
            <input
              type="checkbox"
              checked={im.reviewed}
              title="Mark as done"
              onClick={(e) => e.stopPropagation()}
              onChange={(e) => p.onToggleReviewed(im.image_id, e.currentTarget.checked)}
            />
            <span className="idx">{im.index + 1}</span>
            <span className="name">{shortName(im.filename)}</span>
            <span className="badges">
              {im.frames > 1 && <span className="badge">{im.frames}f</span>}
              {im.annotation_count > 0 && <span className="badge count">{im.annotation_count}</span>}
            </span>
            <button className="x" title="Remove from workspace"
                    onClick={(e) => { e.stopPropagation(); p.onRemove(im.image_id) }}>×</button>
          </div>
        ))}
        {!images.length && <p className="hint pad">Nothing loaded yet.</p>}
      </div>

      {ws && (
        <div className="panel-foot">
          <div className="progress" title="Files marked done">
            <div className="bar" style={{ width: `${images.length ? (done / images.length) * 100 : 0}%` }} />
            <span>{done}/{images.length} done · {ws.annotation_count} mask{ws.annotation_count === 1 ? '' : 's'}</span>
          </div>
          <button
            onClick={p.onExportJson}
            disabled={!ws.annotation_count}
            title="One .json file covering every image here: each mask's label, colour, prompts, brush strokes and outline."
          >
            Export annotations (JSON)
          </button>
          <button
            onClick={p.onExportZip}
            disabled={!ws.annotation_count}
            title="The same .json, plus one PNG per mask — a folder per image, files named f<frame>_<label>_<instance>.png."
          >
            Export annotations + mask PNGs (ZIP)
          </button>
        </div>
      )}
    </aside>
  )
}
