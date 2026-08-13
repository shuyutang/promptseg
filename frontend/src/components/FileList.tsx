import { useRef } from 'react'
import type { ImageListing, WorkspaceListing } from '../types'

type Props = {
  workspace: WorkspaceListing | null
  currentId: string | null
  busy: boolean
  errors: string[]
  onPick: (files: File[]) => void
  onSelect: (id: string) => void
  onToggleReviewed: (id: string, value: boolean) => void
  onRemove: (id: string) => void
  onExportJson: () => void
  onExportZip: () => void
}

// A folder pick only sets webkitRelativePath, which React's types don't know about.
declare module 'react' {
  interface InputHTMLAttributes<T> { webkitdirectory?: string }
}

function shortName(path: string) {
  const parts = path.split('/')
  return parts[parts.length - 1] || path
}

export default function FileList(p: Props) {
  const filesRef = useRef<HTMLInputElement>(null)
  const folderRef = useRef<HTMLInputElement>(null)
  const ws = p.workspace
  const images = ws?.images ?? []
  const done = images.filter((i) => i.reviewed).length

  const pick = (el: HTMLInputElement | null) => {
    const list = el?.files
    if (!list || !list.length) return
    p.onPick(Array.from(list))
    if (el) el.value = ''   // so re-picking the same folder still fires change
  }

  return (
    <aside className="panel files">
      <div className="panel-head">
        <h1>sam2web</h1>
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
      <p className="hint">DICOM, PNG, JPEG, TIFF, or a .zip — a folder loads every image inside it.</p>

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
          <div className="row gap">
            <button onClick={p.onExportJson} disabled={!ws.annotation_count}>Export all (JSON)</button>
            <button onClick={p.onExportZip} disabled={!ws.annotation_count}>+ PNGs (zip)</button>
          </div>
          <p className="hint">One file covers every image in this workspace.</p>
        </div>
      )}
    </aside>
  )
}
