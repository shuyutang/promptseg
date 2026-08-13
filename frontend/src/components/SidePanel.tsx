import type { Annotation, Draft, LabelSummary } from '../types'
import { draftIsEmpty } from '../types'

type Props = {
  draft: Draft
  setDraft: (d: Draft) => void
  labels: LabelSummary[]
  annotations: Annotation[]
  frame: number
  frames: number
  previewArea: number | null
  previewScore: number | null
  busy: boolean
  error: string | null
  onCommit: () => void
  onCancel: () => void
  onEdit: (a: Annotation) => void
  onDelete: (id: string) => void
  onRename: (a: Annotation) => void
}

export default function SidePanel(p: Props) {
  const { draft } = p
  const editing = draft.editing !== null
  const onThisFrame = p.annotations.filter((a) => a.frame === p.frame)
  const canCommit = !draftIsEmpty(draft) && draft.label.trim().length > 0

  const set = (patch: Partial<Draft>) => p.setDraft({ ...draft, ...patch })

  return (
    <aside className="panel side">
      <div className="panel-head"><h2>{editing ? 'Editing mask' : 'New mask'}</h2></div>

      <label className="field">
        Label
        <input list="known-labels" value={draft.label} placeholder="liver, lesion, …"
               onChange={(e) => set({ label: e.currentTarget.value })} />
        <datalist id="known-labels">
          {p.labels.map((l) => <option key={l.name} value={l.name} />)}
        </datalist>
      </label>

      {p.labels.length > 0 && (
        <div className="chips">
          {p.labels.map((l) => (
            <button key={l.name} className="chip" onClick={() => set({ label: l.name })}
                    title={`${l.count} mask(s) in this workspace`}>
              <i style={{ background: l.color }} />{l.name}
            </button>
          ))}
        </div>
      )}

      <div className="row gap">
        <label className="field grow">
          Candidate
          <div className="seg">
            {[0, 1, 2].map((i) => (
              <button key={i} className={draft.maskIndex === i ? 'on' : ''}
                      title={i === 0 ? 'Best-scoring mask' : `Alternative ${i + 1}`}
                      onClick={() => set({ maskIndex: i })}>{i + 1}</button>
            ))}
          </div>
        </label>
        <label className="field grow">
          Threshold <span className="num">{draft.threshold.toFixed(2)}</span>
          <input type="range" min={0.05} max={0.95} step={0.05} value={draft.threshold}
                 onChange={(e) => set({ threshold: Number(e.currentTarget.value) })} />
        </label>
      </div>

      <div className="stats">
        {p.previewArea !== null
          ? <>{p.previewArea.toLocaleString()} px{p.previewScore !== null && <> · IoU {p.previewScore.toFixed(2)}</>}</>
          : <span className="muted">Click the image to start a mask.</span>}
        {draft.strokes.length > 0 && <> · {draft.strokes.length} hand edit{draft.strokes.length === 1 ? '' : 's'}</>}
      </div>

      <div className="row gap">
        <button className="primary grow" disabled={!canCommit || p.busy} onClick={p.onCommit}>
          {editing ? 'Save changes' : 'Add mask'}
        </button>
        <button onClick={p.onCancel} disabled={draftIsEmpty(draft) && !editing}>Cancel</button>
      </div>
      <div className="row gap">
        <button className="ghost" disabled={!draft.strokes.length}
                onClick={() => set({ strokes: draft.strokes.slice(0, -1) })}>Undo brush</button>
        <button className="ghost" disabled={!draft.points.length && !draft.boxes.length}
                onClick={() => set({ points: [], boxes: [] })}>Clear prompts</button>
      </div>

      {p.error && <p className="error">{p.error}</p>}

      <div className="panel-head">
        <h2>Masks{p.frames > 1 && <span className="muted"> · frame {p.frame + 1}</span>}</h2>
        <span className="muted">{onThisFrame.length}</span>
      </div>
      <div className="list">
        {onThisFrame.map((a) => (
          <div key={a.id} className={`ann ${draft.editing === a.id ? 'active' : ''}`}
               onClick={() => p.onEdit(a)}>
            <i className="dot" style={{ background: a.color }} />
            <span className="name">{a.label} <span className="muted">#{a.instance}</span></span>
            <span className="muted small">{a.area.toLocaleString()} px</span>
            <button className="x" title="Rename"
                    onClick={(e) => { e.stopPropagation(); p.onRename(a) }}>✎</button>
            <button className="x" title="Delete"
                    onClick={(e) => { e.stopPropagation(); p.onDelete(a.id) }}>×</button>
          </div>
        ))}
        {!onThisFrame.length && <p className="hint pad">No masks on this frame yet.</p>}
      </div>

      <details className="help">
        <summary>Shortcuts</summary>
        <ul>
          <li><b>1–5</b> tools · <b>right-click</b> excludes</li>
          <li><b>[</b> / <b>]</b> brush size</li>
          <li><b>←</b> / <b>→</b> frame · <b>n</b> / <b>p</b> next / previous file</li>
          <li><b>Enter</b> commit · <b>Esc</b> cancel · <b>d</b> mark file done</li>
        </ul>
      </details>
    </aside>
  )
}
