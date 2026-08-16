/**
 * @fileoverview The resume list: sessions the server still has, ready to reopen.
 *
 * Shown where a user actually looks for their work -- on the empty screen at
 * start-up, and behind a button in the file list once something is open. Each
 * row says enough to recognise a session without opening it, because the folder
 * name alone rarely does.
 */
import type { SessionSummary } from '../types'

/** Props for {@link Sessions}. */
type Props = {
  /** The saved sessions, most recently worked on first. */
  sessions: SessionSummary[]
  /** False when the server keeps state in memory only; explains the empty list. */
  persist: boolean
  /** Workspace id of the session already open, which is shown as current. */
  currentId: string | null
  /** True while a session is being reopened; the buttons are disabled. */
  busy: boolean
  /** Called with the workspace id to reopen. */
  onOpen: (id: string) => void
  /** Called with the workspace id to delete, for good. */
  onDelete: (id: string) => void
}

/**
 * Renders a timestamp as how long ago it was.
 *
 * Wall-clock times make a user do arithmetic to answer the only question they
 * are asking, which is whether this is the session they were in.
 *
 * @param iso UTC ISO-8601 timestamp.
 * @return A short phrase such as `4m ago` or `yesterday`.
 */
function ago(iso: string) {
  const then = Date.parse(iso)
  if (!Number.isFinite(then)) return ''
  const mins = Math.max(0, Math.round((Date.now() - then) / 60000))
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hours = Math.round(mins / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.round(hours / 24)
  return days === 1 ? 'yesterday' : `${days}d ago`
}

/**
 * Renders the resume list.
 *
 * @param p Component props.
 * @return The list, or a line saying why there is nothing in it.
 */
export default function Sessions(p: Props) {
  if (!p.persist) {
    return <p className="hint pad">This server is not saving sessions
      (<code>SAM2_PERSIST=0</code>) — export before you stop it.</p>
  }
  if (!p.sessions.length) {
    return <p className="hint pad">No saved sessions yet. Everything you label is
      kept here automatically.</p>
  }

  return (
    <div className="sessions">
      {p.sessions.map((s) => (
        <div key={s.workspace_id}
             className={`session ${s.workspace_id === p.currentId ? 'active' : ''}`}>
          <button className="session-open" disabled={p.busy}
                  onClick={() => p.onOpen(s.workspace_id)}
                  title={`Reopen ${s.name} as it was left`}>
            <span className="name">{s.name || 'workspace'}</span>
            <span className="muted">
              {s.image_count} file{s.image_count === 1 ? '' : 's'} ·{' '}
              {s.annotation_count} mask{s.annotation_count === 1 ? '' : 's'} · {ago(s.updated_at)}
              {s.workspace_id === p.currentId && ' · open'}
            </span>
            {s.labels.length > 0 && (
              <span className="muted labels">{s.labels.slice(0, 6).join(', ')}
                {s.labels.length > 6 && ` +${s.labels.length - 6}`}</span>
            )}
          </button>
          <button className="x" title="Delete this saved session"
                  disabled={p.busy}
                  onClick={() => p.onDelete(s.workspace_id)}>×</button>
        </div>
      ))}
    </div>
  )
}
