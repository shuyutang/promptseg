/**
 * @fileoverview The application: all state lives here, the panels are views of it.
 *
 * Layout is fixed at three columns -- file list, image, mask panel -- and the
 * flow of a session runs left to right: pick a folder, click a thing, name what
 * you clicked.
 *
 * Two invariants worth keeping. Geometry is read per file (`FrameInfo`), never
 * assumed for the folder, because a picked directory routinely mixes sizes. And
 * mask pixels are never touched here: the server renders the base frame, the
 * overlay and the preview as PNGs, and the canvas holds only prompt markers and
 * the stroke still under the pointer.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import * as api from './api'
import FileList from './components/FileList'
import Sessions from './components/Sessions'
import SidePanel from './components/SidePanel'
import Toolbar from './components/Toolbar'
import Viewer from './components/Viewer'
import type {
  Annotation, Box, Draft, FrameInfo, PreviewResult, SessionSummary, Stroke, Tool,
  Window, WorkspaceListing,
} from './types'
import { draftIsEmpty, emptyDraft } from './types'

/** Used until the workspace has assigned the label a colour of its own. */
const FALLBACK_COLOR = '#E8453C'

/** Milliseconds to wait before previewing; enough to coalesce a burst of clicks. */
const PREVIEW_DEBOUNCE = 70

/**
 * Mask fill opacity, 0-255. Fixed rather than a control: it is a view setting
 * with one sensible value -- masks readable, the image still visible under them --
 * and the checkbox covers the case where it is in the way.
 */
const MASK_ALPHA = 110

/**
 * Renders a thrown value as text.
 *
 * @param e Anything a rejected promise produced.
 * @return Its message, or its string form.
 */
const msg = (e: unknown) => (e instanceof Error ? e.message : String(e))

/**
 * The whole application.
 *
 * @return The three-column layout, wired to one piece of state.
 */
export default function App() {
  const [ws, setWs] = useState<WorkspaceListing | null>(null)
  const [currentId, setCurrentId] = useState<string | null>(null)
  const [frame, setFrame] = useState(0)
  const [info, setInfo] = useState<FrameInfo | null>(null)
  const [win, setWin] = useState<Window | null>(null)
  const [annotations, setAnnotations] = useState<Annotation[]>([])
  const [draft, setDraft] = useState<Draft>(emptyDraft())
  const [preview, setPreview] = useState<PreviewResult | null>(null)

  const [tool, setTool] = useState<Tool>('include')
  const [brushRadius, setBrushRadius] = useState(8)
  const [zoom, setZoom] = useState(1)
  const [showMasks, setShowMasks] = useState(true)

  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [uploadErrors, setUploadErrors] = useState<string[]>([])
  const [bust, setBust] = useState(0)
  const [model, setModel] = useState('')
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [persist, setPersist] = useState(true)
  const [showSessions, setShowSessions] = useState(false)

  const stage = useRef<HTMLDivElement>(null)
  const [stageSize, setStageSize] = useState({ w: 900, h: 700 })

  const images = ws?.images ?? []
  const current = images.find((i) => i.image_id === currentId) ?? null

  // ---- data loading ---------------------------------------------------

  // Show which model answered, so a stub or a CPU fallback is never a surprise.
  useEffect(() => { api.health().then((h) => setModel(h.model)).catch(() => {}) }, [])

  /** Re-reads the saved sessions, so the resume list matches what is on disk. */
  const refreshSessions = useCallback(async () => {
    try {
      const r = await api.listSessions()
      setSessions(r.sessions)
      setPersist(r.persist)
    } catch { /* the resume list is a convenience; its absence is not an error */ }
  }, [])

  useEffect(() => { refreshSessions() }, [refreshSessions])

  // Track the stage's size, since the fit-to-window scale depends on it.
  useEffect(() => {
    const el = stage.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setStageSize({ w: e.contentRect.width, h: e.contentRect.height }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Geometry is per file and per frame, so it is re-read on every move.
  useEffect(() => {
    if (!currentId) { setInfo(null); return }
    let live = true
    api.frameInfo(currentId, frame)
      .then((i) => { if (live) setInfo(i) })
      .catch((e) => { if (live) setError(msg(e)) })
    return () => { live = false }
  }, [currentId, frame])

  /**
   * Reloads one image's annotations from the server.
   *
   * @param id Image id.
   */
  const reloadAnnotations = useCallback(async (id: string) => {
    setAnnotations(await api.listAnnotations(id))
  }, [])

  useEffect(() => {
    if (!currentId) { setAnnotations([]); return }
    reloadAnnotations(currentId).catch((e) => setError(msg(e)))
  }, [currentId, reloadAnnotations])

  /**
   * Looks up the colour the workspace assigned to a label.
   *
   * @param label Label as typed; matched case-insensitively.
   * @return Its hex colour, or the fallback for a label not used yet.
   */
  const colorFor = useCallback((label: string) => {
    const hit = ws?.labels.find((l) => l.name.toLowerCase() === label.trim().toLowerCase())
    return hit?.color ?? FALLBACK_COLOR
  }, [ws])

  // ---- live preview ---------------------------------------------------

  // Re-segment as the draft changes, debounced, cancelling whatever is in flight.
  useEffect(() => {
    if (!currentId || draftIsEmpty(draft)) { setPreview(null); return }
    const ctrl = new AbortController()
    const t = setTimeout(() => {
      api.preview(currentId, frame, draft, win, colorFor(draft.label), ctrl.signal)
        .then((r) => { setPreview(r); setError(null) })
        .catch((e) => { if (!ctrl.signal.aborted) setError(msg(e)) })
    }, PREVIEW_DEBOUNCE)
    return () => { clearTimeout(t); ctrl.abort() }
  }, [currentId, frame, draft, win, colorFor])

  // Blob URLs leak until revoked; the cleanup sees the previous value, which is
  // exactly the one to release.
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview.url) }, [preview])

  // ---- actions --------------------------------------------------------

  /**
   * Opens a file, resetting everything that belongs to the previous one.
   *
   * The label survives, because the next file usually needs the same one.
   *
   * @param id Image id to show.
   */
  const select = useCallback((id: string) => {
    setCurrentId(id)
    setFrame(0)
    setWin(null)
    setPreview(null)
    setError(null)
    setDraft((d) => emptyDraft(d.label))
  }, [])

  /**
   * Uploads what the picker returned.
   *
   * @param files The picked files.
   * @param fresh True to start a new workspace (a folder pick), false to append
   *     to the current one.
   */
  const pick = async (files: File[], fresh: boolean) => {
    setBusy(true)
    setError(null)
    try {
      const r = await api.upload(files, fresh ? null : ws?.workspace_id ?? null)
      setWs(r.workspace)
      setUploadErrors(r.errors)
      const first = fresh ? r.workspace.images[0] : r.images[0]
      if (first && (fresh || !currentId)) select(first.image_id)
    } catch (e) {
      setError(msg(e))
    } finally {
      setBusy(false)
      refreshSessions()
    }
  }

  /** Re-reads the workspace, so counts, labels and progress stay honest. */
  const refreshWorkspace = useCallback(async () => {
    if (!ws) return
    try { setWs(await api.getWorkspace(ws.workspace_id)) } catch (e) { setError(msg(e)) }
    refreshSessions()
  }, [ws, refreshSessions])

  /**
   * Reopens a saved session, putting its files and masks back on screen.
   *
   * @param id Workspace id from the resume list.
   */
  const openSession = async (id: string) => {
    setBusy(true)
    setError(null)
    try {
      const r = await api.openSession(id)
      setWs(r)
      setUploadErrors(r.errors)
      setShowSessions(false)
      if (r.images.length) select(r.images[0].image_id)
      else { setCurrentId(null); setAnnotations([]) }
    } catch (e) {
      setError(msg(e))
    } finally {
      setBusy(false)
      refreshSessions()
    }
  }

  /**
   * Deletes a saved session. Closes it first if it is the one on screen.
   *
   * @param id Workspace id.
   */
  const removeSession = async (id: string) => {
    if (!confirm('Delete this saved session? Its images and masks are removed from the server.')) return
    try {
      await api.deleteSession(id)
      if (ws?.workspace_id === id) {
        setWs(null)
        setCurrentId(null)
        setAnnotations([])
        setUploadErrors([])
      }
    } catch (e) { setError(msg(e)) }
    refreshSessions()
  }

  /**
   * Saves the draft: a new mask, or the edit of the one being worked on.
   *
   * Keeps the label afterwards, since the next instance usually shares it.
   */
  const commit = async () => {
    if (!currentId || draftIsEmpty(draft) || !draft.label.trim()) return
    setBusy(true)
    setError(null)
    try {
      if (draft.editing) {
        await api.patchAnnotation(draft.editing, {
          label: draft.label,
          prompts: { points: draft.points, boxes: draft.boxes },
          strokes: draft.strokes,
          window: win,
          threshold: draft.threshold,
          mask_index: draft.maskIndex,
        })
      } else {
        await api.createAnnotation(currentId, frame, draft, win)
      }
      await reloadAnnotations(currentId)
      await refreshWorkspace()
      setDraft(emptyDraft(draft.label))   // keep the label: the next instance usually shares it
      setPreview(null)
      setBust((b) => b + 1)
    } catch (e) {
      setError(msg(e))
    } finally {
      setBusy(false)
    }
  }

  /**
   * Loads a committed mask back into the draft, so editing resumes from the
   * exact state it was made in -- prompts, strokes, threshold and window alike.
   *
   * @param a The annotation to edit.
   */
  const editAnnotation = (a: Annotation) => {
    setDraft({
      editing: a.id, label: a.label,
      points: a.prompts.points, boxes: a.prompts.boxes, strokes: a.strokes,
      threshold: a.threshold, maskIndex: a.mask_index,
    })
    if (a.window) setWin(a.window)
    setError(null)
  }

  /**
   * Deletes a mask, clearing the draft if it was the one being edited.
   *
   * @param id Annotation id.
   */
  const removeAnnotation = async (id: string) => {
    setBusy(true)
    try {
      await api.deleteAnnotation(id)
      if (currentId) await reloadAnnotations(currentId)
      await refreshWorkspace()
      if (draft.editing === id) setDraft(emptyDraft(draft.label))
      setBust((b) => b + 1)
    } catch (e) { setError(msg(e)) } finally { setBusy(false) }
  }

  /**
   * Renames a mask, which may also move it to another label's colour.
   *
   * @param a The annotation to rename.
   */
  const renameAnnotation = async (a: Annotation) => {
    const next = window.prompt(`Rename "${a.label} #${a.instance}"`, a.label)
    if (!next || next === a.label) return
    try {
      await api.patchAnnotation(a.id, { label: next })
      if (currentId) await reloadAnnotations(currentId)
      await refreshWorkspace()
      setBust((b) => b + 1)
    } catch (e) { setError(msg(e)) }
  }

  /**
   * Marks a file done, or not done.
   *
   * @param id Image id.
   * @param value The new state.
   */
  const toggleReviewed = async (id: string, value: boolean) => {
    // Flip locally first: a controlled checkbox that waits for the server reads
    // as an unresponsive click.
    setWs((w) => w && {
      ...w, images: w.images.map((i) => (i.image_id === id ? { ...i, reviewed: value } : i)),
    })
    try { await api.setReviewed(id, value) } catch (e) { setError(msg(e)) }
    await refreshWorkspace()
  }

  /**
   * Removes a file from the workspace, moving on if it was the open one.
   *
   * @param id Image id.
   */
  const removeImage = async (id: string) => {
    try {
      await api.deleteImage(id)
      const rest = images.filter((i) => i.image_id !== id)
      await refreshWorkspace()
      if (currentId === id) {
        if (rest.length) select(rest[0].image_id)
        else { setCurrentId(null); setAnnotations([]) }
      }
    } catch (e) { setError(msg(e)) }
  }

  /**
   * Saves a server response to disk, via a synthetic link.
   *
   * @param path URL to download.
   * @param filename Name to suggest to the browser.
   */
  const download = (path: string, filename: string) => {
    const a = document.createElement('a')
    a.href = path
    a.download = filename
    document.body.appendChild(a)
    a.click()
    a.remove()
  }

  /**
   * Moves through the file list, stopping at either end.
   *
   * @param delta How many files to move; 1 for next, -1 for previous.
   */
  const step = useCallback((delta: number) => {
    if (!images.length) return
    const i = images.findIndex((im) => im.image_id === currentId)
    const next = images[Math.max(0, Math.min(images.length - 1, (i < 0 ? 0 : i) + delta))]
    if (next && next.image_id !== currentId) select(next.image_id)
  }, [images, currentId, select])

  // ---- keyboard -------------------------------------------------------

  // Deliberately re-bound on every render: the handler closes over the draft and
  // the current file, and stale copies would commit the wrong thing.
  useEffect(() => {
    /**
     * Handles a global shortcut.
     *
     * @param e The key event. Keys are ignored while a text field has focus, so
     *     typing a label never fires a tool.
     */
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) {
        if (e.key === 'Enter') (t as HTMLInputElement).blur()
        return
      }
      const tools: Record<string, Tool> = { '1': 'include', '2': 'exclude', '3': 'box', '4': 'brush', '5': 'eraser' }
      if (tools[e.key]) { setTool(tools[e.key]); return }
      if (e.key === '[') { setBrushRadius((r) => Math.max(1, r - 2)); return }
      if (e.key === ']') { setBrushRadius((r) => Math.min(60, r + 2)); return }
      if (e.key === 'ArrowLeft') { setFrame((f) => Math.max(0, f - 1)); return }
      if (e.key === 'ArrowRight') { setFrame((f) => Math.min((current?.frames ?? 1) - 1, f + 1)); return }
      if (e.key === 'n') { step(1); return }
      if (e.key === 'p') { step(-1); return }
      if (e.key === 'd' && currentId) { toggleReviewed(currentId, !(current?.reviewed ?? false)); return }
      if (e.key === 'Enter') { commit(); return }
      if (e.key === 'Escape') { setDraft((d) => emptyDraft(d.label)); setPreview(null); return }
      if ((e.key === 'Delete' || e.key === 'Backspace') && draft.editing) {
        e.preventDefault()
        removeAnnotation(draft.editing)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  // ---- geometry -------------------------------------------------------

  // Fit the frame to the stage, then apply the user's zoom on top.
  const scale = useMemo(() => {
    if (!info) return 1
    const fit = Math.min((stageSize.w - 40) / info.columns, (stageSize.h - 40) / info.rows)
    return Math.max(0.05, (Number.isFinite(fit) && fit > 0 ? fit : 1) * zoom)
  }, [info, stageSize, zoom])

  // Skip the overlay request entirely when this frame has nothing on it.
  const overlay = currentId && showMasks && annotations.some((a) => a.frame === frame)
    ? api.overlayUrl(currentId, frame, {
        selected: draft.editing, exclude: draft.editing, alpha: MASK_ALPHA, bust,
      })
    : null

  return (
    <div className="app">
      <FileList
        workspace={ws}
        currentId={currentId}
        busy={busy}
        errors={uploadErrors}
        sessions={sessions}
        persist={persist}
        showSessions={showSessions}
        onToggleSessions={() => { setShowSessions((v) => !v); refreshSessions() }}
        onOpenSession={openSession}
        onDeleteSession={removeSession}
        onPick={(files) => pick(files, files.some((f) => (f as File & { webkitRelativePath?: string }).webkitRelativePath))}
        onSelect={select}
        onToggleReviewed={toggleReviewed}
        onRemove={removeImage}
        onExportJson={() => ws && download(`/export.json?workspace_id=${ws.workspace_id}`, 'annotations.json')}
        onExportZip={() => ws && download(`/export.zip?workspace_id=${ws.workspace_id}`, 'annotations.zip')}
      />

      <main className="main">
        <Toolbar
          tool={tool} setTool={setTool}
          brushRadius={brushRadius} setBrushRadius={setBrushRadius}
          frame={frame} frames={current?.frames ?? 1} setFrame={setFrame}
          win={win} defaultWin={info?.default_window ?? null} setWin={setWin}
          windowing={info?.windowing ?? false}
          zoom={zoom} setZoom={setZoom}
          showMasks={showMasks} setShowMasks={setShowMasks}
        />

        <div className="stage" ref={stage}>
          {currentId && info ? (
            <Viewer
              rows={info.rows}
              columns={info.columns}
              scale={scale}
              baseUrl={api.frameUrl(currentId, frame, win)}
              overlayUrl={overlay}
              previewUrl={showMasks ? preview?.url ?? null : null}
              draft={draft}
              tool={tool}
              brushRadius={brushRadius}
              brushColor={colorFor(draft.label)}
              onPoint={(x, y, label) => setDraft((d) => ({ ...d, points: [...d.points, { x, y, label }] }))}
              onBox={(b: Box) => setDraft((d) => ({ ...d, boxes: [...d.boxes, b] }))}
              onStroke={(s: Stroke) => setDraft((d) => ({ ...d, strokes: [...d.strokes, s] }))}
            />
          ) : (
            <div className="empty">
              <h2>Open a folder of images</h2>
              <p>DICOM, PNG, JPEG, TIFF or a .zip. Every file shows up in the list on the left;
                 label them one by one and export the whole batch in a single file.</p>
              {(persist || sessions.length > 0) && (
                <div className="resume">
                  <h3>Pick up where you left off</h3>
                  <Sessions sessions={sessions} persist={persist} currentId={ws?.workspace_id ?? null}
                            busy={busy} onOpen={openSession} onDelete={removeSession} />
                </div>
              )}
            </div>
          )}
        </div>

        <div className="statusbar">
          <span>{current ? current.filename : 'no file'}</span>
          {info && <span>{info.columns}×{info.rows}{current?.modality ? ` · ${current.modality}` : ''}</span>}
          {busy && <span className="pulse">working…</span>}
          <span className="grow" />
          <span className="muted">{model && `model: ${model}`}</span>
        </div>
      </main>

      <SidePanel
        draft={draft}
        setDraft={setDraft}
        labels={ws?.labels ?? []}
        annotations={annotations}
        frame={frame}
        frames={current?.frames ?? 1}
        previewArea={preview?.area ?? null}
        previewScore={preview?.score ?? null}
        busy={busy}
        error={error}
        onCommit={commit}
        onCancel={() => { setDraft(emptyDraft(draft.label)); setPreview(null) }}
        onEdit={editAnnotation}
        onDelete={removeAnnotation}
        onRename={renameAnnotation}
      />
    </div>
  )
}
