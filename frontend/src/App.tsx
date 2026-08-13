import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import * as api from './api'
import FileList from './components/FileList'
import SidePanel from './components/SidePanel'
import Toolbar from './components/Toolbar'
import Viewer from './components/Viewer'
import type {
  Annotation, Box, Draft, FrameInfo, PreviewResult, Stroke, Tool, Window, WorkspaceListing,
} from './types'
import { draftIsEmpty, emptyDraft } from './types'

const FALLBACK_COLOR = '#E8453C'
const PREVIEW_DEBOUNCE = 70   // ms; enough to coalesce a burst of clicks

const msg = (e: unknown) => (e instanceof Error ? e.message : String(e))

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
  const [opacity, setOpacity] = useState(110)
  const [showMasks, setShowMasks] = useState(true)

  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [uploadErrors, setUploadErrors] = useState<string[]>([])
  const [bust, setBust] = useState(0)
  const [model, setModel] = useState('')

  const stage = useRef<HTMLDivElement>(null)
  const [stageSize, setStageSize] = useState({ w: 900, h: 700 })

  const images = ws?.images ?? []
  const current = images.find((i) => i.image_id === currentId) ?? null

  // ---- data loading ---------------------------------------------------

  useEffect(() => { api.health().then((h) => setModel(h.model)).catch(() => {}) }, [])

  useEffect(() => {
    const el = stage.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setStageSize({ w: e.contentRect.width, h: e.contentRect.height }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  useEffect(() => {
    if (!currentId) { setInfo(null); return }
    let live = true
    api.frameInfo(currentId, frame)
      .then((i) => { if (live) setInfo(i) })
      .catch((e) => { if (live) setError(msg(e)) })
    return () => { live = false }
  }, [currentId, frame])

  const reloadAnnotations = useCallback(async (id: string) => {
    setAnnotations(await api.listAnnotations(id))
  }, [])

  useEffect(() => {
    if (!currentId) { setAnnotations([]); return }
    reloadAnnotations(currentId).catch((e) => setError(msg(e)))
  }, [currentId, reloadAnnotations])

  const colorFor = useCallback((label: string) => {
    const hit = ws?.labels.find((l) => l.name.toLowerCase() === label.trim().toLowerCase())
    return hit?.color ?? FALLBACK_COLOR
  }, [ws])

  // ---- live preview ---------------------------------------------------

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

  const select = useCallback((id: string) => {
    setCurrentId(id)
    setFrame(0)
    setWin(null)
    setPreview(null)
    setError(null)
    setDraft((d) => emptyDraft(d.label))
  }, [])

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
    }
  }

  const refreshWorkspace = useCallback(async () => {
    if (!ws) return
    try { setWs(await api.getWorkspace(ws.workspace_id)) } catch (e) { setError(msg(e)) }
  }, [ws])

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

  const editAnnotation = (a: Annotation) => {
    setDraft({
      editing: a.id, label: a.label,
      points: a.prompts.points, boxes: a.prompts.boxes, strokes: a.strokes,
      threshold: a.threshold, maskIndex: a.mask_index,
    })
    if (a.window) setWin(a.window)
    setError(null)
  }

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

  const toggleReviewed = async (id: string, value: boolean) => {
    // Flip locally first: a controlled checkbox that waits for the server reads
    // as an unresponsive click.
    setWs((w) => w && {
      ...w, images: w.images.map((i) => (i.image_id === id ? { ...i, reviewed: value } : i)),
    })
    try { await api.setReviewed(id, value) } catch (e) { setError(msg(e)) }
    await refreshWorkspace()
  }

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

  const download = (path: string, filename: string) => {
    const a = document.createElement('a')
    a.href = path
    a.download = filename
    document.body.appendChild(a)
    a.click()
    a.remove()
  }

  const step = useCallback((delta: number) => {
    if (!images.length) return
    const i = images.findIndex((im) => im.image_id === currentId)
    const next = images[Math.max(0, Math.min(images.length - 1, (i < 0 ? 0 : i) + delta))]
    if (next && next.image_id !== currentId) select(next.image_id)
  }, [images, currentId, select])

  // ---- keyboard -------------------------------------------------------

  useEffect(() => {
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

  const scale = useMemo(() => {
    if (!info) return 1
    const fit = Math.min((stageSize.w - 40) / info.columns, (stageSize.h - 40) / info.rows)
    return Math.max(0.05, (Number.isFinite(fit) && fit > 0 ? fit : 1) * zoom)
  }, [info, stageSize, zoom])

  const overlay = currentId && showMasks && annotations.some((a) => a.frame === frame)
    ? api.overlayUrl(currentId, frame, {
        selected: draft.editing, exclude: draft.editing, alpha: opacity, bust,
      })
    : null

  return (
    <div className="app">
      <FileList
        workspace={ws}
        currentId={currentId}
        busy={busy}
        errors={uploadErrors}
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
          opacity={opacity} setOpacity={setOpacity}
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
