/**
 * @fileoverview The image itself: four stacked layers and all pointer handling.
 *
 * Bottom to top the layers are the base frame PNG, the overlay of committed
 * masks, the live preview, and a canvas. Only the canvas is drawn here, and it
 * holds nothing but prompt markers and the stroke still under the pointer --
 * every mask pixel comes from the server, so client and server can never
 * disagree about what a mask covers.
 *
 * The canvas is sized in native image pixels and scaled by CSS, which is what
 * lets a click be mapped straight back to image coordinates at any zoom.
 */
import { useEffect, useRef, useState } from 'react'
import type { Box, Draft, Stroke, Tool } from '../types'

/** Props for {@link Viewer}. */
type Props = {
  /** Frame height in native pixels. */
  rows: number
  /** Frame width in native pixels. */
  columns: number
  /** Screen pixels per image pixel. */
  scale: number
  /** URL of the rendered frame. */
  baseUrl: string
  /** URL of the committed-mask overlay, or null when there is nothing to show. */
  overlayUrl: string | null
  /** Object URL of the live preview, or null. */
  previewUrl: string | null
  /** The draft, whose points and boxes are drawn as markers. */
  draft: Draft
  /** The selected pointer tool. */
  tool: Tool
  /** Brush and eraser radius, in image pixels. */
  brushRadius: number
  /** The label's colour, used for the in-flight brush stroke. */
  brushColor: string
  /** Called with a click, in image pixels; `label` 1 includes, 0 excludes. */
  onPoint: (x: number, y: number, label: 1 | 0) => void
  /** Called with a finished box, in image pixels. */
  onBox: (box: Box) => void
  /** Called with a finished brush or eraser stroke. */
  onStroke: (stroke: Stroke) => void
}

/** Marker radius in screen px, kept constant across zoom. */
const DOT = 4

/** Drags shorter than this, in image px, are treated as clicks. */
const MIN_BOX = 3

/**
 * Renders the image stack and turns pointer events into prompts.
 *
 * @param p Component props.
 * @return The layered viewer at the current scale.
 */
export default function Viewer(p: Props) {
  const { rows, columns, scale } = p
  const canvas = useRef<HTMLCanvasElement>(null)
  const [drag, setDrag] = useState<{ kind: 'box' | 'paint'; pts: [number, number][] } | null>(null)
  const [hover, setHover] = useState<[number, number] | null>(null)
  // The painted stroke stays on screen until the server's preview catches up,
  // otherwise the brush blinks off between mouse-up and the new PNG.
  const [wip, setWip] = useState<Stroke | null>(null)

  useEffect(() => { setWip(null) }, [p.previewUrl])

  /**
   * Maps a pointer event to image coordinates.
   *
   * @param e The pointer event.
   * @return `[x, y]` in native image pixels, clamped to the frame.
   */
  const native = (e: React.PointerEvent): [number, number] => {
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect()
    return [
      Math.max(0, Math.min(columns - 1, Math.floor((e.clientX - r.left) / r.width * columns))),
      Math.max(0, Math.min(rows - 1, Math.floor((e.clientY - r.top) / r.height * rows))),
    ]
  }

  /**
   * Starts a click, a box or a stroke, depending on the tool and the button.
   *
   * @param e The pointer event.
   */
  const down = (e: React.PointerEvent) => {
    if (e.button !== 0 && e.button !== 2) return
    e.preventDefault()
    ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
    const [x, y] = native(e)

    // Right-click always means "not this", whichever tool is selected.
    if (e.button === 2) { p.onPoint(x, y, 0); return }
    if (p.tool === 'box') { setDrag({ kind: 'box', pts: [[x, y], [x, y]] }); return }
    if (p.tool === 'brush' || p.tool === 'eraser') { setDrag({ kind: 'paint', pts: [[x, y]] }); return }
    p.onPoint(x, y, p.tool === 'exclude' ? 0 : 1)
  }

  /**
   * Extends the drag in progress and tracks the brush cursor.
   *
   * @param e The pointer event.
   */
  const move = (e: React.PointerEvent) => {
    const pt = native(e)
    setHover(pt)
    if (!drag) return
    if (drag.kind === 'box') setDrag({ kind: 'box', pts: [drag.pts[0], pt] })
    else setDrag({ kind: 'paint', pts: [...drag.pts, pt] })
  }

  /** Finishes the drag, emitting a box, a stroke, or a click if it was tiny. */
  const up = () => {
    if (!drag) return
    const pts = drag.pts
    setDrag(null)
    if (drag.kind === 'box') {
      const [[x0, y0], [x1, y1]] = [pts[0], pts[pts.length - 1]]
      if (Math.abs(x1 - x0) < MIN_BOX || Math.abs(y1 - y0) < MIN_BOX) {
        p.onPoint(x0, y0, 1)   // a click with the box tool is still a click
        return
      }
      p.onBox([Math.min(x0, x1), Math.min(y0, y1), Math.max(x0, x1), Math.max(y0, y1)])
    } else {
      const stroke: Stroke = {
        mode: p.tool === 'eraser' ? 'erase' : 'add',
        radius: p.brushRadius,
        points: pts,
      }
      setWip(stroke)
      p.onStroke(stroke)
    }
  }

  // ---- marker layer ---------------------------------------------------

  // Redraw the markers: the in-flight stroke, boxes, points and the brush
  // cursor. Line widths divide by the scale so they stay constant on screen.
  useEffect(() => {
    const c = canvas.current
    if (!c) return
    const ctx = c.getContext('2d')!
    ctx.clearRect(0, 0, columns, rows)
    const s = 1 / scale                       // screen px -> native px

    if (wip || drag?.kind === 'paint') {
      const stroke = wip ?? { mode: p.tool === 'eraser' ? 'erase' : 'add', radius: p.brushRadius, points: drag!.pts }
      ctx.strokeStyle = stroke.mode === 'erase' ? 'rgba(20,20,24,.85)' : p.brushColor
      ctx.lineWidth = stroke.radius * 2
      ctx.lineCap = 'round'
      ctx.lineJoin = 'round'
      ctx.globalAlpha = 0.55
      ctx.beginPath()
      stroke.points.forEach(([x, y], i) => (i ? ctx.lineTo(x + .5, y + .5) : ctx.moveTo(x + .5, y + .5)))
      if (stroke.points.length === 1) ctx.lineTo(stroke.points[0][0] + .5, stroke.points[0][1] + .5)
      ctx.stroke()
      ctx.globalAlpha = 1
    }

    for (const b of p.draft.boxes) {
      ctx.strokeStyle = '#22d3ee'
      ctx.lineWidth = 1.5 * s
      ctx.setLineDash([5 * s, 4 * s])
      ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1])
      ctx.setLineDash([])
    }
    if (drag?.kind === 'box') {
      const [[x0, y0], [x1, y1]] = [drag.pts[0], drag.pts[1]]
      ctx.strokeStyle = '#22d3ee'
      ctx.lineWidth = 1.5 * s
      ctx.setLineDash([5 * s, 4 * s])
      ctx.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0))
      ctx.setLineDash([])
    }

    for (const pt of p.draft.points) {
      ctx.beginPath()
      ctx.arc(pt.x + .5, pt.y + .5, DOT * s, 0, Math.PI * 2)
      ctx.fillStyle = pt.label === 1 ? '#22c55e' : '#ef4444'
      ctx.fill()
      ctx.lineWidth = 1.5 * s
      ctx.strokeStyle = '#0b0d12'
      ctx.stroke()
    }

    if (hover && (p.tool === 'brush' || p.tool === 'eraser') && !drag) {
      ctx.beginPath()
      ctx.arc(hover[0] + .5, hover[1] + .5, p.brushRadius, 0, Math.PI * 2)
      ctx.strokeStyle = p.tool === 'eraser' ? '#f87171' : '#e5e7eb'
      ctx.lineWidth = 1 * s
      ctx.stroke()
    }
  }, [p.draft, drag, hover, wip, columns, rows, scale, p.tool, p.brushRadius, p.brushColor])

  const w = Math.round(columns * scale)
  const h = Math.round(rows * scale)

  return (
    <div className="stack" style={{ width: w, height: h }}>
      <img className="layer" src={p.baseUrl} alt="" width={w} height={h} draggable={false} />
      {p.overlayUrl && <img className="layer" src={p.overlayUrl} alt="" width={w} height={h} draggable={false} />}
      {p.previewUrl && <img className="layer" src={p.previewUrl} alt="" width={w} height={h} draggable={false} />}
      <canvas
        ref={canvas}
        className={`layer marks tool-${p.tool}`}
        width={columns}
        height={rows}
        style={{ width: w, height: h }}
        onPointerDown={down}
        onPointerMove={move}
        onPointerUp={up}
        onPointerLeave={() => setHover(null)}
        onContextMenu={(e) => e.preventDefault()}
      />
    </div>
  )
}
