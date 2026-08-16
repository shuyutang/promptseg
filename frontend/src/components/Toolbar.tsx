/**
 * @fileoverview The strip above the image: tool, brush size, frame, window/level,
 * mask visibility and zoom.
 *
 * Window and level sit here rather than in a menu because they decide what the
 * model is given, not just what the screen shows -- they are worth turning
 * before the first click, not after a bad mask.
 */
import type { Tool, Window } from '../types'

/** Props for {@link Toolbar}. */
type Props = {
  /** The selected pointer tool. */
  tool: Tool
  setTool: (t: Tool) => void
  /** Brush and eraser radius, in image pixels. */
  brushRadius: number
  setBrushRadius: (r: number) => void
  /** Current frame index; the slider only appears for multi-frame files. */
  frame: number
  frames: number
  setFrame: (f: number) => void
  /** The window in force, or null to use the file's own. */
  win: Window | null
  /** The file's own window, which also sets the sliders' range. */
  defaultWin: Window | null
  setWin: (w: Window | null) => void
  /** False for images with no meaningful intensity range; hides the sliders. */
  windowing: boolean
  /** Zoom on top of fit-to-window, where 1 is a fit. */
  zoom: number
  setZoom: (z: number) => void
  /** Whether committed masks and the preview are drawn at all. */
  showMasks: boolean
  setShowMasks: (v: boolean) => void
}

/** The tools, in shortcut order; the keys match the global handler in App. */
const TOOLS: { id: Tool; label: string; key: string; hint: string }[] = [
  { id: 'include', label: '＋ Point', key: '1', hint: 'Click what you want (right-click always excludes)' },
  { id: 'exclude', label: '－ Point', key: '2', hint: 'Click what to leave out' },
  { id: 'box', label: '▭ Box', key: '3', hint: 'Drag a box around the target' },
  { id: 'brush', label: '✎ Brush', key: '4', hint: 'Paint pixels into the mask' },
  { id: 'eraser', label: '⌫ Erase', key: '5', hint: 'Paint pixels out of the mask' },
]

/**
 * Renders the toolbar.
 *
 * @param p Component props.
 * @return The strip, with controls that do not apply to the open file omitted.
 */
export default function Toolbar(p: Props) {
  const painting = p.tool === 'brush' || p.tool === 'eraser'
  return (
    <div className="toolbar">
      <div className="tools">
        {TOOLS.map((t) => (
          <button key={t.id} className={p.tool === t.id ? 'tool on' : 'tool'}
                  title={`${t.hint}  (${t.key})`} onClick={() => p.setTool(t.id)}>
            {t.label}
          </button>
        ))}
      </div>

      {painting && (
        <label className="ctl">
          Size
          <input type="range" min={1} max={60} value={p.brushRadius}
                 onChange={(e) => p.setBrushRadius(Number(e.currentTarget.value))} />
          <span className="num">{p.brushRadius}</span>
        </label>
      )}

      {p.frames > 1 && (
        <label className="ctl">
          Frame
          <input type="range" min={0} max={p.frames - 1} value={p.frame}
                 onChange={(e) => p.setFrame(Number(e.currentTarget.value))} />
          <span className="num">{p.frame + 1}/{p.frames}</span>
        </label>
      )}

      {p.windowing && p.defaultWin && (
        <>
          <label className="ctl" title="Window centre (level)">
            L
            <input type="range"
                   min={p.defaultWin.center - p.defaultWin.width}
                   max={p.defaultWin.center + p.defaultWin.width}
                   step={Math.max(1, p.defaultWin.width / 200)}
                   value={p.win?.center ?? p.defaultWin.center}
                   onChange={(e) => p.setWin({
                     center: Number(e.currentTarget.value),
                     width: p.win?.width ?? p.defaultWin!.width,
                   })} />
          </label>
          <label className="ctl" title="Window width">
            W
            <input type="range"
                   min={Math.max(1, p.defaultWin.width / 20)}
                   max={p.defaultWin.width * 4}
                   step={Math.max(1, p.defaultWin.width / 200)}
                   value={p.win?.width ?? p.defaultWin.width}
                   onChange={(e) => p.setWin({
                     center: p.win?.center ?? p.defaultWin!.center,
                     width: Number(e.currentTarget.value),
                   })} />
          </label>
          <button className="ghost" onClick={() => p.setWin(null)} title="Back to the file's own window">
            reset
          </button>
        </>
      )}

      <label className="ctl" title="Show the masks drawn on this frame">
        <input type="checkbox" checked={p.showMasks}
               onChange={(e) => p.setShowMasks(e.currentTarget.checked)} />
        Masks
      </label>

      <div className="zoom">
        <button onClick={() => p.setZoom(Math.max(0.25, p.zoom / 1.25))} title="Zoom out">−</button>
        <span className="num">{Math.round(p.zoom * 100)}%</span>
        <button onClick={() => p.setZoom(Math.min(16, p.zoom * 1.25))} title="Zoom in">+</button>
        <button className="ghost" onClick={() => p.setZoom(1)}>fit</button>
      </div>
    </div>
  )
}
