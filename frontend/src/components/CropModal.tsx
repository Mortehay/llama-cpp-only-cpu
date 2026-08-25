import { useCallback, useRef, useState } from 'react'
import { api } from '../api'

/**
 * Drag a box on an image and save the crop as a new concept.
 *
 * Hand-rolled rather than cropper.js, which the legacy page pulled from a CDN.
 * The interaction here is one rectangle - the library was 40 KB and an external
 * dependency for a drag handler.
 *
 * THE ONE THING THAT MATTERS: the server crops in the ORIGINAL image's pixel
 * space, but the user drags in DISPLAY space, and a 1024px concept shown at
 * 520px wide is a factor of ~2 out. Every coordinate is scaled by
 * naturalWidth / clientWidth before it leaves here. Getting this wrong does not
 * error - it silently crops the wrong region, which is the kind of bug that
 * gets blamed on the model.
 */

interface Rect {
  x: number
  y: number
  w: number
  h: number
}

export default function CropModal({
  sourceId,
  url,
  onClose,
  onSaved,
}: {
  sourceId: number
  url: string
  onClose: () => void
  onSaved?: () => void
}) {
  const imgRef = useRef<HTMLImageElement>(null)
  const [start, setStart] = useState<{ x: number; y: number } | null>(null)
  const [rect, setRect] = useState<Rect | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  /** Pointer position relative to the image, clamped to its box. */
  const local = useCallback((e: React.PointerEvent) => {
    const img = imgRef.current
    if (!img) return { x: 0, y: 0 }
    const box = img.getBoundingClientRect()
    return {
      x: Math.max(0, Math.min(box.width, e.clientX - box.left)),
      y: Math.max(0, Math.min(box.height, e.clientY - box.top)),
    }
  }, [])

  function down(e: React.PointerEvent) {
    e.preventDefault()
    // Capture, so a drag that leaves the image still delivers move/up events
    // instead of stranding the selection mid-drag.
    ;(e.target as Element).setPointerCapture?.(e.pointerId)
    const p = local(e)
    setStart(p)
    setRect({ x: p.x, y: p.y, w: 0, h: 0 })
  }

  function move(e: React.PointerEvent) {
    if (!start) return
    const p = local(e)
    setRect({
      x: Math.min(start.x, p.x),
      y: Math.min(start.y, p.y),
      w: Math.abs(p.x - start.x),
      h: Math.abs(p.y - start.y),
    })
  }

  function up() {
    setStart(null)
  }

  async function save() {
    const img = imgRef.current
    if (!img || !rect || rect.w < 4 || rect.h < 4) {
      setError('Drag a box on the image first.')
      return
    }
    // Display space -> natural space. Guarded because a zero clientWidth (image
    // not laid out yet) would divide to Infinity and post NaN coordinates - the
    // server would take them and crop something arbitrary rather than refuse.
    if (!img.clientWidth || !img.clientHeight || !img.naturalWidth) {
      setError('The image has not finished loading - try again in a moment.')
      return
    }
    const sx = img.naturalWidth / img.clientWidth
    const sy = img.naturalHeight / img.clientHeight
    setBusy(true)
    setError(null)
    try {
      await api.crop(
        sourceId,
        Math.round(rect.x * sx),
        Math.round(rect.y * sy),
        Math.round(rect.w * sx),
        Math.round(rect.h * sy),
      )
      onSaved?.()
      onClose()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const natural = imgRef.current
  const scaleX = natural && natural.clientWidth ? natural.naturalWidth / natural.clientWidth : 1
  const scaleY = natural && natural.clientHeight ? natural.naturalHeight / natural.clientHeight : 1

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0,0,0,.72)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 50,
        padding: 20,
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose()
      }}
    >
      <div className="card" style={{ maxWidth: 720, width: '100%', margin: 0 }}>
        <h2>Crop to a character</h2>
        <p className="hint">
          Drag a box around the character. The crop is background-keyed and saved as a
          new concept, so the original is untouched.
        </p>

        {error && <div className="note err">{error}</div>}

        <div
          style={{
            position: 'relative',
            display: 'inline-block',
            touchAction: 'none',
            cursor: 'crosshair',
            maxWidth: '100%',
          }}
          onPointerDown={down}
          onPointerMove={move}
          onPointerUp={up}
        >
          <img
            ref={imgRef}
            src={url}
            alt="crop source"
            draggable={false}
            style={{
              maxWidth: '100%',
              maxHeight: '58vh',
              display: 'block',
              imageRendering: 'pixelated',
              userSelect: 'none',
            }}
          />
          {rect && rect.w > 0 && (
            <div
              style={{
                position: 'absolute',
                left: rect.x,
                top: rect.y,
                width: rect.w,
                height: rect.h,
                border: '1px solid var(--accent2)',
                boxShadow: '0 0 0 9999px rgba(0,0,0,.45)',
                pointerEvents: 'none',
              }}
            />
          )}
        </div>

        {rect && rect.w > 4 && (
          <div className="muted" style={{ marginTop: 8 }}>
            {Math.round(rect.w * scaleX)} × {Math.round(rect.h * scaleY)} source pixels
          </div>
        )}

        <div className="acts" style={{ marginTop: 14 }}>
          <button className="btn" disabled={busy || !rect} onClick={() => void save()}>
            {busy ? 'Saving…' : 'Save crop'}
          </button>
          <button className="btn ghost" onClick={onClose}>
            Cancel
          </button>
          {rect && (
            <button className="btn ghost" onClick={() => setRect(null)}>
              Clear
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
