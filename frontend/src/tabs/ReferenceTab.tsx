import { useRef, useState } from 'react'
import { api, type Reference, type ReferenceKind } from '../api'
import { useAsync } from '../hooks'
import StyleProfilePanel from '../components/StyleProfilePanel'

/**
 * One tab per reference kind. The copy differs because the three kinds answer
 * genuinely different questions, and telling the user which one they are
 * answering is most of the value.
 */
const COPY: Record<
  ReferenceKind,
  { title: string; hint: string; want: string; measures: string[] }
> = {
  tile: {
    title: 'Reference · Ground tiles',
    hint:
      'The most valuable thing you can upload. A ground tile is a rhombus, and its ' +
      'width:height ratio IS your world’s camera angle — a 2:1 tile means a 26.6° ' +
      'camera. Everything else can be guessed at; this cannot, and one tile settles it.',
    want: 'A single ground tile with transparent corners, at its native size.',
    measures: ['projection ratio', 'camera elevation', 'palette', 'pixel scale'],
  },
  sprite: {
    title: 'Reference · Finished sprites',
    hint:
      'Sprites you already consider correct. These fix the palette, the pixel grid, the ' +
      'colour budget and the outline — all measured, all applied as hard constraints, ' +
      'no training required.',
    want: 'Finished character sprites or spritesheets, PNG with transparency.',
    measures: ['pixel scale', 'exact colour count', 'palette', 'outline', 'alpha hardness'],
  },
  core: {
    title: 'Reference · Character concepts',
    hint:
      'Full-body character art of the kind step 1 should produce. Judged for isolation: ' +
      'one character, no scenery, no frame. A landscape once produced twelve structurally ' +
      'perfect cells of a tree before anything noticed.',
    want: 'A single character on a plain or transparent background.',
    measures: ['frame coverage', 'border contact', 'aspect ratio'],
  },
}

/** Metrics worth putting on the card, in the order a human reads them. */
const HEADLINE: Record<ReferenceKind, string[]> = {
  tile: ['projection_ratio', 'elevation_deg', 'elevation', 'colors'],
  sprite: ['scale', 'art_w', 'art_h', 'colors', 'outline_color'],
  core: ['coverage', 'border', 'aspect', 'colors'],
}

const LABELS: Record<string, string> = {
  projection_ratio: 'ratio',
  elevation_deg: 'angle',
  elevation: 'camera',
  colors: 'colours',
  scale: 'px scale',
  art_w: 'art w',
  art_h: 'art h',
  outline_color: 'outline',
  coverage: 'coverage',
  border: 'border',
  aspect: 'aspect',
}

export default function ReferenceTab({ kind }: { kind: ReferenceKind }) {
  const copy = COPY[kind]
  const list = useAsync(() => api.references(kind), [kind])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [over, setOver] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)

  async function upload(files: FileList | null) {
    if (!files?.length) return
    setBusy(true)
    setError(null)
    try {
      // Sequential, not Promise.all: each upload runs a measurement, and a
      // dozen at once would compete for the same CPU for no gain.
      for (const file of Array.from(files)) {
        await api.uploadReference(kind, file, file.name)
      }
      list.reload()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
      if (fileInput.current) fileInput.current.value = ''
    }
  }

  async function act(fn: () => Promise<unknown>) {
    setBusy(true)
    setError(null)
    try {
      await fn()
      list.reload()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <div className="card">
        <h2>{copy.title}</h2>
        <p className="hint">{copy.hint}</p>

        {error && <div className="note err">{error}</div>}

        <div
          className={`drop ${over ? 'over' : ''}`}
          onClick={() => fileInput.current?.click()}
          onDragOver={(e) => {
            e.preventDefault()
            setOver(true)
          }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setOver(false)
            void upload(e.dataTransfer.files)
          }}
        >
          {busy ? 'Measuring…' : <>
            <strong>Drop images here</strong>, or click to choose.
            <div className="muted" style={{ marginTop: 6 }}>{copy.want}</div>
          </>}
        </div>
        <input
          ref={fileInput}
          type="file"
          accept="image/*"
          multiple
          style={{ display: 'none' }}
          onChange={(e) => void upload(e.target.files)}
        />

        <div className="muted" style={{ marginTop: 10 }}>
          Measured on upload: {copy.measures.join(' · ')}
        </div>
      </div>

      <div className="card">
        <h2>
          Uploaded {kind} references{' '}
          {list.data ? (
            <>
              <span className="tag ok">{list.data.trainable} trainable</span>{' '}
              <span className="tag neutral">{list.data.usable} measurable</span>
            </>
          ) : null}
        </h2>
        <p className="hint">
          Every reference carries <strong>two</strong> verdicts, because they answer
          different questions. <strong>Trainable</strong> — can a style LoRA learn
          from this? Nearly everything qualifies; a JPEG reference board is fine.
          <strong> Measurable</strong> — can an exact palette, pixel grid or camera
          angle be read off it? That needs palette-locked art with hard edges, and
          only measurable references feed a style profile.
        </p>
        <div className="row tight" style={{ marginBottom: 12 }}>
          <div style={{ flex: '0 0 auto' }}>
            <button className="btn ghost sm" disabled={busy} onClick={() => void act(() => api.remeasureAll(kind))}>
              Re-measure all {kind} references
            </button>
          </div>
          <div className="muted" style={{ flex: '1 1 auto' }}>
            Run this after the measurement rules change — it re-judges what is
            already uploaded instead of making you upload it again.
          </div>
        </div>

        {list.error && <div className="note err">{list.error}</div>}
        {list.loading && !list.data && <div className="empty">Loading…</div>}
        {list.data?.items.length === 0 && (
          <div className="empty">No {kind} references yet.</div>
        )}

        <div className="grid">
          {list.data?.items.map((r) => (
            <Card
              key={r.id}
              r={r}
              fields={HEADLINE[kind]}
              busy={busy}
              onDelete={() => act(() => api.deleteReference(r.id))}
              onRemeasure={() => act(() => api.remeasure(r.id))}
            />
          ))}
        </div>
      </div>

      <StyleProfilePanel onChange={list.reload} />
    </>
  )
}

function Card({
  r,
  fields,
  busy,
  onDelete,
  onRemeasure,
}: {
  r: Reference
  fields: string[]
  busy: boolean
  onDelete: () => void
  onRemeasure: () => void
}) {
  const palette = (r.metrics?.palette as string[] | undefined) ?? []
  return (
    <div className="thumb">
      <div className="pic">{r.url && <img src={r.url} alt={r.label} />}</div>
      <div className="meta">
        <div className="name" title={r.label}>
          {r.label}
        </div>
        {/* TWO verdicts, because they answer different questions. An image can
            be perfect training material and still useless for measuring a
            palette - most reference boards are exactly that. */}
        <div style={{ marginTop: 5, display: 'flex', gap: 5, flexWrap: 'wrap' }}>
          <span className={`tag ${r.trainable ? 'ok' : 'no'}`}>
            {r.trainable ? 'trainable' : 'not trainable'}
          </span>
          <span className={`tag ${r.usable ? 'ok' : 'neutral'}`}>
            {r.usable ? 'measurable' : 'not measurable'}
          </span>
        </div>
        <div className="why">
          {r.trainable === false ? r.trainable_why : r.why}
        </div>

        <div className="why" style={{ marginTop: 5 }}>
          {fields
            .filter((f) => r.metrics?.[f] !== undefined && r.metrics?.[f] !== null)
            .map((f) => `${LABELS[f] ?? f} ${String(r.metrics[f])}`)
            .join(' · ')}
        </div>

        {palette.length > 0 && (
          <div className="swatches">
            {palette.slice(0, 16).map((c) => (
              <span key={c} className="swatch" style={{ background: c }} title={c} />
            ))}
          </div>
        )}

        <div className="acts">
          <button className="btn ghost sm" disabled={busy} onClick={onRemeasure}>
            Re-measure
          </button>
          <button className="btn danger sm" disabled={busy} onClick={onDelete}>
            Delete
          </button>
        </div>
      </div>
    </div>
  )
}
