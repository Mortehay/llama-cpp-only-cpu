import { useState } from 'react'
import { api, isAudioKind } from '../api'
import type { ReferenceKind } from '../api'
import { useAsync } from '../hooks'

const PAGE = 48

/**
 * Everything the system has produced.
 *
 * The old gallery read `sprite_images` only. On 2026-08-25 that meant it showed
 * 2 rows while 13 finished spritesheets sat in `jobs`, invisible. This reads
 * `/api/assets`, which unions both.
 */
export default function Gallery() {
  const [kind, setKind] = useState('')
  const [q, setQ] = useState('')
  const [offset, setOffset] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  // Cards whose full prompt is showing. A set rather than one id, because
  // comparing two prompts means having both open at once - which is the whole
  // reason a truncated title was not good enough.
  const [open, setOpen] = useState<Set<string>>(new Set())

  function toggle(key: string) {
    setOpen((prev) => {
      const next = new Set(prev)
      if (!next.delete(key)) next.add(key)
      return next
    })
  }

  const kinds = useAsync(() => api.assetKinds(), [])
  const page = useAsync(
    () => api.assets({ kind: kind || undefined, q: q || undefined, limit: PAGE, offset }),
    [kind, q, offset],
  )

  async function remove(source: string, id: string, purge = false) {
    setBusy(true)
    setError(null)
    try {
      await api.deleteAsset(source, id, purge)
      page.reload()
      kinds.reload()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  function confirmDelete(a: { source: string; id: string; title: string }) {
    const ok = window.confirm(
      `Delete the file for:\n\n${a.title}\n\n` +
        `This removes the image from disk and cannot be undone. The job record ` +
        `is kept, so something2 still resolves the id - but it will no longer ` +
        `be able to fetch the image.\n\nUse Hide instead to keep the file.`,
    )
    if (ok) void remove(a.source, a.id, true)
  }

  const total = page.data?.total ?? 0
  const shown = page.data?.items.length ?? 0

  return (
    <div className="card">
      <h2>Gallery</h2>
      <p className="hint">
        Generated concepts and finished spritesheets in one list. <strong>Click a
        title</strong> to read the whole prompt. <strong>Hide</strong> drops an item
        from this list and keeps its file; <strong>Delete</strong> also removes the
        file from disk. The job record survives either way, because something2 may
        still be polling that id.
      </p>

      {error && <div className="note err">{error}</div>}

      <div className="row tight" style={{ marginBottom: 14 }}>
        <div style={{ flex: '1 1 160px' }}>
          <label htmlFor="g-kind">Kind</label>
          <select
            id="g-kind"
            value={kind}
            onChange={(e) => {
              setKind(e.target.value)
              setOffset(0)
            }}
          >
            <option value="">All</option>
            {kinds.data?.groups.map((g) => (
              <option key={`${g.source}-${g.kind}`} value={g.kind}>
                {g.kind} ({g.n})
              </option>
            ))}
          </select>
        </div>
        <div style={{ flex: '2 1 240px' }}>
          <label htmlFor="g-q">Search prompt</label>
          <input
            id="g-q"
            type="text"
            value={q}
            onChange={(e) => {
              setQ(e.target.value)
              setOffset(0)
            }}
            placeholder="zombie, knight…"
          />
        </div>
        <div style={{ flex: '0 0 auto' }}>
          <button className="btn ghost" onClick={page.reload}>
            Refresh
          </button>
        </div>
      </div>

      {page.error && <div className="note err">{page.error}</div>}
      {page.loading && !page.data && <div className="empty">Loading…</div>}
      {page.data && total === 0 && <div className="empty">Nothing generated yet.</div>}

      <div className="grid">
        {page.data?.items.map((a) => (
          <div className="thumb" key={`${a.source}-${a.id}`}>
            <div className="pic">
              {a.url && isAudioKind(a.kind) ? (
                <audio controls preload="none" src={a.url} style={{ width: '100%' }} />
              ) : (
                a.url && <img src={a.url} alt={a.title} loading="lazy" />
              )}
            </div>
            <div className="meta">
              <button
                type="button"
                className={`name ${open.has(`${a.source}-${a.id}`) ? 'open' : ''}`}
                title={open.has(`${a.source}-${a.id}`) ? 'Collapse' : 'Show the full prompt'}
                onClick={() => toggle(`${a.source}-${a.id}`)}
              >
                {a.title}
              </button>
              <div className="why">
                <span className="tag neutral">{a.kind}</span>{' '}
                {a.created_at ? new Date(a.created_at).toLocaleString() : ''}
                {open.has(`${a.source}-${a.id}`) && a.model && (
                  <>
                    <br />
                    {a.model}
                  </>
                )}
              </div>
              <div className="acts">
                {a.url && (
                  <a className="btn ghost sm" href={a.url} target="_blank" rel="noreferrer">
                    Open
                  </a>
                )}
                {a.atlas_url && (
                  <a className="btn ghost sm" href={a.atlas_url} target="_blank" rel="noreferrer">
                    Atlas
                  </a>
                )}
                <button
                  className="btn ghost sm"
                  disabled={busy}
                  title="Remove from the gallery. The file stays on disk."
                  onClick={() => void remove(a.source, a.id)}
                >
                  Hide
                </button>
                <button
                  className="btn danger sm"
                  disabled={busy}
                  title="Remove from the gallery AND delete the file from disk."
                  onClick={() => confirmDelete(a)}
                >
                  Delete
                </button>
                {a.url && <Promote source={a.source} id={a.id} />}
              </div>
            </div>
          </div>
        ))}
      </div>

      {total > PAGE && (
        <div className="row" style={{ marginTop: 16, justifyContent: 'space-between' }}>
          <button
            className="btn ghost"
            disabled={offset === 0}
            onClick={() => setOffset(Math.max(0, offset - PAGE))}
          >
            ← Previous
          </button>
          <span className="muted" style={{ textAlign: 'center' }}>
            {offset + 1}–{offset + shown} of {total}
          </span>
          <button
            className="btn ghost"
            disabled={offset + PAGE >= total}
            onClick={() => setOffset(offset + PAGE)}
          >
            Next →
          </button>
        </div>
      )}
    </div>
  )
}

/**
 * File this image into a reference set — one image, one deliberate click.
 *
 * DELIBERATELY NOT A BULK ACTION, and not automatic on generation. References
 * are what style profiles are derived from and what LoRA training reads, so
 * admitting model output wholesale would make the model its own yardstick and
 * nothing here would say so. `.ai/decisions/0009` and the entity-cutout
 * findings are both records of that set being the fragile part of this system.
 *
 * The verdict is shown inline rather than swallowed, because "promoted" and
 * "promoted and it measures as unusable" are different outcomes and only one
 * of them is worth repeating.
 */
function Promote({ source, id }: { source: string; id: string }) {
  const [kind, setKind] = useState<ReferenceKind | ''>('')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<string | null>(null)

  async function go(k: ReferenceKind) {
    setBusy(true)
    setResult(null)
    try {
      const r = await api.promoteAsset({ source, id, kind: k })
      setResult(
        r.usable === null
          ? 'filed — could not be measured'
          : r.usable
            ? `filed as ${k} · usable`
            : `filed as ${k} · NOT usable: ${r.why}`,
      )
    } catch (e) {
      setResult(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
      setKind('')
    }
  }

  if (result) {
    return (
      <span className="muted" style={{ fontSize: 11 }} title={result}>
        {result.length > 44 ? `${result.slice(0, 44)}…` : result}
      </span>
    )
  }

  return (
    <select
      value={kind}
      disabled={busy}
      title="Copy this image into a reference set and measure it"
      style={{ fontSize: 11, padding: '4px 6px', flex: '0 0 auto' }}
      onChange={(e) => {
        const v = e.target.value as ReferenceKind | ''
        setKind(v)
        if (v) void go(v)
      }}
    >
      <option value="">→ reference…</option>
      <option value="core">as core</option>
      <option value="sprite">as sprite</option>
      <option value="tile">as tile</option>
      <option value="map">as map</option>
    </select>
  )
}
