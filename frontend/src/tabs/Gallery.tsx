import { useState } from 'react'
import { api } from '../api'
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

  const kinds = useAsync(() => api.assetKinds(), [])
  const page = useAsync(
    () => api.assets({ kind: kind || undefined, q: q || undefined, limit: PAGE, offset }),
    [kind, q, offset],
  )

  async function remove(source: string, id: string) {
    setBusy(true)
    setError(null)
    try {
      await api.deleteAsset(source, id)
      page.reload()
      kinds.reload()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const total = page.data?.total ?? 0
  const shown = page.data?.items.length ?? 0

  return (
    <div className="card">
      <h2>Gallery</h2>
      <p className="hint">
        Generated concepts and finished spritesheets in one list. Deleting hides an
        item; the underlying job record is kept, because something2 may still be
        polling that id.
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
              {a.url && <img src={a.url} alt={a.title} loading="lazy" />}
            </div>
            <div className="meta">
              <div className="name" title={a.title}>
                {a.title}
              </div>
              <div className="why">
                <span className="tag neutral">{a.kind}</span>{' '}
                {a.created_at ? new Date(a.created_at).toLocaleString() : ''}
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
                  className="btn danger sm"
                  disabled={busy}
                  onClick={() => void remove(a.source, a.id)}
                >
                  Hide
                </button>
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
