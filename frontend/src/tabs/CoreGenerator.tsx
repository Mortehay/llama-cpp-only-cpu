import { useEffect, useRef, useState } from 'react'
import { api, imageUrl } from '../api'
import { useAsync, useDebounced, usePoll } from '../hooks'
import TaskQueue from '../components/TaskQueue'
import EditPanel from '../components/EditPanel'
import CropModal from '../components/CropModal'

/** Thumbnails per page. The server caps a page at 200. */
const PAGE = 24

/**
 * Step 1: one entity - character, creature, prop or item.
 *
 * Called "core" throughout the API and the database (`image_type = core`),
 * which is left alone: renaming a storage kind to match a UI label would mean
 * a migration and a break for something2, for no behavioural gain. The tab is
 * "Entity" because that is what people actually generate here.
 *
 * Everything downstream inherits from this image, so the isolation guard
 * matters: a landscape here produces a structurally perfect spritesheet of
 * scenery.
 */
export default function CoreGenerator() {
  const models = useAsync(() => api.coreModels(), [])

  const [prompt, setPrompt] = useState('')
  const [model, setModel] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [status, setStatus] = useState<string | null>(null)
  const [cropping, setCropping] = useState<{ id: number; url: string } | null>(null)

  // The list used to be whatever the server's hard-coded LIMIT 24 returned, so
  // entity 25 onwards existed on disk and in the database but could not be
  // reached from anywhere in the UI. It is searched and paged server side now;
  // `q` is debounced because every change to it is a query.
  const [q, setQ] = useState('')
  const [offset, setOffset] = useState(0)
  const search = useDebounced(q)
  const cores = useAsync(
    () => api.cores({ q: search || undefined, limit: PAGE, offset }),
    [search, offset],
  )

  const total = cores.data?.total ?? 0
  const shown = cores.data?.items.length ?? 0

  // Default to whichever model the roster marks default AND available; an
  // archived checkpoint is rendered but not selectable, because the failure it
  // used to produce arrived minutes later as "model failed to load".
  useEffect(() => {
    if (model || !models.data) return
    const d = models.data.models.find((m) => m.default && m.available)
      ?? models.data.models.find((m) => m.available)
    if (d) setModel(d.value)
  }, [models.data, model])

  // The entity lands in the DB when the worker finishes, so poll the list
  // rather than the task: it is the thing the user is actually waiting for.
  usePoll(() => cores.reload(), 4000, busy)

  // The unfiltered row count at the moment the job was queued. Unfiltered on
  // purpose: comparing against the *visible* count would make "is it done
  // yet?" depend on the search box, so a filtered list would never grow and
  // the button would stay on "Generating..." forever.
  const countAtQueue = useRef<number | null>(null)
  useEffect(() => {
    if (!busy || countAtQueue.current === null) return
    if (total > countAtQueue.current) {
      countAtQueue.current = null
      setBusy(false)
      setStatus('Entity ready.')
    }
  }, [total, busy])

  async function generate() {
    setBusy(true)
    setError(null)
    setStatus('Queued — the first run also loads the checkpoint, so allow a few minutes.')
    // Show the list the new entity will actually arrive in, rather than
    // leaving the user on page 3 of a filter it may not match.
    setQ('')
    setOffset(0)
    try {
      // `limit: 1` because only `total` is wanted: the baseline the poll above
      // compares against. A failure here costs the "Entity ready." message,
      // not the generation, so it degrades to null rather than throwing.
      countAtQueue.current = await api
        .cores({ limit: 1 })
        .then((r) => r.total)
        .catch(() => null)
      await api.generateCore(prompt.trim(), model)
      cores.reload()
    } catch (e) {
      setBusy(false)
      setStatus(null)
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <>
      <div className="card">
        <h2>Entity</h2>
        <p className="hint">
          One isolated entity - character, creature, prop or item - on a transparent
          background. Everything downstream is an edit of this image, so an unusable
          one costs an entire sheet: the spritesheet job checks isolation before
          spending GPU time on it. Ground tiles are a different shape and live on the
          Tiles tab.
        </p>

        {error && <div className="note err">{error}</div>}
        {status && <div className="note info">{status}</div>}

        <label htmlFor="prompt">Prompt</label>
        <textarea
          id="prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="green zombie, tattered clothes, pixel art, solid transparent background"
        />

        <div className="spacer" />

        <label htmlFor="model">Model</label>
        <select id="model" value={model} onChange={(e) => setModel(e.target.value)}>
          {models.data?.models.map((m) => (
            <option key={m.value} value={m.value} disabled={!m.available}>
              {m.label}
              {!m.available ? ` — unavailable${m.reason ? `: ${m.reason}` : ''}` : ''}
            </option>
          ))}
        </select>

        <div className="spacer" />
        <button className="btn" disabled={busy || !prompt.trim() || !model} onClick={() => void generate()}>
          {busy ? 'Generating…' : 'Generate entity'}
        </button>
      </div>

      <EditPanel onDone={() => cores.reload()} />

      <div className="card">
        <h2>Entities</h2>
        <p className="hint">
          Every entity generated so far, newest first. Pick one on the Spritesheet
          tab. Tiles are listed separately, on the Tiles tab.
        </p>

        <div className="row tight" style={{ marginBottom: 14 }}>
          <div style={{ flex: '2 1 240px' }}>
            <label htmlFor="core-q">Search prompt</label>
            <input
              id="core-q"
              type="text"
              list="core-terms"
              value={q}
              onChange={(e) => {
                setQ(e.target.value)
                setOffset(0)
              }}
              placeholder="zombie, tattered clothes…"
            />
            {/* The tags actually used in this project's prompts, most used
                first. Typing one letter offers them, so nobody has to recall
                the exact wording of a prompt from a fortnight ago. */}
            <datalist id="core-terms">
              {cores.data?.suggestions.map((s) => (
                <option key={s} value={s} />
              ))}
            </datalist>
          </div>
          <div style={{ flex: '0 0 auto' }}>
            <button className="btn ghost" onClick={cores.reload}>
              Refresh
            </button>
          </div>
          {q && (
            <div style={{ flex: '0 0 auto' }}>
              <button
                className="btn ghost"
                onClick={() => {
                  setQ('')
                  setOffset(0)
                }}
              >
                Clear
              </button>
            </div>
          )}
        </div>

        {cores.error && <div className="note err">{cores.error}</div>}
        {cores.loading && !cores.data && <div className="empty">Loading…</div>}
        {cores.data && total === 0 && (
          <div className="empty">
            {search ? `Nothing matches "${search}".` : 'No entities yet.'}
          </div>
        )}

        <div className="grid">
          {cores.data?.items.map((c) => (
            <div className="thumb" key={c.id}>
              <div className="pic">
                <img src={imageUrl(c.file_path)} alt={c.prompt} loading="lazy" />
              </div>
              <div className="meta">
                <div className="name" title={c.prompt}>
                  {c.prompt}
                </div>
                {c.created_at && (
                  <div className="why">{new Date(c.created_at).toLocaleString()}</div>
                )}
                <div className="acts">
                  <button
                    className="btn ghost sm"
                    onClick={() => setCropping({ id: c.id, url: imageUrl(c.file_path) })}
                  >
                    Crop
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>

        {total > 0 && (
          <div className="row" style={{ marginTop: 16, justifyContent: 'space-between' }}>
            <button
              className="btn ghost"
              disabled={offset === 0}
              onClick={() => setOffset(Math.max(0, offset - PAGE))}
            >
              ← Previous
            </button>
            <span className="muted" style={{ textAlign: 'center' }}>
              {offset + 1}-{offset + shown} of {total}
              {search ? ` matching "${search}"` : ''}
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

      <TaskQueue />

      {cropping && (
        <CropModal
          sourceId={cropping.id}
          url={cropping.url}
          onClose={() => setCropping(null)}
          onSaved={() => {
            cores.reload()
            setStatus('Crop saved as a new entity.')
          }}
        />
      )}
    </>
  )
}
