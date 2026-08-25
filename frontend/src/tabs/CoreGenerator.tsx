import { useEffect, useState } from 'react'
import { api, imageUrl } from '../api'
import { useAsync, usePoll } from '../hooks'

/**
 * Step 1: one character concept.
 *
 * Everything downstream inherits from this image, so the concept guard matters:
 * a landscape here produces a structurally perfect spritesheet of scenery.
 */
export default function CoreGenerator() {
  const models = useAsync(() => api.coreModels(), [])
  const cores = useAsync(() => api.cores(), [])

  const [prompt, setPrompt] = useState('')
  const [model, setModel] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [status, setStatus] = useState<string | null>(null)

  // Default to whichever model the roster marks default AND available; an
  // archived checkpoint is rendered but not selectable, because the failure it
  // used to produce arrived minutes later as "model failed to load".
  useEffect(() => {
    if (model || !models.data) return
    const d = models.data.models.find((m) => m.default && m.available)
      ?? models.data.models.find((m) => m.available)
    if (d) setModel(d.value)
  }, [models.data, model])

  // The concept lands in the DB when the worker finishes, so poll the list
  // rather than the task: it is the thing the user is actually waiting for.
  usePoll(() => cores.reload(), 4000, busy)

  const initialCount = cores.data?.length ?? 0
  useEffect(() => {
    if (busy && (cores.data?.length ?? 0) > initialCount) {
      setBusy(false)
      setStatus('Concept ready.')
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cores.data])

  async function generate() {
    setBusy(true)
    setError(null)
    setStatus('Queued — the first run also loads the checkpoint, so allow a few minutes.')
    try {
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
        <h2>Core character</h2>
        <p className="hint">
          One isolated character on a transparent background. Everything downstream is
          an edit of this image, so an unusable concept costs an entire sheet — the
          spritesheet job checks isolation before spending GPU time on it.
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
          {busy ? 'Generating…' : 'Generate concept'}
        </button>
      </div>

      <div className="card">
        <h2>Concepts</h2>
        <p className="hint">The most recent concepts, newest first. Pick one on the Spritesheet tab.</p>
        {cores.error && <div className="note err">{cores.error}</div>}
        {cores.data?.length === 0 && <div className="empty">No concepts yet.</div>}
        <div className="grid">
          {cores.data?.map((c) => (
            <div className="thumb" key={c.id}>
              <div className="pic">
                <img src={imageUrl(c.file_path)} alt={c.prompt} loading="lazy" />
              </div>
              <div className="meta">
                <div className="name" title={c.prompt}>
                  {c.prompt}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </>
  )
}
