import { useEffect, useState } from 'react'
import { api, type Job } from '../api'
import { useAsync, useAuthedObjectUrl, usePoll } from '../hooks'

/**
 * Ground tiles.
 *
 * A tile is a SHAPE, not a small sprite: the rhombus has to tessellate exactly
 * or the ground shows seams, so the model paints texture and the outline is
 * applied afterwards at the projection the world actually uses.
 */
export default function Tiles() {
  const profiles = useAsync(() => api.profiles(), [])
  const models = useAsync(() => api.coreModels(), [])
  // Tiles keep their own list rather than sharing the entity list: they are a
  // different shape with a different job kind, and mixing them made it hard to
  // find either.
  const made = useAsync(() => api.assets({ kind: 'tile', limit: 48 }), [])

  const [prompt, setPrompt] = useState('lush green grass')
  const [profile, setProfile] = useState('')
  const [tileW, setTileW] = useState(64)
  const [colors, setColors] = useState(16)
  const [seed, setSeed] = useState(0)
  const [model, setModel] = useState('')

  const [jobId, setJobId] = useState<string | null>(null)
  const [job, setJob] = useState<Job | null>(null)

  // `/api/jobs/{id}/sheet` requires a key, and neither `<img src>` nor
  // `<a href>` carries one. Fetch with the token and point at a blob instead.
  const tile = useAuthedObjectUrl(
    job?.status === 'done' ? `/api/jobs/${job.id}/sheet` : null,
  )
  const [info, setInfo] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const chosen = profiles.data?.items.find((p) => p.name === profile)
  const ratio = chosen?.projection_ratio ?? 2.0
  const previewH = Math.max(2, Math.round(tileW / ratio))

  const polling = !!jobId && (!job || !['done', 'failed', 'cancelled'].includes(job.status))

  const refresh = async () => {
    if (!jobId) return
    try {
      const j = await api.job(jobId)
      setJob(j)
      if (['done', 'failed', 'cancelled'].includes(j.status)) {
        setBusy(false)
        made.reload()
      }
    } catch (e) {
      setJobId(null)
      setBusy(false)
      setError(e instanceof Error ? e.message : String(e))
    }
  }
  useEffect(() => {
    void refresh()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId])
  usePoll(() => void refresh(), 3000, polling)

  async function submit() {
    setBusy(true)
    setError(null)
    setJob(null)
    try {
      const res = await api.createTile({
        prompt: prompt.trim(),
        style_profile: profile || null,
        tile_w: tileW,
        colors,
        seed,
        llm_name: model || null,
      })
      setJobId(res.job_id)
      setInfo(`${res.tile.w}×${res.tile.h} at ${res.tile.ratio}:1 — ${res.projection}`)
    } catch (e) {
      setBusy(false)
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <>
      <div className="card">
        <h2>Ground tile</h2>
        <p className="hint">
          The projection comes from a style profile when you have one — a reference
          tile is a direct readout of your world's camera angle. Without one this
          assumes classic 2:1, which looks fine alone and wrong the moment it is tiled.
        </p>

        {error && <div className="note err">{error}</div>}
        {info && <div className="note info">{info}</div>}

        <label htmlFor="tile-prompt">Material</label>
        <input
          id="tile-prompt"
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="lush green grass, cracked stone, desert sand"
        />

        <div className="spacer" />

        <div className="row">
          <div>
            <label htmlFor="tile-profile">Style profile</label>
            <select
              id="tile-profile"
              value={profile}
              onChange={(e) => setProfile(e.target.value)}
            >
              <option value="">None — assume 2:1</option>
              {profiles.data?.items
                .filter((p) => p.projection_ratio)
                .map((p) => (
                  <option key={p.id} value={p.name}>
                    {p.name} — {p.projection_ratio}:1
                  </option>
                ))}
            </select>
          </div>
          <div>
            <label htmlFor="tile-w">Tile width</label>
            <input
              id="tile-w"
              type="number"
              min={8}
              max={512}
              value={tileW}
              onChange={(e) => setTileW(Number(e.target.value) || 64)}
            />
          </div>
          <div>
            <label htmlFor="tile-colors">Colours</label>
            <input
              id="tile-colors"
              type="number"
              min={2}
              max={64}
              value={colors}
              onChange={(e) => setColors(Number(e.target.value) || 16)}
            />
          </div>
          <div>
            <label htmlFor="tile-seed">Seed</label>
            <input
              id="tile-seed"
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value) || 0)}
            />
          </div>
        </div>

        <div className="spacer" />
        <label htmlFor="tile-model">Model</label>
        <select id="tile-model" value={model} onChange={(e) => setModel(e.target.value)}>
          <option value="">Default (SDXL + pixel-art LoRA)</option>
          {models.data?.models
            .filter((m) => m.available)
            .map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
        </select>
        <div className="muted" style={{ marginTop: 6 }}>
          If you have trained an adapter on <strong>Reference · Tile</strong>, pick it
          here — and put its trigger word in the material box above.
        </div>

        <div className="note info" style={{ marginTop: 14 }}>
          Will produce <strong>{tileW}×{previewH}</strong> at {ratio}:1
          {chosen ? ' (measured from your references)' : ' (assumed)'}.
        </div>

        <button className="btn" disabled={busy || !prompt.trim()} onClick={() => void submit()}>
          {busy ? 'Queued…' : 'Queue tile'}
        </button>
      </div>

      {job && (
        <div className="card">
          <h2>
            Tile job <code>{job.id.slice(0, 8)}</code> · {job.status}
          </h2>
          <div className="muted">{job.progress_msg ?? ''}</div>
          <div className="bar">
            <i style={{ width: `${job.progress_pct}%` }} />
          </div>
          {job.error && <div className="note err" style={{ marginTop: 12 }}>{job.error}</div>}
          {job.status === 'done' && (
            <div style={{ marginTop: 14 }}>
              <div className="thumb" style={{ maxWidth: 320 }}>
                <div className="pic">
                  {/* Scaled up so a 64x32 tile is actually visible; the CSS
                      keeps it pixelated rather than smoothing it. */}
                  {tile.url && (
                    <img
                      src={tile.url}
                      alt="tile"
                      style={{ width: tileW * 4, imageRendering: 'pixelated' }}
                    />
                  )}
                  {tile.error && <div className="note err">{tile.error}</div>}
                </div>
              </div>
              <div className="acts" style={{ marginTop: 10 }}>
                {tile.url && (
                  <a
                    className="btn ghost sm"
                    href={tile.url}
                    download={`tile-${job.id}.png`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Download PNG
                  </a>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      <div className="card">
        <h2>Tiles</h2>
        <p className="hint">
          Every tile generated so far, kept separate from entities: a tile is a
          different shape with a different job kind, and one mixed list made both
          harder to find.
        </p>
        {made.error && <div className="note err">{made.error}</div>}
        {made.data?.items.length === 0 && <div className="empty">No tiles yet.</div>}
        <div className="grid">
          {made.data?.items.map((t) => (
            <div className="thumb" key={`${t.source}-${t.id}`}>
              <div className="pic">
                {t.url && (
                  <img
                    src={t.url}
                    alt={t.title}
                    loading="lazy"
                    style={{ width: '80%', imageRendering: 'pixelated' }}
                  />
                )}
              </div>
              <div className="meta">
                <div className="name" title={t.title}>
                  {t.title}
                </div>
                <div className="acts">
                  {t.url && (
                    <a className="btn ghost sm" href={t.url} target="_blank" rel="noreferrer">
                      Open
                    </a>
                  )}
                  <button
                    className="btn danger sm"
                    onClick={() =>
                      void api.deleteAsset(t.source, t.id).then(() => made.reload())
                    }
                  >
                    Hide
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </>
  )
}
