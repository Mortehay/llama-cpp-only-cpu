import { useState } from 'react'
import { api, type Job, type Terrain } from '../api'
import { useAsync, useAuthedObjectUrl, usePoll } from '../hooks'

/**
 * World maps.
 *
 * A map is a biome painting quantised to a declared terrain set. The terrain
 * colours are not decoration: the painting is FORCED to them, so a tile id is a
 * lookup rather than a nearest-match, and the picture cannot disagree with the
 * walkable grid.
 *
 * Reusing an existing tile costs no GPU, which is the difference between a map
 * that builds in seconds and one that builds in minutes.
 */

const STARTER: Terrain[] = [
  { name: 'grass', color: '#4a7c3f', prompt: 'lush green grass' },
  { name: 'water', color: '#2850c8', prompt: 'shallow blue water' },
  { name: 'sand', color: '#c8be78', prompt: 'pale beach sand' },
  { name: 'stone', color: '#6e6e73', prompt: 'grey rocky stone' },
]

export default function Maps() {
  const tiles = useAsync(() => api.assets({ kind: 'tile', limit: 60 }), [])
  const refs = useAsync(() => api.references('map'), [])
  const profiles = useAsync(() => api.profiles(), [])
  const made = useAsync(() => api.maps(), [])

  const [name, setName] = useState('overworld')
  const [size, setSize] = useState(48)
  const [prompt, setPrompt] = useState('an island continent with a central mountain range')
  const [paintingFrom, setPaintingFrom] = useState('')
  const [tileW, setTileW] = useState(64)
  const [profile, setProfile] = useState('')
  const [terrains, setTerrains] = useState<Terrain[]>(STARTER)

  const [jobId, setJobId] = useState<string | null>(null)
  const [job, setJob] = useState<Job | null>(null)
  const [info, setInfo] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const picture = useAuthedObjectUrl(
    job?.status === 'done' ? `/api/jobs/${jobId}/sheet` : null,
  )

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
  usePoll(() => void refresh(), 3000, polling)

  function patch(i: number, field: keyof Terrain, value: string) {
    setTerrains((prev) =>
      prev.map((t, n) => (n === i ? { ...t, [field]: value || null } : t)),
    )
  }

  const toGenerate = terrains.filter((t) => !t.tile).length
  // Both axes scale with the SUM of the grid, so this grows fast. Shown before
  // submitting because a 256 map at 64px tiles is a 16k-pixel canvas.
  const picW = ((size + size) * tileW) / 2
  const picH = picW / 2

  async function submit() {
    setBusy(true)
    setError(null)
    setJob(null)
    setInfo(null)
    try {
      const res = await api.createMap({
        name: name.trim(),
        size,
        terrains,
        prompt: paintingFrom ? null : prompt.trim(),
        painting_from: paintingFrom || null,
        tile_w: tileW,
        style_profile: profile || null,
      })
      setJobId(res.job_id)
      setInfo(
        `${res.grid.w}×${res.grid.h} grid → ${res.picture.w}×${res.picture.h} picture · ` +
          `${res.tiles.reused} tile(s) reused, ${res.tiles.to_generate} to generate · ${res.projection}`,
      )
    } catch (e) {
      setBusy(false)
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <>
      <div className="card">
        <h2>World map</h2>
        <p className="hint">
          A low-res biome layout is painted, then <strong>forced to your terrain
          colours</strong> and composited from their tiles — so the picture and the
          walkable grid come from the same array and cannot disagree. Terrains that
          reuse an existing tile cost no GPU.
        </p>

        {error && <div className="note err">{error}</div>}
        {info && <div className="note">{info}</div>}

        <div className="row tight">
          <div style={{ flex: '2 1 200px' }}>
            <label htmlFor="m-name">Name</label>
            <input id="m-name" value={name} onChange={(e) => setName(e.target.value)} />
            <span className="muted">something2 asks for a map by name.</span>
          </div>
          <div style={{ flex: '1 1 110px' }}>
            <label htmlFor="m-size">Grid</label>
            <input
              id="m-size"
              type="number"
              min={8}
              max={256}
              value={size}
              onChange={(e) => setSize(Number(e.target.value))}
            />
          </div>
          <div style={{ flex: '1 1 110px' }}>
            <label htmlFor="m-tw">Tile width</label>
            <input
              id="m-tw"
              type="number"
              min={8}
              max={256}
              value={tileW}
              onChange={(e) => setTileW(Number(e.target.value))}
            />
          </div>
          <div style={{ flex: '1 1 160px' }}>
            <label htmlFor="m-prof">Style profile</label>
            <select id="m-prof" value={profile} onChange={(e) => setProfile(e.target.value)}>
              <option value="">assume 2:1</option>
              {profiles.data?.items.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.name}
                </option>
              ))}
            </select>
          </div>
        </div>

        <p className="muted" style={{ marginTop: 6 }}>
          Picture will be ~{Math.round(picW)}×{Math.round(picH)} px. Both axes scale
          with the sum of the grid, so doubling the grid quadruples the pixels.
        </p>

        <div className="row tight" style={{ marginTop: 12 }}>
          <div style={{ flex: '2 1 280px' }}>
            <label htmlFor="m-prompt">Biome brief</label>
            <input
              id="m-prompt"
              value={prompt}
              disabled={!!paintingFrom}
              onChange={(e) => setPrompt(e.target.value)}
            />
          </div>
          <div style={{ flex: '1 1 220px' }}>
            <label htmlFor="m-paint">…or use a reference as the layout</label>
            <select
              id="m-paint"
              value={paintingFrom}
              onChange={(e) => setPaintingFrom(e.target.value)}
            >
              <option value="">paint it from the brief</option>
              {refs.data?.items.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.label}
                </option>
              ))}
            </select>
            <span className="muted">
              Uses an uploaded map reference directly — no adapter needed.
            </span>
          </div>
        </div>
      </div>

      <div className="card">
        <h2>Terrain set</h2>
        <p className="hint">
          Colours must be far enough apart to tell apart — two that are not are
          rejected before any GPU is spent, because a map silently missing a terrain
          gives no hint which one it lost.
        </p>

        {terrains.map((t, i) => (
          <div className="row tight" key={i} style={{ marginBottom: 8 }}>
            <div style={{ flex: '1 1 120px' }}>
              <label>Name</label>
              <input value={t.name} onChange={(e) => patch(i, 'name', e.target.value)} />
            </div>
            <div style={{ flex: '0 0 70px' }}>
              <label>Colour</label>
              <input
                type="color"
                value={t.color}
                onChange={(e) => patch(i, 'color', e.target.value)}
              />
            </div>
            <div style={{ flex: '2 1 200px' }}>
              <label>Prompt</label>
              <input
                value={t.prompt ?? ''}
                disabled={!!t.tile}
                onChange={(e) => patch(i, 'prompt', e.target.value)}
              />
            </div>
            <div style={{ flex: '1 1 180px' }}>
              <label>…or reuse a tile</label>
              <select
                value={t.tile ?? ''}
                onChange={(e) => patch(i, 'tile', e.target.value)}
              >
                <option value="">generate one</option>
                {tiles.data?.items.map((a) => (
                  <option key={a.id} value={(a.url ?? '').split('/').pop()}>
                    {a.title.slice(0, 28)}
                  </option>
                ))}
              </select>
            </div>
            <div style={{ flex: '0 0 auto', alignSelf: 'flex-end' }}>
              <button
                className="btn ghost sm"
                disabled={terrains.length <= 2}
                onClick={() => setTerrains(terrains.filter((_, n) => n !== i))}
              >
                Remove
              </button>
            </div>
          </div>
        ))}

        <div className="row" style={{ marginTop: 10 }}>
          <button
            className="btn ghost"
            disabled={terrains.length >= 16}
            onClick={() =>
              setTerrains([...terrains, { name: '', color: '#888888', prompt: '' }])
            }
          >
            + Terrain
          </button>
          <button className="btn" disabled={busy} onClick={() => void submit()}>
            {busy ? 'Building…' : `Build map (${toGenerate} tile(s) to generate)`}
          </button>
        </div>
      </div>

      {job && (
        <div className="card">
          <h2>{job.status === 'done' ? 'Map' : 'Building'}</h2>
          {job.status !== 'done' && (
            <div className="note">
              {job.progress_pct}% — {job.progress_msg}
            </div>
          )}
          {job.status === 'failed' && <div className="note err">{job.error}</div>}
          {picture.error && <div className="note err">{picture.error}</div>}
          {job.status === 'done' && picture.url && (
            <div className="pic" style={{ overflow: 'auto' }}>
              <img src={picture.url} alt={name} style={{ imageRendering: 'pixelated' }} />
            </div>
          )}
          {job.status === 'done' && (
            <p className="muted">
              Tilemap JSON: <code>/api/maps/{jobId}</code>
            </p>
          )}
        </div>
      )}

      <div className="card">
        <h2>Maps</h2>
        {made.data && made.data.items.length === 0 && (
          <div className="empty">No maps yet.</div>
        )}
        <div className="grid">
          {made.data?.items.map((m) => (
            <div className="thumb" key={m.job_id}>
              <div className="meta">
                <div className="name">{m.name ?? '(unnamed)'}</div>
                <div className="why">
                  <span className="tag neutral">{m.status}</span> {m.size}×{m.size} ·{' '}
                  {m.terrains.join(', ')}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </>
  )
}
