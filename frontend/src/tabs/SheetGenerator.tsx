import { useEffect, useMemo, useState } from 'react'
import { api, imageUrl, type Core, type Job } from '../api'
import { useAsync, useAuthedObjectUrl, useDebounced, usePoll } from '../hooks'

/** Measured on this card in ADR 0005: ~33 s per cell, plus model load. */
const SECONDS_PER_CELL = 33
const LOAD_OVERHEAD_S = 360

const DIRECTIONS = [
  { id: 's', label: 'S (front)' },
  { id: 'se', label: 'SE' },
  { id: 'e', label: 'E (side)' },
  { id: 'ne', label: 'NE' },
  { id: 'n', label: 'N (back)' },
  { id: 'nw', label: 'NW' },
  { id: 'w', label: 'W (side)' },
  { id: 'sw', label: 'SW' },
]

const JOB_KEY = 'sheetJobId'

/** Concepts in the picker at once. Enough to browse, few enough to render. */
const PICKER_PAGE = 48

export default function SheetGenerator() {
  // Searchable rather than "the 48 newest": once a project has a few dozen
  // concepts, the one worth turning into a sheet is rarely the latest.
  const [coreQ, setCoreQ] = useState('')
  const coreSearch = useDebounced(coreQ)
  const cores = useAsync(
    () => api.cores({ q: coreSearch || undefined, limit: PICKER_PAGE }),
    [coreSearch],
  )
  const catalog = useAsync(() => api.actionCatalog(), [])
  const profiles = useAsync(() => api.profiles(), [])

  // The whole concept, not its id: the selection has to survive a search that
  // filters it out of the visible grid, or submit would report "choose a
  // concept first" about one that is plainly still highlighted.
  const [core, setCore] = useState<Core | null>(null)
  const coreId = core?.id ?? null
  const [actions, setActions] = useState<string[]>(['walk'])
  const [directions, setDirections] = useState<string[]>(['s'])
  const [frames, setFrames] = useState(4)
  const [cell, setCell] = useState('48x64')
  const [colors, setColors] = useState(24)
  const [styleProfile, setStyleProfile] = useState('')

  const [jobId, setJobId] = useState<string | null>(() => {
    try {
      return localStorage.getItem(JOB_KEY)
    } catch {
      return null
    }
  })
  const [job, setJob] = useState<Job | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  // The sheet and atlas routes require a key, and a browser sends no auth
  // header for `<img src>` or `<a href>`. Fetch them with the token instead and
  // point at blobs. Null until the job is done, so nothing is requested early.
  const done = job?.status === 'done'
  const sheet = useAuthedObjectUrl(done ? `/api/jobs/${job.job_id}/sheet` : null)
  const atlas = useAuthedObjectUrl(done ? `/api/jobs/${job.job_id}/atlas` : null)

  // The frame ceiling is a MINIMUM across the selected actions - one 4-pose
  // action caps the whole sheet, because a sheet cannot have ragged rows.
  const ceiling = useMemo(() => {
    if (!catalog.data || actions.length === 0) return 4
    return Math.min(
      ...actions.map(
        (a) => catalog.data!.actions.find((x) => x.id === a)?.max_frames ?? 4,
      ),
    )
  }, [catalog.data, actions])

  useEffect(() => {
    if (frames > ceiling) setFrames(ceiling)
  }, [ceiling, frames])

  const cells = actions.length * Math.max(directions.length, 1) * frames
  const minutes = cells ? Math.round((cells * SECONDS_PER_CELL + LOAD_OVERHEAD_S) / 60) : 0

  const polling = !!jobId && (!job || !['done', 'failed', 'cancelled'].includes(job.status))

  const refresh = async () => {
    if (!jobId) return
    try {
      const j = await api.job(jobId)
      setJob(j)
      if (['done', 'failed', 'cancelled'].includes(j.status)) {
        setBusy(false)
        try {
          localStorage.removeItem(JOB_KEY)
        } catch { /* ignore */ }
      }
    } catch (e) {
      // A 404 means the job is gone; stop chasing it.
      setJobId(null)
      setBusy(false)
      setError(e instanceof Error ? e.message : String(e))
      try {
        localStorage.removeItem(JOB_KEY)
      } catch { /* ignore */ }
    }
  }

  useEffect(() => {
    void refresh()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId])
  usePoll(() => void refresh(), 3000, polling)

  async function submit() {
    if (!core) {
      setError('Choose a concept first.')
      return
    }
    setBusy(true)
    setError(null)
    setJob(null)
    try {
      const res = await api.createJob({
        concept_image: core.file_path.split('/').pop(),
        actions,
        directions,
        frames,
        cell,
        colors,
        style_profile: styleProfile || null,
      })
      setJobId(res.job_id)
      try {
        localStorage.setItem(JOB_KEY, res.job_id)
      } catch { /* ignore */ }
    } catch (e) {
      setBusy(false)
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  function toggle(list: string[], set: (v: string[]) => void, id: string) {
    set(list.includes(id) ? list.filter((x) => x !== id) : [...list, id])
  }

  return (
    <>
      <div className="card">
        <h2>Spritesheet from a concept</h2>
        <p className="hint">
          Queued, not synchronous — a full character is hours of GPU time. You get a job
          id immediately and can close this page; the job survives it.
        </p>

        {error && <div className="note err">{error}</div>}

        <label htmlFor="sheet-core-q">Concept</label>
        <div className="row tight" style={{ marginBottom: 12 }}>
          <div style={{ flex: '2 1 240px' }}>
            <input
              id="sheet-core-q"
              type="text"
              list="sheet-core-terms"
              value={coreQ}
              onChange={(e) => setCoreQ(e.target.value)}
              placeholder="Search concepts: zombie, knight…"
            />
            <datalist id="sheet-core-terms">
              {cores.data?.suggestions.map((s) => (
                <option key={s} value={s} />
              ))}
            </datalist>
          </div>
          <div style={{ flex: '0 0 auto', alignSelf: 'center' }}>
            <span className="muted">
              {(cores.data?.total ?? 0) > PICKER_PAGE
                ? `${cores.data?.items.length} of ${cores.data?.total} — search to narrow`
                : `${cores.data?.total ?? 0} concept${cores.data?.total === 1 ? '' : 's'}`}
            </span>
          </div>
        </div>
        {cores.data?.total === 0 && (
          <div className="empty">
            {coreSearch
              ? `Nothing matches "${coreSearch}".`
              : 'No concepts yet — generate one on the Entity tab.'}
          </div>
        )}
        <div className="grid" style={{ marginBottom: 16 }}>
          {cores.data?.items.map((c) => (
            <button
              key={c.id}
              className="thumb"
              onClick={() => setCore(c)}
              style={{
                cursor: 'pointer',
                padding: 0,
                borderColor: coreId === c.id ? 'var(--accent)' : undefined,
                boxShadow: coreId === c.id ? '0 0 0 2px var(--accent)' : undefined,
              }}
            >
              <div className="pic">
                <img src={imageUrl(c.file_path)} alt={c.prompt} loading="lazy" />
              </div>
              <div className="meta">
                <div className="name" title={c.prompt}>
                  {c.prompt}
                </div>
              </div>
            </button>
          ))}
        </div>

        {/* The selection survives a search that hides it, so say so - an
            invisible highlight is indistinguishable from nothing selected. */}
        {core && !cores.data?.items.some((c) => c.id === core.id) && (
          <div className="note info" style={{ marginBottom: 16 }}>
            Selected (not in the current search): {core.prompt}
          </div>
        )}

        <label>Actions</label>
        <div className="row tight" style={{ marginBottom: 14 }}>
          {catalog.data?.actions.map((a) => (
            <label className="check" key={a.id} style={{ flex: '0 0 auto' }}>
              <input
                type="checkbox"
                checked={actions.includes(a.id)}
                onChange={() => toggle(actions, setActions, a.id)}
              />
              {a.label} <span className="muted">({a.max_frames})</span>
            </label>
          ))}
        </div>

        <label>Directions</label>
        <div className="row tight" style={{ marginBottom: 14 }}>
          {DIRECTIONS.map((d) => (
            <label className="check" key={d.id} style={{ flex: '0 0 auto' }}>
              <input
                type="checkbox"
                checked={directions.includes(d.id)}
                onChange={() => toggle(directions, setDirections, d.id)}
              />
              {d.label}
            </label>
          ))}
        </div>

        <div className="row">
          <div>
            <label htmlFor="frames">Frames per action (max {ceiling})</label>
            <input
              id="frames"
              type="number"
              min={1}
              max={ceiling}
              value={frames}
              onChange={(e) => setFrames(Math.min(ceiling, Number(e.target.value) || 1))}
            />
          </div>
          <div>
            <label htmlFor="cell">Cell size</label>
            <input id="cell" type="text" value={cell} onChange={(e) => setCell(e.target.value)} />
          </div>
          <div>
            <label htmlFor="colors">Palette colours</label>
            <input
              id="colors"
              type="number"
              min={2}
              max={64}
              value={colors}
              onChange={(e) => setColors(Number(e.target.value) || 24)}
            />
          </div>
        </div>

        <div className="spacer" />
        <label htmlFor="style-profile">Style profile</label>
        <select
          id="style-profile"
          value={styleProfile}
          onChange={(e) => setStyleProfile(e.target.value)}
        >
          <option value="">None — use the settings above</option>
          {profiles.data?.items.map((p) => (
            <option key={p.id} value={p.name}>
              {p.name}
              {p.elevation ? ` — camera ${p.elevation}` : ''}
              {p.cell_w && p.cell_h ? `, ${p.cell_w}×${p.cell_h}` : ''}
            </option>
          ))}
        </select>
        {styleProfile && (
          <div className="note info" style={{ marginTop: 10 }}>
            The profile's measured camera, cell size and colour count override the
            fields above. The camera is the important one: without a profile the
            build uses <code>eye</code> (0°), which was never measured against your
            game's projection.
          </div>
        )}

        <div className="note info" style={{ marginTop: 14 }}>
          {cells === 0 ? (
            'Select at least one action.'
          ) : (
            <>
              <strong>{cells} cells</strong> — roughly {minutes} min on this card.
              {directions.length === 0 && ' No direction ticked, so front only.'}
              {frames > ceiling && ' Frames capped by the shortest selected action.'}
            </>
          )}
        </div>

        <button className="btn" disabled={busy || !coreId || !actions.length} onClick={() => void submit()}>
          {busy ? 'Queued…' : 'Queue spritesheet'}
        </button>
      </div>

      {job && (
        <div className="card">
          <h2>
            Job <code>{job.job_id.slice(0, 8)}</code> · {job.status}
          </h2>
          <div className="muted">
            {job.stage ? `${job.stage} — ` : ''}
            {job.progress_msg ?? ''}
          </div>
          <div className="bar">
            <i style={{ width: `${job.progress_pct}%` }} />
          </div>
          {job.error && <div className="note err" style={{ marginTop: 12 }}>{job.error}</div>}
          {job.status === 'done' && (
            <div style={{ marginTop: 14 }}>
              <div className="thumb" style={{ maxWidth: 520 }}>
                <div className="pic">
                  {sheet.url && <img src={sheet.url} alt="spritesheet" />}
                  {sheet.error && <div className="note err">{sheet.error}</div>}
                </div>
              </div>
              <div className="acts" style={{ marginTop: 10 }}>
                {sheet.url && (
                  <a
                    className="btn ghost sm"
                    href={sheet.url}
                    download={`sheet-${job.job_id}.png`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Sheet PNG
                  </a>
                )}
                {atlas.url && (
                  <a
                    className="btn ghost sm"
                    href={atlas.url}
                    download={`atlas-${job.job_id}.json`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Atlas JSON
                  </a>
                )}
              </div>
            </div>
          )}
          {polling && (
            <button
              className="btn danger sm"
              style={{ marginTop: 12 }}
              onClick={() => void api.cancelJob(job.job_id).then(refresh)}
            >
              Cancel
            </button>
          )}
        </div>
      )}
    </>
  )
}
