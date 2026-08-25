import { useState } from 'react'
import { api } from '../api'
import { useAsync, usePoll } from '../hooks'

/**
 * Train a style LoRA on the uploaded references.
 *
 * The button is disabled with a *reason* rather than offering an action that
 * will 400: not enough usable references, or the single GPU already busy.
 */
const KINDS = [
  { id: 'core', label: 'Reference · Core' },
  { id: 'sprite', label: 'Reference · Sprite' },
  { id: 'tile', label: 'Reference · Tile' },
] as const

export default function Training() {
  const [kinds, setKinds] = useState<string[]>(['sprite', 'core'])
  const ready = useAsync(() => api.trainingReadiness(kinds), [kinds.join(',')])
  const runs = useAsync(() => api.training(), [])
  const profiles = useAsync(() => api.profiles(), [])

  const [name, setName] = useState('something2')
  const [profile, setProfile] = useState('')
  const [steps, setSteps] = useState(1000)
  const [rank, setRank] = useState(32)
  const [resolution, setResolution] = useState(1024)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [started, setStarted] = useState<string | null>(null)

  const active = runs.data?.items.some((r) => r.status === 'running' || r.status === 'queued')
  usePoll(() => {
    runs.reload()
    ready.reload()
  }, 5000, !!active)

  async function start() {
    setBusy(true)
    setError(null)
    try {
      const res = await api.startTraining({
        name: name.trim(),
        profile: profile || null,
        steps,
        rank,
        resolution,
        kinds,
      })
      setStarted(
        `Queued on ${res.dataset_size} references (${res.kinds.join(', ')}). ` +
          `Prompt with ${res.trigger} once it finishes.` +
          (res.note ? ` ${res.note}` : ''),
      )
      runs.reload()
      ready.reload()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <div className="card">
        <h2>Train a style LoRA</h2>
        <p className="hint">
          Counts below are TRAINABLE references, not measurable ones. A JPEG
          reference board cannot yield an exact palette but trains a style perfectly
          well - gating training on the measurement rules rejected 100 of 106 real
          sprites here before that was split apart.
          Trains SDXL at native 1024, measured at 47% of this card.
        </p>

        {error && <div className="note err">{error}</div>}
        {started && <div className="note ok">{started}</div>}

        {ready.data && (
          <div className={`note ${ready.data.ready ? 'ok' : 'warn'}`}>
            {ready.data.ready
              ? `Ready —  trainable references.`
              : `Not ready: ${ready.data.why}.`}
          </div>
        )}

        <label>Train on which reference tabs</label>
        <div className="row tight" style={{ marginBottom: 6 }}>
          {KINDS.map((k) => (
            <label className="check" key={k.id} style={{ flex: '0 0 auto' }}>
              <input
                type="checkbox"
                checked={kinds.includes(k.id)}
                onChange={(e) =>
                  setKinds((cur) =>
                    e.target.checked ? [...cur, k.id] : cur.filter((x) => x !== k.id),
                  )
                }
              />
              {k.label}{' '}
              <span className="muted">({ready.data?.per_kind?.[k.id] ?? 0} trainable)</span>
            </label>
          ))}
        </div>
        {kinds.includes('tile') && kinds.length > 1 && (
          <div className="note warn">
            Tiles and characters in one adapter bind a single trigger to both, and the
            model cannot tell which you meant — usually characters with terrain texture
            in them. Two adapters (one <code>core+sprite</code>, one <code>tile</code>)
            give sharper results. This will still run if you want it.
          </div>
        )}

        <div className="spacer" />

        <div className="row">
          <div>
            <label htmlFor="t-name">Adapter name</label>
            <input id="t-name" type="text" value={name} onChange={(e) => setName(e.target.value)} />
          </div>
          <div>
            <label htmlFor="t-profile">Attach to profile</label>
            <select id="t-profile" value={profile} onChange={(e) => setProfile(e.target.value)}>
              <option value="">None</option>
              {profiles.data?.items.map((p) => (
                <option key={p.id} value={p.name}>
                  {p.name}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="spacer" />

        <div className="row">
          <div>
            <label htmlFor="t-steps">Steps</label>
            <input
              id="t-steps"
              type="number"
              min={50}
              max={20000}
              value={steps}
              onChange={(e) => setSteps(Number(e.target.value) || 1000)}
            />
          </div>
          <div>
            <label htmlFor="t-rank">Rank</label>
            <input
              id="t-rank"
              type="number"
              min={4}
              max={128}
              value={rank}
              onChange={(e) => setRank(Number(e.target.value) || 32)}
            />
          </div>
          <div>
            <label htmlFor="t-res">Resolution</label>
            <select
              id="t-res"
              value={resolution}
              onChange={(e) => setResolution(Number(e.target.value))}
            >
              <option value={1024}>1024 — native (5.6 GiB)</option>
              <option value={768}>768 — faster (5.4 GiB)</option>
              <option value={512}>512 — quick test</option>
            </select>
          </div>
        </div>

        <div className="spacer" />
        <button
          className="btn"
          disabled={busy || !name.trim() || !ready.data?.ready}
          onClick={() => void start()}
        >
          {busy ? 'Queuing…' : 'Start training'}
        </button>
        {!ready.data?.ready && (
          <span className="muted" style={{ marginLeft: 12 }}>
            {ready.data?.why}
          </span>
        )}
      </div>

      <div className="card">
        <h2>Runs</h2>
        <p className="hint">
          Training shares the queue with generation — one GPU means a run and a sheet
          build cannot overlap. Deleting a run removes the record, never the adapter.
        </p>
        {runs.error && <div className="note err">{runs.error}</div>}
        {runs.data?.items.length === 0 && <div className="empty">No training runs yet.</div>}
        {runs.data && runs.data.items.length > 0 && (
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Status</th>
                <th>Progress</th>
                <th>Loss</th>
                <th>Images</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {runs.data.items.map((r) => {
                const cfgName = (r.config?.name as string) ?? r.id.slice(0, 8)
                const pct = r.steps_total
                  ? Math.round((100 * r.steps_done) / r.steps_total)
                  : 0
                return (
                  <tr key={r.id}>
                    <td>{cfgName}</td>
                    <td>
                      <span
                        className={`tag ${
                          r.status === 'done' ? 'ok' : r.status === 'failed' ? 'no' : 'neutral'
                        }`}
                      >
                        {r.status}
                      </span>
                      {r.error && <div className="why">{r.error.slice(0, 160)}</div>}
                    </td>
                    <td style={{ minWidth: 140 }}>
                      {r.steps_done}/{r.steps_total ?? '?'}
                      <div className="bar">
                        <i style={{ width: `${pct}%` }} />
                      </div>
                    </td>
                    <td>{r.loss != null ? r.loss.toFixed(4) : '—'}</td>
                    <td>{r.dataset_size ?? '—'}</td>
                    <td style={{ textAlign: 'right' }}>
                      {r.status !== 'running' && r.status !== 'queued' && (
                        <button
                          className="btn danger sm"
                          onClick={() => void api.deleteRun(r.id).then(runs.reload)}
                        >
                          Forget
                        </button>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>
    </>
  )
}
