import { useState } from 'react'
import { api, type ReferenceKind } from '../api'
import { useAsync, usePoll } from '../hooks'

/**
 * Train from a reference tab, and show what the run is doing.
 *
 * On each tab rather than only on the Training tab, because "I just uploaded
 * twenty tiles" and "train on tiles" are the same thought, and making the user
 * carry it to another screen is where it gets dropped.
 *
 * The counts are the honest ones: an incremental run trains on the references
 * this adapter has NOT already seen, so that is the number shown next to the
 * button, not the tab total.
 */
export default function TrainPanel({ kind }: { kind: ReferenceKind }) {
  // One adapter per kind: a shared trigger cannot mean "character" and
  // "ground" and "world map" at once.
  const SUGGESTED: Record<ReferenceKind, string> = {
    tile: 'something2-terrain',
    map: 'something2-maps',
    core: 'something2',
    sprite: 'something2',
  }
  const suggested = SUGGESTED[kind]
  const [name, setName] = useState(suggested)
  const [fullRetrain, setFullRetrain] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [started, setStarted] = useState<string | null>(null)

  const ready = useAsync(() => api.trainingReadiness([kind]), [kind])
  const runs = useAsync(() => api.training(), [])

  const active = runs.data?.items.find(
    (r) => r.status === 'running' || r.status === 'queued',
  )
  // Only poll while something is actually happening. A 5s poll on an idle tab
  // is pure noise on a machine whose GPU is the scarce resource.
  usePoll(() => {
    runs.reload()
    ready.reload()
  }, 5000, !!active)

  const total = ready.data?.per_kind?.[kind] ?? 0
  const fresh = ready.data?.new_per_kind?.[kind] ?? 0
  const willUse = fullRetrain ? total : fresh
  const floor = fullRetrain
    ? (ready.data?.min_images ?? 8)
    : (ready.data?.min_new_images ?? 4)

  // Nothing trained yet means there is no adapter to continue, so the first run
  // is a full one whatever the checkbox says. Saying so beats a surprise.
  const firstRun = fresh === total

  async function start() {
    setBusy(true)
    setError(null)
    setStarted(null)
    try {
      const res = await api.startTraining({
        name: name.trim(),
        kinds: [kind],
        full_retrain: fullRetrain,
      })
      setStarted(
        `${res.mode === 'incremental' ? 'Continuing' : 'Training'} ` +
          `"${name.trim()}" on ${res.dataset_size} ${kind} reference(s)` +
          `${res.resuming ? ', resuming from the existing adapter' : ''}. ` +
          `Prompt with ${res.trigger} when it finishes.`,
      )
      runs.reload()
      ready.reload()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const pct =
    active?.steps_total && active.steps_total > 0
      ? Math.round((100 * active.steps_done) / active.steps_total)
      : 0

  return (
    <div className="card">
      <h2>Train on {kind} references</h2>
      <p className="hint">
        By default this continues the existing adapter on the {kind} references it
        has not seen yet — so adding twelve images costs twelve images of training,
        not two hundred. Tick “retrain on all” to rebuild it from scratch.
      </p>

      {error && <div className="note err">{error}</div>}
      {started && <div className="note ok">{started}</div>}

      {active && (
        <div className="note info">
          <div>
            <strong>{(active.config?.name as string) ?? 'run'}</strong> ·{' '}
            {active.status}
            {active.steps_total
              ? ` · step ${active.steps_done}/${active.steps_total}`
              : ''}
            {active.loss != null ? ` · loss ${active.loss.toFixed(4)}` : ''}
          </div>
          <div className="bar">
            <i style={{ width: `${pct}%` }} />
          </div>
          <div className="muted" style={{ marginTop: 6 }}>
            {active.steps_done === 0
              ? 'Encoding images — training steps start after this.'
              : 'One GPU, so generation waits until this finishes.'}
          </div>
        </div>
      )}

      <div className="row tight">
        <div style={{ flex: '2 1 220px' }}>
          <label htmlFor={`tr-name-${kind}`}>Adapter name</label>
          <input
            id={`tr-name-${kind}`}
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </div>
        <div style={{ flex: '1 1 200px' }}>
          <label>Dataset</label>
          <label className="check">
            <input
              type="checkbox"
              checked={fullRetrain}
              onChange={(e) => setFullRetrain(e.target.checked)}
            />
            Retrain on all {total}
          </label>
        </div>
        <div style={{ flex: '0 0 auto' }}>
          <button
            className="btn"
            disabled={busy || !!active || !name.trim() || willUse < floor}
            onClick={() => void start()}
          >
            {busy ? 'Queuing…' : active ? 'Training…' : `Train on ${willUse}`}
          </button>
        </div>
      </div>

      <div className="muted" style={{ marginTop: 8 }}>
        {total} trainable {kind} reference(s) ·{' '}
        {firstRun
          ? 'none trained yet, so the first run uses all of them'
          : `${fresh} not yet trained on`}
        {willUse < floor && !active
          ? ` · need at least ${floor} to start`
          : ''}
      </div>
    </div>
  )
}
