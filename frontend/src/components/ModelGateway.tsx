import { useEffect, useRef, useState } from 'react'
import {
  api,
  ApiError,
  type GatewayChoice,
  type GatewayPending,
  type GatewayPhase,
  type GatewayStatus,
  type GatewaySwitching,
} from '../api'
import { useModelGateway } from '../hooks'

/**
 * The active image model: a nav pill, a switcher modal, and the blocking
 * "switching" popup.
 *
 * One image model holds the 12 GB card at a time. Jobs for any other model are
 * deferred rather than forcing a reload mid-queue. Spec:
 * .ai/specs/model-gateway/plan.md.
 *
 * Mounted once in the nav so it is live on every tab.
 */

const OVERLAY: React.CSSProperties = {
  position: 'fixed',
  inset: 0,
  background: 'rgba(0,0,0,.72)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  zIndex: 50,
  padding: 20,
}

/** 252 -> "4:12". */
function mmss(s: number): string {
  const t = Math.max(0, Math.round(s))
  return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, '0')}`
}

/** 300 -> "5 min", 90 -> "1.5 min", 45 -> "45 s". */
function minutes(s: number): string {
  if (s < 60) return `${Math.round(s)} s`
  const m = s / 60
  return `${Number.isInteger(m) ? m : m.toFixed(1)} min`
}

function labelOf(choices: GatewayChoice[] | undefined, model: string | null | undefined): string {
  if (!model) return 'no model yet'
  return choices?.find((c) => c.value === model)?.label ?? model
}

/** `tasks.generate_core_task` -> `generate_core`. */
function taskName(task: string): string {
  return (task || '?').replace(/^.*\./, '').replace(/_task$/, '')
}

const PHASE_TEXT: Record<GatewayPhase, string> = {
  queued: 'Queued - waiting for the worker to pick up the switch.',
  waiting: 'Waiting for the running job to finish. A running job is never interrupted.',
  loading: 'Loading the new model onto the GPU.',
}

export default function ModelGateway() {
  const gw = useModelGateway()
  const [open, setOpen] = useState(false)
  // Hiding the popup applies to ONE switch only, keyed by its start time; the
  // next switch shows it again.
  const [hiddenFor, setHiddenFor] = useState<number | null>(null)

  const d = gw.data
  const sw = d?.switching ?? null
  const active = d?.active ?? null

  const deferred = Object.values(d?.by_model ?? {}).reduce((n, m) => n + m.deferred, 0)
  const queued = Object.values(d?.by_model ?? {}).reduce((n, m) => n + m.queued + m.running, 0)

  const badges: string[] = []
  if (active?.pinned) {
    badges.push(
      active.pin_release_in_s != null
        ? `pinned · releases in ${mmss(active.pin_release_in_s)}`
        : 'pinned',
    )
  }
  if (queued) badges.push(`${queued} queued`)
  if (deferred) badges.push(`${deferred} deferred`)

  return (
    <>
      <button
        type="button"
        className={`mode model ${sw ? 'busy' : ''}`}
        title={gw.error ? `Model gateway: ${gw.error}` : 'Active image model - click to switch'}
        onClick={() => setOpen(true)}
      >
        {!d
          ? gw.error
            ? '⚠ model: unknown'
            : 'model…'
          : sw
            ? `⏳ switching → ${labelOf(d.choices, sw.to)}`
            : `🧠 ${labelOf(d.choices, active?.model)}`}
        {!sw && badges.length > 0 && <span className="badge">{badges.join(' · ')}</span>}
      </button>

      {open && d && (
        <SwitchModal status={d} onClose={() => setOpen(false)} onChanged={gw.reload} />
      )}
      {open && !d && (
        <div style={OVERLAY} onClick={(e) => e.target === e.currentTarget && setOpen(false)}>
          <div className="card" style={{ maxWidth: 480, width: '100%', margin: 0 }}>
            <h2>Active model</h2>
            <div className="note err">{gw.error ?? 'Loading…'}</div>
            <button className="btn ghost" onClick={() => setOpen(false)}>
              Close
            </button>
          </div>
        </div>
      )}

      {sw && hiddenFor !== sw.started && d && (
        <SwitchingPopup
          switching={sw}
          choices={d.choices}
          onHide={() => setHiddenFor(sw.started)}
        />
      )}
    </>
  )
}

function SwitchModal({
  status,
  onClose,
  onChanged,
}: {
  status: GatewayStatus
  onClose: () => void
  onChanged: () => void
}) {
  const { choices, active, pending, idle_s } = status
  const firstAvailable = choices.find((c) => c.available && !c.fixed)?.value ?? ''
  const [model, setModel] = useState<string>(active?.model ?? firstAvailable)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [conflict, setConflict] = useState<string | null>(null)
  const [done, setDone] = useState<string | null>(null)

  const models = choices.filter((c) => !c.fixed)
  const fixed = choices.filter((c) => c.fixed)
  const label = labelOf(choices, model)

  async function doSwitch(force: boolean) {
    if (!model) return
    if (force) {
      const ok = window.confirm(
        `Force switch to ${label}?\n\n` +
          'Queued jobs for other models are not cancelled - they are deferred, and ' +
          `run only after ${label} has been idle for ${minutes(idle_s)} ` +
          '(no queued or running work). A job already running finishes first.',
      )
      if (!ok) return
    }
    setBusy(true)
    setError(null)
    setConflict(null)
    setDone(null)
    try {
      const r = await api.gatewaySwitch(model, force)
      onChanged()
      if (r.status === 'already_active') {
        setDone(`${label} is already active - pinned.`)
      } else {
        // The blocking popup takes over from here.
        onClose()
      }
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) setConflict(e.message)
      else setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div style={OVERLAY} onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div
        className="card"
        style={{ maxWidth: 620, width: '100%', margin: 0, maxHeight: '90vh', overflow: 'auto' }}
      >
        <h2>Active model</h2>
        <p className="hint">
          One image model holds the GPU at a time. Currently{' '}
          <strong>{labelOf(choices, active?.model)}</strong>
          {active?.pinned && active.pin_release_in_s != null
            ? ` - pinned, releases in ${mmss(active.pin_release_in_s)} if idle.`
            : '.'}{' '}
          Jobs for other models are deferred until this one has been idle for{' '}
          {minutes(idle_s)}.
        </p>

        {error && <div className="note err">{error}</div>}
        {conflict && (
          <div className="note warn">
            {conflict}
            <div style={{ marginTop: 8 }}>
              <button className="btn sm" disabled={busy} onClick={() => void doSwitch(true)}>
                Force switch
              </button>
            </div>
          </div>
        )}
        {done && <div className="note ok">{done}</div>}

        <label htmlFor="gw-model">Switch to</label>
        <select id="gw-model" value={model} onChange={(e) => setModel(e.target.value)}>
          {!model && <option value="">- choose -</option>}
          {models.map((c) => (
            <option key={c.value} value={c.value} disabled={!c.available}>
              {c.label}
              {c.available ? '' : ' (not available)'}
              {c.value === active?.model ? ' - active' : ''}
            </option>
          ))}
          {fixed.length > 0 && (
            <optgroup label="Fixed-model jobs">
              {fixed.map((c) => (
                <option key={c.value} value={c.value} disabled={!c.available}>
                  {c.label}
                  {c.value === active?.model ? ' - active' : ''}
                </option>
              ))}
            </optgroup>
          )}
        </select>

        <div className="row tight" style={{ marginTop: 12 }}>
          <button className="btn" disabled={busy || !model} onClick={() => void doSwitch(false)}>
            {busy ? 'Switching…' : 'Switch'}
          </button>
          <button
            className="btn danger"
            disabled={busy || !model}
            onClick={() => void doSwitch(true)}
            title="Switch now; queued jobs for other models are deferred"
          >
            Force switch
          </button>
          <button className="btn ghost" onClick={onClose}>
            Close
          </button>
        </div>
        <p className="muted" style={{ marginTop: 8 }}>
          Switch waits for an empty queue and is refused otherwise. Force switch takes the
          card now; nothing is cancelled.
        </p>

        <h2 style={{ marginTop: 18 }}>Pending jobs</h2>
        {pending.length === 0 ? (
          <div className="muted">Nothing queued.</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Task</th>
                <th>Model</th>
                <th>State</th>
              </tr>
            </thead>
            <tbody>
              {pending.map((p) => (
                <PendingRow key={p.task_id} p={p} choices={choices} />
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

function PendingRow({ p, choices }: { p: GatewayPending; choices: GatewayChoice[] }) {
  return (
    <tr>
      <td>
        <code title={p.task_id}>{taskName(p.task)}</code>
      </td>
      <td>{labelOf(choices, p.model)}</td>
      <td>
        {p.running ? (
          <span className="tag ok">running</span>
        ) : p.deferred ? (
          <>
            <span className="tag wait">deferred</span>
            <div className="muted">{p.deferred}</div>
          </>
        ) : (
          <span className="tag neutral">queued</span>
        )}
      </td>
    </tr>
  )
}

function SwitchingPopup({
  switching,
  choices,
  onHide,
}: {
  switching: GatewaySwitching
  choices: GatewayChoice[]
  onHide: () => void
}) {
  // The server's elapsed_s is only as fresh as the last 3 s poll. Count on
  // locally from when that response arrived, so the number moves every second
  // without trusting the browser clock against the server's.
  const received = useRef({ at: Date.now(), elapsed: switching.elapsed_s })
  useEffect(() => {
    received.current = { at: Date.now(), elapsed: switching.elapsed_s }
  }, [switching.elapsed_s])
  const [, tick] = useState(0)
  useEffect(() => {
    const id = setInterval(() => tick((t) => t + 1), 1000)
    return () => clearInterval(id)
  }, [])

  const elapsed = received.current.elapsed + (Date.now() - received.current.at) / 1000
  const expected = switching.expected_s || 10
  // expected_s is a LOAD time, so the bar measures only the loading phase; the
  // queued and waiting phases have no honest estimate and stay indeterminate.
  const loadElapsed =
    switching.phase === 'loading' && switching.loading_started
      ? elapsed - (switching.loading_started - switching.started)
      : null
  const determinate = loadElapsed != null && loadElapsed <= expected
  const over = elapsed > expected

  return (
    // No backdrop close on purpose: the user should see the switch finish. The
    // Hide button is the deliberate way out.
    <div style={{ ...OVERLAY, zIndex: 60 }}>
      <div className="card" style={{ maxWidth: 460, width: '100%', margin: 0 }} role="alertdialog" aria-live="polite">
        <h2>Switching model to {labelOf(choices, switching.to)}, please wait</h2>
        <p className="hint">
          {switching.from ? `From ${labelOf(choices, switching.from)}. ` : ''}
          {PHASE_TEXT[switching.phase] ?? switching.phase}
        </p>
        <div className="muted">
          elapsed {Math.floor(elapsed)}s / expected ~{Math.round(expected)}s
          {loadElapsed != null ? ` (loading for ${Math.max(0, Math.floor(loadElapsed))}s)` : ''}
        </div>
        <div className={`bar ${determinate ? '' : 'indeterminate'}`}>
          <i style={determinate ? { width: `${(100 * (loadElapsed ?? 0)) / expected}%` } : undefined} />
        </div>
        {over && (
          <p className="muted" style={{ marginTop: 8 }}>
            Taking longer than the last measured load - a cold load can take minutes. Still
            working; this closes itself when the switch is done.
          </p>
        )}
        <div style={{ marginTop: 14, textAlign: 'right' }}>
          <button className="btn ghost sm" onClick={onHide}>
            Hide
          </button>
        </div>
      </div>
    </div>
  )
}
