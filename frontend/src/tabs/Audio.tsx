import { useEffect, useRef, useState } from 'react'
import { audioApi } from '../api'
import type { AudioGenerateOutcome, AudioRow, AudioStyle } from '../api'
import { useAsync } from '../hooks'

type Kind = 'music' | 'ambience'

/**
 * Generate, listen to, and audition music and ambience loops by hand.
 *
 * Drives the same `POST /api/audio` something2 calls, so what works here works
 * there. Two things shape the page:
 *
 * - The endpoint is CACHE-FIRST BY NAME. Submitting a name that already has a
 *   finished track returns that track, not a new one. The name field is
 *   pre-filled with a fresh one so a click means "make me a new take".
 * - A 503 with `reason: building` is not an error: the build carries on and
 *   the list below picks it up when it lands. Only `busy` and `gpu_faulted`
 *   are shown as problems, and neither is retried automatically.
 *
 * The seam is the property under test, so every finished row has a Loop
 * button (plays the loop region forever, sample-accurate via Web Audio) and a
 * Seam button (starts 3 s before the loop end, so the join is heard at once).
 */
export default function Audio() {
  const [kind, setKind] = useState<Kind>('ambience')
  const styles = useAsync(() => audioApi.styles(), [])
  const list = useAsync(() => audioApi.list({ limit: 50 }), [])

  const forKind = (styles.data ?? []).filter((s) => s.kind === kind)
  const [style, setStyle] = useState('')
  const entry: AudioStyle | undefined =
    forKind.find((s) => s.value === style) ?? forKind.find((s) => s.default) ?? forKind[0]

  const [slots, setSlots] = useState<Record<string, string | number>>({})
  const [name, setName] = useState(freshName('ambience'))
  const [seed, setSeed] = useState('')
  const [duration, setDuration] = useState('')
  const [prompt, setPrompt] = useState('')
  const [busy, setBusy] = useState(false)
  const [outcome, setOutcome] = useState<AudioGenerateOutcome | null>(null)

  // A style change resets its slots: another entry's mood value is not valid
  // here, and the server would silently swap it for the default.
  useEffect(() => setSlots({}), [kind, entry?.value])

  // While something is building, poll the list so it appears the moment it
  // finishes - the request that started it may already have answered 503.
  const building = (list.data?.items ?? []).some((r) => r.status === 'running')
  useEffect(() => {
    if (!building && !busy) return
    const t = window.setInterval(list.reload, 5000)
    return () => window.clearInterval(t)
  }, [building, busy, list.reload])

  function switchKind(k: Kind) {
    setKind(k)
    setStyle('')
    setName(freshName(k))
  }

  async function submit() {
    if (!entry) return
    setBusy(true)
    setOutcome(null)
    try {
      const res = await audioApi.generate({
        kind,
        name: name.trim(),
        style: entry.value,
        prompt: prompt.trim() || undefined,
        slots: Object.keys(slots).length ? slots : undefined,
        seed: seed ? Number(seed) : undefined,
        duration_s: duration ? Number(duration) : undefined,
      })
      setOutcome(res)
      // Next click should be a new take, not a cache read of this one.
      if (res.ok || res.reason === 'building') setName(freshName(kind))
    } catch (e) {
      setOutcome({
        ok: false,
        status: 0,
        reason: 'error',
        detail: e instanceof Error ? e.message : String(e),
        retry_after_s: null,
      })
    } finally {
      setBusy(false)
      list.reload()
    }
  }

  const bounds = kind === 'music' ? [60, 240, 120] : [20, 45, 30]

  return (
    <>
      <div className="card">
        <h2>Audio — music and ambience loops</h2>
        <p className="hint">
          Every take is a seamless loop: an OGG with loop points, and a WAV master beside
          it in <code>audio\&lt;kind&gt;\</code> on disk. House style is{' '}
          <strong>smooth and medieval</strong>. A name that already has a finished track
          is served from cache, so a fresh name is filled in for you.
        </p>

        {styles.error && <div className="note err">{styles.error}</div>}

        <div className="row tight">
          <div style={{ flex: '1 1 150px' }}>
            <label htmlFor="a-kind">Kind</label>
            <select id="a-kind" value={kind} onChange={(e) => switchKind(e.target.value as Kind)}>
              <option value="ambience">ambience (Stable Audio Open)</option>
              <option value="music">music (ACE-Step)</option>
            </select>
          </div>
          <div style={{ flex: '2 1 220px' }}>
            <label htmlFor="a-style">Style</label>
            <select
              id="a-style"
              value={entry?.value ?? ''}
              onChange={(e) => setStyle(e.target.value)}
            >
              {forKind.map((s) => (
                <option key={s.value} value={s.value}>
                  {s.label}
                </option>
              ))}
            </select>
          </div>
          <div style={{ flex: '2 1 200px' }}>
            <label htmlFor="a-name">Name (map)</label>
            <input id="a-name" value={name} onChange={(e) => setName(e.target.value)} />
          </div>
        </div>

        {kind === 'music' && (
          <p className="hint" style={{ marginTop: 10 }}>
            A 2-minute loop takes about a minute cold (measured 56 s): ACE-Step loads,
            generates, and the loop is cut on whole bars at the style&apos;s tempo.
          </p>
        )}

        {entry && (
          <div className="row tight" style={{ marginTop: 10 }}>
            {Object.entries(entry.slots).map(([slot, spec]) => (
              <div key={slot} style={{ flex: '1 1 160px' }}>
                <label htmlFor={`a-slot-${slot}`}>{slot.replace(/_/g, ' ')}</label>
                {spec.type === 'enum' ? (
                  <select
                    id={`a-slot-${slot}`}
                    value={String(slots[slot] ?? spec.default)}
                    onChange={(e) => setSlots({ ...slots, [slot]: e.target.value })}
                  >
                    {spec.values?.map((v) => (
                      <option key={v} value={v}>
                        {v}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    id={`a-slot-${slot}`}
                    type="number"
                    min={spec.min}
                    max={spec.max}
                    value={slots[slot] ?? spec.default}
                    onChange={(e) => setSlots({ ...slots, [slot]: Number(e.target.value) })}
                  />
                )}
              </div>
            ))}
            <div style={{ flex: '1 1 120px' }}>
              <label htmlFor="a-seed">Seed</label>
              <input
                id="a-seed"
                type="number"
                placeholder="random"
                value={seed}
                onChange={(e) => setSeed(e.target.value)}
              />
            </div>
            <div style={{ flex: '1 1 120px' }}>
              <label htmlFor="a-dur">Seconds</label>
              <input
                id="a-dur"
                type="number"
                min={bounds[0]}
                max={bounds[1]}
                placeholder={String(bounds[2])}
                value={duration}
                onChange={(e) => setDuration(e.target.value)}
              />
            </div>
          </div>
        )}

        <details style={{ marginTop: 10 }}>
          <summary className="muted">Prompt override (bypasses the style template)</summary>
          <textarea
            rows={3}
            style={{ width: '100%', marginTop: 6 }}
            value={prompt}
            placeholder="Leave empty to use the style. The style's negative list and metre are kept either way."
            onChange={(e) => setPrompt(e.target.value)}
          />
        </details>

        <div className="row" style={{ marginTop: 12, alignItems: 'center' }}>
          <button className="btn" disabled={busy || !entry || !name.trim()} onClick={submit}>
            {busy ? 'Generating… (up to 4 min)' : 'Generate'}
          </button>
          {outcome && <OutcomeNote o={outcome} />}
        </div>
      </div>

      <div className="card">
        <h2>Takes</h2>
        {list.error && <div className="note err">{list.error}</div>}
        {list.data && list.data.items.length === 0 && (
          <div className="empty">Nothing generated yet.</div>
        )}
        <div className="rows">
          {(list.data?.items ?? []).map((r) => (
            <TakeRow key={r.id} r={r} />
          ))}
        </div>
      </div>
    </>
  )
}

function OutcomeNote({ o }: { o: AudioGenerateOutcome }) {
  if (o.ok) return <span className="tag ok">done — it is at the top of the list</span>
  if (o.reason === 'building')
    return (
      <span className="muted">
        Still building (not cancelled). It will appear below when it lands.
      </span>
    )
  return (
    <div className="note err" style={{ flex: '1 1 100%' }}>
      {o.reason === 'busy' && 'The GPU worker is busy with another job. '}
      {o.reason === 'gpu_faulted' && 'The GPU breaker is open; nothing was queued. '}
      {o.detail}
      {o.retry_after_s != null && ` Try again in ~${o.retry_after_s}s.`}
    </div>
  )
}

function TakeRow({ r }: { r: AudioRow }) {
  const loop = useLoopPlayer(r)
  const wav = r.url?.replace(/\.ogg$/, '.wav')

  return (
    <div style={{ padding: '10px 0', borderBottom: '1px solid rgba(255,255,255,.06)' }}>
      <div>
        <strong>{r.name}</strong> <span className="tag neutral">{r.kind}</span>{' '}
        {r.style && <span className="tag neutral">{r.style}</span>}{' '}
        <span
          className={`tag ${r.status === 'done' ? 'ok' : r.status === 'failed' ? 'no' : 'neutral'}`}
        >
          {r.status}
        </span>{' '}
        <span className="muted">
          {r.duration_s != null && `${r.duration_s.toFixed(1)}s`}
          {r.seam_rms_jump_db != null && ` · seam ${r.seam_rms_jump_db.toFixed(2)} dB`}
          {r.seed != null && ` · seed ${r.seed}`}
          {r.created_at && ` · ${new Date(r.created_at).toLocaleString()}`}
        </span>
      </div>
      {r.error && <div className="note err">{r.error}</div>}
      {r.prompt && (
        <div className="muted" style={{ fontSize: 12, marginTop: 4 }}>
          {r.prompt}
        </div>
      )}
      {r.url && (
        <div className="row tight" style={{ marginTop: 6, alignItems: 'center' }}>
          <audio controls preload="none" src={r.url} style={{ height: 32 }} />
          <button className="btn ghost sm" onClick={() => loop.play(0)} disabled={!loop.ready}>
            {loop.playing === 'loop' ? 'Stop' : 'Loop'}
          </button>
          <button className="btn ghost sm" onClick={() => loop.play(3)} disabled={!loop.ready}>
            {loop.playing === 'seam' ? 'Stop' : 'Seam'}
          </button>
          <a className="btn ghost sm" href={r.url} download>
            OGG
          </a>
          {wav && (
            <a className="btn ghost sm" href={wav} download>
              WAV
            </a>
          )}
          {loop.error && <span className="muted">{loop.error}</span>}
        </div>
      )}
    </div>
  )
}

/**
 * Sample-accurate loop playback over the row's loop points.
 *
 * `<audio loop>` restarts at the file's end and may gap; this loops exactly
 * `loop_start..loop_end` as the game will. Loop points arrive as SAMPLE
 * offsets at the file's own rate - converted to seconds, because
 * decodeAudioData resamples to the context's rate and seconds survive that.
 * One source at a time per row; everything stops on unmount (tab change).
 */
function useLoopPlayer(r: AudioRow) {
  const ctx = useRef<AudioContext | null>(null)
  const src = useRef<AudioBufferSourceNode | null>(null)
  const buf = useRef<AudioBuffer | null>(null)
  const [playing, setPlaying] = useState<'loop' | 'seam' | null>(null)
  const [error, setError] = useState<string | null>(null)
  const ready = !!(r.url && r.sample_rate && r.loop_end != null)

  useEffect(
    () => () => {
      src.current?.stop()
      void ctx.current?.close()
    },
    [],
  )

  async function play(leadIn: number) {
    const mode = leadIn > 0 ? 'seam' : 'loop'
    if (src.current) {
      src.current.stop()
      src.current = null
      const was = playing
      setPlaying(null)
      if (was === mode) return
    }
    if (!ready || !r.url) return
    try {
      ctx.current ??= new AudioContext()
      if (!buf.current) {
        const bytes = await (await fetch(r.url)).arrayBuffer()
        buf.current = await ctx.current.decodeAudioData(bytes)
      }
      const sr = r.sample_rate as number
      const start = (r.loop_start ?? 0) / sr
      const end = Math.min((r.loop_end as number) / sr, buf.current.duration)
      const node = ctx.current.createBufferSource()
      node.buffer = buf.current
      node.loop = true
      node.loopStart = start
      node.loopEnd = end
      node.connect(ctx.current.destination)
      node.start(0, Math.max(start, end - leadIn))
      src.current = node
      setPlaying(mode)
      setError(null)
    } catch (e) {
      setError(`could not play: ${e instanceof Error ? e.message : String(e)}`)
    }
  }

  return { play, playing, ready, error }
}

function freshName(kind: Kind) {
  const d = new Date()
  const stamp = `${d.getMonth() + 1}${String(d.getDate()).padStart(2, '0')}-${String(
    d.getHours(),
  ).padStart(2, '0')}${String(d.getMinutes()).padStart(2, '0')}${String(d.getSeconds()).padStart(2, '0')}`
  return `test-${kind}-${stamp}`
}
