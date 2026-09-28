import { useEffect, useRef, useState } from 'react'
import { audioApi } from '../api'
import type { AudioGenerateOutcome, AudioProposal, AudioRow, AudioStyle, SfxCue } from '../api'
import { useAsync } from '../hooks'

type Kind = 'music' | 'ambience' | 'sfx'

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
  const [context, setContext] = useState('')
  const [proposing, setProposing] = useState(false)
  const [proposal, setProposal] = useState<AudioProposal | null>(null)
  const [proposeError, setProposeError] = useState<string | null>(null)

  // A style change resets its slots: another entry's mood value is not valid
  // here, and the server would silently swap it for the default. A proposal
  // sets style and slots together, so it marks its slots to survive this.
  const keepSlots = useRef(false)
  useEffect(() => {
    if (keepSlots.current) {
      keepSlots.current = false
      return
    }
    setSlots({})
  }, [kind, entry?.value])

  async function propose() {
    setProposing(true)
    setProposeError(null)
    try {
      const p = await audioApi.propose(context.trim(), kind)
      keepSlots.current = p.style !== entry?.value
      setStyle(p.style)
      setSlots(p.slots)
      setProposal(p)
    } catch (e) {
      setProposeError(e instanceof Error ? e.message : String(e))
    } finally {
      setProposing(false)
    }
  }

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
        kind: kind as 'music' | 'ambience',
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

        {kind !== 'sfx' && (<>
        <div className="row tight" style={{ alignItems: 'flex-end', marginBottom: 10 }}>
          <div style={{ flex: '3 1 300px' }}>
            <label htmlFor="a-context">Describe the map (optional)</label>
            <input
              id="a-context"
              value={context}
              placeholder="e.g. abandoned dwarven mine, danger"
              onChange={(e) => setContext(e.target.value)}
            />
          </div>
          <button
            className="btn ghost"
            disabled={proposing || !context.trim()}
            onClick={propose}
            title="The brain picks a style and fills its slots; nothing is generated"
          >
            {proposing ? 'Asking… (a cold LLM can take ~1 min)' : 'Propose style'}
          </button>
        </div>
        {proposeError && <div className="note err">{proposeError}</div>}
        {proposal && (
          <p className="hint" style={{ marginTop: 0 }}>
            <strong>{proposal.style}</strong> — {proposal.author}
            {proposal.adjusted.length > 0 && ` (corrected: ${proposal.adjusted.join('; ')})`}
          </p>
        )}
        </>)}

        <div className="row tight">
          <div style={{ flex: '1 1 150px' }}>
            <label htmlFor="a-kind">Kind</label>
            <select id="a-kind" value={kind} onChange={(e) => switchKind(e.target.value as Kind)}>
              <option value="ambience">ambience (Stable Audio Open)</option>
              <option value="music">music (ACE-Step)</option>
              <option value="sfx">sound effects (one-shot cues)</option>
            </select>
          </div>
          {kind !== 'sfx' && (<>
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
          </>)}
        </div>

        {kind === 'sfx' ? (
          <SfxPanel onBuilt={list.reload} />
        ) : (<>
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
        </>)}
      </div>

      <BatchPanel onProgress={list.reload} />

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

/**
 * Sound effects: one cue, or a pack of cues built in ONE model load.
 *
 * Addressed as `<cue>/<entity>`. The engine field is optional: left on
 * "auto", the server resolves it (request > world > cue default) and says
 * which level decided. Asking for an engine a cue has no recipe for is a 422
 * with the reason - retro arrives with ticket 17 - never a silent substitute.
 * Unlike music, a repeated cue/entity IS served from cache on purpose: the
 * name is the game's handle for that sound.
 */
function SfxPanel({ onBuilt }: { onBuilt: () => void }) {
  const cues = useAsync(() => audioApi.cues(), [])
  const [cue, setCue] = useState('hit')
  const [entity, setEntity] = useState('')
  const [engine, setEngine] = useState('')
  const [world, setWorld] = useState('')
  const [variants, setVariants] = useState(3)
  const [pack, setPack] = useState<Set<string>>(new Set())
  const [busy, setBusy] = useState(false)
  const [outcome, setOutcome] = useState<AudioGenerateOutcome | null>(null)
  const [packNote, setPackNote] = useState<string | null>(null)

  const current: SfxCue | undefined = cues.data?.find((c) => c.value === cue)

  async function run(isPack: boolean) {
    setBusy(true)
    setOutcome(null)
    setPackNote(null)
    const common = {
      world: world.trim() || undefined,
      variants,
    }
    try {
      const res = isPack
        ? await audioApi.sfxPack({
            ...common,
            items: [...pack].map((c) => ({
              cue: c,
              entity: entity.trim() || undefined,
              engine: engine || undefined,
            })),
          })
        : await audioApi.sfx({
            ...common,
            cue,
            entity: entity.trim() || undefined,
            engine: engine || undefined,
          })
      setOutcome(res)
      if (res.ok && isPack) {
        const items = (res.info as { items?: { name: string; served_from?: string; error?: string }[] })
          .items ?? []
        setPackNote(
          items
            .map((i) => `${i.name}: ${i.error ? `failed (${i.error})` : i.served_from}`)
            .join(' · '),
        )
      }
    } finally {
      setBusy(false)
      onBuilt()
    }
  }

  return (
    <>
      {cues.error && <div className="note err">{cues.error}</div>}
      <div className="row tight" style={{ marginTop: 10 }}>
        <div style={{ flex: '1 1 160px' }}>
          <label htmlFor="s-cue">Cue</label>
          <select id="s-cue" value={cue} onChange={(e) => setCue(e.target.value)}>
            {(cues.data ?? []).map((c) => (
              <option key={c.value} value={c.value}>
                {c.label}
              </option>
            ))}
          </select>
        </div>
        <div style={{ flex: '2 1 200px' }}>
          <label htmlFor="s-entity">Entity (optional)</label>
          <input
            id="s-entity"
            value={entity}
            placeholder={current ? `e.g. ${current.entity_default}` : ''}
            onChange={(e) => setEntity(e.target.value)}
          />
        </div>
        <div style={{ flex: '1 1 150px' }}>
          <label htmlFor="s-engine">Engine</label>
          <select id="s-engine" value={engine} onChange={(e) => setEngine(e.target.value)}>
            <option value="">auto (world, then cue default)</option>
            <option value="realistic">realistic</option>
            <option value="retro">retro (8-bit)</option>
          </select>
        </div>
        <div style={{ flex: '1 1 150px' }}>
          <label htmlFor="s-world">World (optional)</label>
          <input
            id="s-world"
            value={world}
            placeholder="reads its sfx_engine"
            onChange={(e) => setWorld(e.target.value)}
          />
        </div>
        <div style={{ flex: '0 1 90px' }}>
          <label htmlFor="s-var">Variants</label>
          <input
            id="s-var"
            type="number"
            min={1}
            max={5}
            value={variants}
            onChange={(e) => setVariants(Math.max(1, Math.min(5, Number(e.target.value) || 1)))}
          />
        </div>
      </div>
      {current && engine && !current.engines.includes(engine) && (
        <div className="note" style={{ marginTop: 8 }}>
          {cue} has no {engine} recipe yet ({current.engines.join(', ')} only) — the server
          will refuse it rather than substitute.
        </div>
      )}
      <div className="row" style={{ marginTop: 12, alignItems: 'center' }}>
        <button className="btn" disabled={busy || !cue} onClick={() => run(false)}>
          {busy ? 'Generating…' : `Generate ${variants} variant${variants > 1 ? 's' : ''}`}
        </button>
        {outcome && <OutcomeNote o={outcome} />}
      </div>

      <details style={{ marginTop: 14 }}>
        <summary className="muted">Pack: several cues in one model load</summary>
        <div className="row tight" style={{ marginTop: 8 }}>
          {(cues.data ?? []).map((c) => (
            <label key={c.value} style={{ flex: '0 1 auto', display: 'flex', gap: 6 }}>
              <input
                type="checkbox"
                checked={pack.has(c.value)}
                onChange={(e) => {
                  const next = new Set(pack)
                  if (e.target.checked) next.add(c.value)
                  else next.delete(c.value)
                  setPack(next)
                }}
              />
              {c.value}
            </label>
          ))}
        </div>
        <p className="hint">
          Uses the entity, engine, world and variants above for every cue. Measured: ~8 s
          per variant after a one-off ~4 s load, so 6 cues × 3 variants is about 2.5 min.
        </p>
        <button className="btn ghost" disabled={busy || pack.size === 0} onClick={() => run(true)}>
          Generate pack ({pack.size} cue{pack.size === 1 ? '' : 's'})
        </button>
        {packNote && <p className="hint">{packNote}</p>}
      </details>
    </>
  )
}

type BatchKind = 'music' | 'ambience' | 'sfx'
type BatchState = 'waiting' | 'running' | 'done' | 'cached' | 'failed' | 'stopped'
interface BatchRow {
  line: string
  state: BatchState
  note: string
}

/**
 * Many takes from one list - the batch counterpart of the form above.
 *
 * music/ambience: one line = `name` or `name, style`, sent ONE AT A TIME (the
 * GPU runs one job; parallel requests would only queue). A 503 `building` is
 * not a failure - the build continues server-side - so the row waits and asks
 * again, which is then a cache hit. `busy` waits out its Retry-After.
 * sfx: one line = `cue` or `cue/entity`, sent as `sfx-pack` calls of up to 40,
 * so the whole list shares one model load.
 */
function BatchPanel({ onProgress }: { onProgress: () => void }) {
  const [kind, setKind] = useState<BatchKind>('music')
  const [text, setText] = useState('')
  const [engine, setEngine] = useState('')
  const [variants, setVariants] = useState(3)
  const [rows, setRows] = useState<BatchRow[]>([])
  const [running, setRunning] = useState(false)
  const stop = useRef(false)

  const lines = text
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith('#'))

  function set(i: number, state: BatchState, note = '') {
    setRows((rs) => rs.map((r, j) => (j === i ? { ...r, state, note } : r)))
  }
  const sleep = (s: number) => new Promise((ok) => window.setTimeout(ok, s * 1000))

  async function runTracks(k: 'music' | 'ambience') {
    for (let i = 0; i < lines.length; i++) {
      if (stop.current) {
        setRows((rs) => rs.map((r, j) => (j >= i && r.state === 'waiting' ? { ...r, state: 'stopped' } : r)))
        return
      }
      const [name, style] = lines[i].split(',').map((s) => s.trim())
      set(i, 'running')
      for (let attempt = 0; attempt < 30; attempt++) {
        const res = await audioApi.generate({ kind: k, name, style: style || undefined })
        if (res.ok) {
          const cached = (res.info as { cached?: boolean }).cached
          set(i, cached ? 'cached' : 'done')
          break
        }
        if (res.reason === 'building' || res.reason === 'busy') {
          set(i, 'running', `${res.reason}; retrying in ${res.retry_after_s ?? 60}s`)
          await sleep(res.retry_after_s ?? 60)
          if (stop.current) break
          continue
        }
        set(i, 'failed', res.detail)
        break
      }
      onProgress()
    }
  }

  async function runCues() {
    const items = lines.map((l) => {
      const [cue, ...rest] = l.split('/')
      return { cue: cue.trim(), entity: rest.join('/').trim() || undefined,
               engine: engine || undefined }
    })
    for (let start = 0; start < items.length; start += 40) {
      if (stop.current) break
      const chunk = items.slice(start, start + 40)
      chunk.forEach((_, n) => set(start + n, 'running'))
      const res = await audioApi.sfxPack({ items: chunk, variants })
      if (!res.ok) {
        chunk.forEach((_, n) => set(start + n, 'failed', res.detail))
        continue
      }
      const out = (res.info as { items?: { error?: string; served_from?: string }[] }).items ?? []
      out.forEach((it, n) =>
        set(start + n, it.error ? 'failed' : it.served_from === 'cache' ? 'cached' : 'done', it.error ?? ''),
      )
      onProgress()
    }
  }

  async function run() {
    stop.current = false
    setRows(lines.map((line) => ({ line, state: 'waiting', note: '' })))
    setRunning(true)
    try {
      if (kind === 'sfx') await runCues()
      else await runTracks(kind)
    } finally {
      setRunning(false)
      onProgress()
    }
  }

  const done = rows.filter((r) => r.state === 'done' || r.state === 'cached').length
  const placeholder =
    kind === 'sfx'
      ? 'hit/slime\nhit/orc\nslash/an axe\npickup/a gold coin\ndeath/skeleton\nui_click'
      : kind === 'music'
        ? 'emerald-reach:explore:1, medieval_fantasy\nemerald-reach:combat:1, battle\nemerald-reach:village:1, village'
        : 'emerald-reach:forest, forest\nemerald-reach:cave, cave'

  return (
    <div className="card">
      <h2>Batch</h2>
      <p className="hint">
        One item per line (<code>#</code> starts a comment). Music and ambience:{' '}
        <code>name</code> or <code>name, style</code>, built one at a time. Sound
        effects: <code>cue</code> or <code>cue/entity</code>, built in packs sharing one
        model load. Names already built come back from cache instantly.
      </p>
      <div className="row tight">
        <div style={{ flex: '1 1 150px' }}>
          <label htmlFor="b-kind">Kind</label>
          <select id="b-kind" value={kind} disabled={running}
                  onChange={(e) => setKind(e.target.value as BatchKind)}>
            <option value="music">music</option>
            <option value="ambience">ambience</option>
            <option value="sfx">sound effects</option>
          </select>
        </div>
        {kind === 'sfx' && (
          <>
            <div style={{ flex: '1 1 150px' }}>
              <label htmlFor="b-engine">Engine</label>
              <select id="b-engine" value={engine} disabled={running}
                      onChange={(e) => setEngine(e.target.value)}>
                <option value="">auto</option>
                <option value="realistic">realistic</option>
                <option value="retro">retro</option>
              </select>
            </div>
            <div style={{ flex: '0 1 90px' }}>
              <label htmlFor="b-var">Variants</label>
              <input id="b-var" type="number" min={1} max={5} value={variants}
                     disabled={running}
                     onChange={(e) => setVariants(Math.max(1, Math.min(5, Number(e.target.value) || 1)))} />
            </div>
          </>
        )}
      </div>
      <textarea
        rows={6}
        style={{ width: '100%', marginTop: 8, fontFamily: 'monospace' }}
        value={text}
        disabled={running}
        placeholder={placeholder}
        onChange={(e) => setText(e.target.value)}
      />
      <div className="row" style={{ marginTop: 8, alignItems: 'center' }}>
        <button className="btn" disabled={running || lines.length === 0} onClick={() => void run()}>
          {running ? `Working… ${done}/${rows.length}` : `Run batch (${lines.length})`}
        </button>
        {running && (
          <button className="btn ghost" onClick={() => (stop.current = true)}>
            Stop after current
          </button>
        )}
      </div>
      {rows.length > 0 && (
        <table style={{ marginTop: 10 }}>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                <td><code>{r.line}</code></td>
                <td>
                  <span className={`tag ${r.state === 'failed' ? 'no'
                    : r.state === 'done' || r.state === 'cached' ? 'ok' : 'neutral'}`}>
                    {r.state}
                  </span>
                </td>
                <td className="muted">{r.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
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
      {r.kind === 'sfx' && r.variants && (
        <div className="row tight" style={{ marginTop: 6, alignItems: 'center' }}>
          <span className="muted">
            {r.engine} (from {r.engine_from ?? '?'})
          </span>
          {r.variants.map((v, i) => (
            <span key={v} style={{ display: 'inline-flex', gap: 4, alignItems: 'center' }}>
              <span className="muted">v{i + 1}</span>
              <audio controls preload="none" src={v} style={{ height: 32, width: 180 }} />
              <a className="btn ghost sm" href={v} download>
                OGG
              </a>
            </span>
          ))}
        </div>
      )}
      {r.url && r.kind !== 'sfx' && (
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
