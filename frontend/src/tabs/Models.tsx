import { useMemo, useState } from 'react'
import { api } from '../api'
import type { ModelEntry, ModelsStatus, ModelDownload } from '../api'
import { useAsync, usePoll } from '../hooks'

/**
 * Every model this machine should have, which it actually has, which holds the
 * card, and how each one performs.
 *
 * THREE FAMILIES, NEVER ONE TABLE. `.ai/domain.md`: `llm_name` is the IMAGE
 * model; a *brain* is the text/vision model that plans and never draws. Tokens
 * and tok/s exist only for brains, so brains and image models get separate
 * tables rather than one with half the columns blank - which is precisely the
 * confusion domain.md records costing real time. Audio is a third case and
 * gets neither: its ledger rows carry no model at all, so it is reported as
 * per-kind totals and labelled as such instead of as an empty breakdown.
 *
 * NOTHING HERE IS NEW MEASUREMENT. Every brain call already wrote its `usage`
 * and `timings` into the generations ledger (tasks._text_ledger_params); this
 * reads 2,000+ existing rows. The sizes come from models.toml, where they are
 * the Hub's own sums over the files each entry's globs select.
 *
 * DOWNLOADS ARE QUEUED, NOT RUN. The button posts a row; the downloader
 * container is the single consumer. Two writers on one /models cache deadlock
 * on HuggingFace's file locks - see .ai/decisions/0014.
 */

// --- chart palette ------------------------------------------------------
//
// VALIDATED, not chosen by eye: `validate_palette.js --mode dark --surface
// #212138` passes all five checks on these three in this order - lightness
// band, chroma floor, CVD separation (worst adjacent ΔE 9.3 deutan),
// normal-vision floor (16.5), contrast vs surface.
//
// Fixed order, never cycled. The violet is brand-adjacent to --accent on
// purpose; teal and pink sit either side of it so the weakest CVD pair is
// never adjacent. --success/--warn/--danger are NOT used here: they are
// reserved for status and reusing them as a series is how a chart starts
// implying "audio is bad".
const SERIES = [
  { key: 'image', label: 'Image', color: '#1fa5c4' },
  { key: 'text', label: 'Text (brains)', color: '#8577e8' },
  { key: 'audio', label: 'Audio', color: '#d96a93' },
] as const

// Ledger `kind` -> family. Mirrors models_status.py; a kind missing from both
// shows up as uncounted rather than silently joining another family.
const FAMILY: Record<string, string> = {
  entity: 'image', raw: 'image', core: 'image', spritesheet: 'image',
  tile: 'image', map: 'image',
  text: 'text',
  music: 'audio', ambience: 'audio', sfx: 'audio',
}

/** A size a human reads. Scales the unit: "0.00 GB" looks like a bug. */
function gib(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return '—'
  if (bytes === 0) return '0'
  if (bytes < 1024 ** 2) return `${Math.max(1, Math.round(bytes / 1024))} KB`
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(0)} MB`
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`
}

/**
 * The first SENTENCE of a manifest `why`, not its first line.
 *
 * These are multi-line TOML strings wrapped at ~78 columns, so taking line one
 * cut them mid-clause - the table showed "Hybrid: attention and KV on the
 * card," with a trailing comma, which reads as truncation damage rather than a
 * summary. The full text is in models.toml; this is a label.
 */
function summary(why: string | null): string | null {
  if (!why) return null
  const flat = why.replace(/\s+/g, ' ').trim()
  // Any first sentence, however short. A `> 20` guard here fell through for
  // "Music generation." and spilled acestep-main's whole size breakdown into
  // the cell, truncated mid-word at the clamp.
  const stop = flat.indexOf('. ')
  const first = stop > 0 ? flat.slice(0, stop + 1) : flat
  return first.length > 120 ? `${first.slice(0, 119).trimEnd()}…` : first
}

function num(n: number | null | undefined, digits = 1): string {
  return n === null || n === undefined ? '—' : n.toFixed(digits)
}

function int(n: number | null | undefined): string {
  return n === null || n === undefined ? '—' : Math.round(n).toLocaleString()
}

function ago(iso: string | null): string {
  if (!iso) return 'never'
  const s = (Date.now() - new Date(iso).getTime()) / 1000
  if (s < 90) return `${Math.round(s)}s ago`
  if (s < 5400) return `${Math.round(s / 60)}m ago`
  if (s < 172800) return `${Math.round(s / 3600)}h ago`
  return `${Math.round(s / 86400)}d ago`
}

/** A status pill. Icon + word, never colour alone. */
function Pill({ state, children }: { state: 'ok' | 'warn' | 'bad' | 'idle'; children: React.ReactNode }) {
  const icon = state === 'ok' ? '●' : state === 'warn' ? '◐' : state === 'bad' ? '▲' : '○'
  const color = state === 'ok' ? 'var(--success)'
    : state === 'warn' ? 'var(--warn)'
    : state === 'bad' ? 'var(--danger)' : 'var(--muted)'
  return (
    <span style={{ color, whiteSpace: 'nowrap' }}>
      <span aria-hidden="true">{icon}</span> {children}
    </span>
  )
}

// --- the time chart -----------------------------------------------------
//
// ONE MEASURE PER CHART. Calls and tokens are different scales, so they are
// two charts rather than one with two y-axes - a dual axis is the single most
// common charting mistake and makes any crossing point meaningless.

type Point = { day: string; values: Record<string, number> }

function LineChart({
  points, series, title, format, oneSeries,
}: {
  points: Point[]
  series: readonly { key: string; label: string; color: string }[]
  title: string
  format: (n: number) => string
  oneSeries?: boolean
}) {
  const [hover, setHover] = useState<number | null>(null)
  const W = 720, H = 190, L = 52, R = 14, T = 12, B = 26

  const max = Math.max(1, ...points.flatMap((p) => series.map((s) => p.values[s.key] ?? 0)))
  const x = (i: number) => L + (points.length < 2 ? 0 : (i * (W - L - R)) / (points.length - 1))
  const y = (v: number) => T + (H - T - B) * (1 - v / max)

  if (!points.length) return <p className="muted">No activity in this window.</p>

  // Four gridlines, labelled. Recessive: the data is the thing to read.
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(max * f))
  const uniqueTicks = [...new Set(ticks)]

  return (
    <figure style={{ margin: '0 0 1.25rem' }}>
      <figcaption style={{ fontWeight: 600, marginBottom: '.4rem' }}>{title}</figcaption>
      <svg
        viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
        aria-label={title}
        onMouseLeave={() => setHover(null)}
        style={{ display: 'block', maxWidth: '100%' }}
      >
        {uniqueTicks.map((t) => (
          <g key={t}>
            <line x1={L} x2={W - R} y1={y(t)} y2={y(t)}
                  stroke="var(--border)" strokeWidth="1" />
            <text x={L - 8} y={y(t) + 4} textAnchor="end"
                  fill="var(--muted)" fontSize="10">{format(t)}</text>
          </g>
        ))}

        {series.map((s) => {
          const d = points
            .map((p, i) => `${i ? 'L' : 'M'}${x(i)},${y(p.values[s.key] ?? 0)}`)
            .join(' ')
          return (
            <path key={s.key} d={d} fill="none" stroke={s.color}
                  strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
          )
        })}

        {/* Crosshair + per-day readout. An SVG chart on a page is interactive;
            a static one makes the reader guess at every value between labels. */}
        {hover !== null && (
          <line x1={x(hover)} x2={x(hover)} y1={T} y2={H - B}
                stroke="var(--muted)" strokeWidth="1" strokeDasharray="3 3" />
        )}
        {hover !== null && series.map((s) => (
          <circle key={s.key} cx={x(hover)} cy={y(points[hover].values[s.key] ?? 0)}
                  r="4" fill={s.color} stroke="var(--card)" strokeWidth="2" />
        ))}

        {/* Hit targets wider than the marks, so hovering is not a precision
            exercise on a 2px line. */}
        {points.map((_, i) => (
          <rect key={i} x={x(i) - (W - L - R) / Math.max(1, points.length) / 2}
                y={T} width={(W - L - R) / Math.max(1, points.length)} height={H - T - B}
                fill="transparent" onMouseEnter={() => setHover(i)} />
        ))}

        <text x={L} y={H - 8} fill="var(--muted)" fontSize="10">
          {points[0]?.day.slice(0, 10)}
        </text>
        <text x={W - R} y={H - 8} textAnchor="end" fill="var(--muted)" fontSize="10">
          {points[points.length - 1]?.day.slice(0, 10)}
        </text>
      </svg>

      <div className="row" style={{ gap: '1rem', flexWrap: 'wrap', marginTop: '.3rem' }}>
        {/* A legend is always present for >=2 series, so identity never rests
            on colour alone. A single series needs none - the title names it. */}
        {!oneSeries && series.map((s) => (
          <span key={s.key} className="muted" style={{ fontSize: '.85rem' }}>
            <span aria-hidden="true" style={{
              display: 'inline-block', width: 10, height: 10, borderRadius: 2,
              background: s.color, marginRight: 6,
            }} />
            {s.label}
            {hover !== null && `: ${format(points[hover].values[s.key] ?? 0)}`}
          </span>
        ))}
        {oneSeries && hover !== null && (
          <span className="muted" style={{ fontSize: '.85rem' }}>
            {points[hover].day.slice(0, 10)}: {format(points[hover].values[series[0].key] ?? 0)}
          </span>
        )}
      </div>
    </figure>
  )
}

export default function Models() {
  const [days, setDays] = useState(30)
  const status = useAsync(() => api.modelsStatus(days), [days])
  const downloads = useAsync(() => api.modelDownloads(40), [])
  const [busy, setBusy] = useState<string | null>(null)
  const [note, setNote] = useState<string | null>(null)

  const active = (downloads.data?.downloads ?? []).some(
    (d) => d.status === 'queued' || d.status === 'running')
  // Poll only while something is moving, as Activity does: a machine at rest
  // does not need a request every three seconds.
  usePoll(() => { downloads.reload(); status.reload() }, 3000, active)

  const d: ModelsStatus | null = status.data

  const points = useMemo<Point[]>(() => {
    if (!d) return []
    const byDay = new Map<string, Record<string, number>>()
    for (const r of d.timeseries) {
      const fam = FAMILY[r.kind]
      if (!fam) continue
      const slot = byDay.get(r.day) ?? {}
      slot[fam] = (slot[fam] ?? 0) + r.calls
      slot[`${fam}_tokens`] = (slot[`${fam}_tokens`] ?? 0) + Number(r.tokens ?? 0)
      byDay.set(r.day, slot)
    }
    return [...byDay.entries()]
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([day, values]) => ({ day, values }))
  }, [d])

  const tokenPoints = useMemo<Point[]>(
    () => points.map((p) => ({ day: p.day, values: { text: p.values.text_tokens ?? 0 } })),
    [points])

  async function queue(entry: string) {
    setBusy(entry); setNote(null)
    try {
      await api.queueModelDownload(entry)
      setNote(`Queued ${entry}. The downloader fetches one model at a time.`)
      downloads.reload()
    } catch (e) {
      setNote(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  async function cancel(id: string) {
    setBusy(id); setNote(null)
    try {
      await api.cancelModelDownload(id)
      downloads.reload()
    } catch (e) {
      setNote(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  if (status.loading && !d) return <p className="muted">Loading models…</p>
  if (status.error) return <p style={{ color: 'var(--danger)' }}>{status.error}</p>
  if (!d) return null

  const required = d.declared.filter((m) => m.tier === 'required')
  const optional = d.declared.filter((m) => m.tier === 'optional')

  return (
    <div className="wrap">
      {!d.manifest_loaded && (
        <div className="card" style={{ borderColor: 'var(--warn)' }}>
          <strong>The manifest is not readable at {d.manifest_path}.</strong>
          <p className="muted">
            Statistics still work, but declared-vs-present cannot be shown. The
            API container needs <code>models.toml</code> mounted — see
            <code> compose/develop/docker-compose.yml</code>.
          </p>
        </div>
      )}

      {/* Headline numbers are tiles, not a chart: four values with no shape
          worth plotting. */}
      <div className="grid" style={{ gap: '.75rem', marginBottom: '1rem' }}>
        <Tile label="Declared" value={`${d.totals.present} / ${d.totals.declared}`}
              sub="on disk / in models.toml" />
        <Tile label="Required tier" value={gib(d.totals.required_gb * 1024 ** 3)}
              sub={`${gib(d.totals.all_gb * 1024 ** 3)} with every option`} />
        <Tile label="Missing" value={d.totals.missing_gb ? gib(d.totals.missing_gb * 1024 ** 3) : 'nothing'}
              sub={d.missing_required.length
                ? `${d.missing_required.length} required`
                : 'required tier complete'}
              state={d.missing_required.length ? 'bad' : 'ok'} />
        <Tile label="Holding the card" value={d.active ?? 'idle'}
              sub="model gateway" />
      </div>

      {d.missing_required.length > 0 && (
        <div className="card" style={{ borderColor: 'var(--danger)' }}>
          <Pill state="bad">
            {d.missing_required.length} required model(s) missing
          </Pill>
          <p className="muted">
            {d.missing_required.join(', ')} — generation will fail on the paths
            that need them. Queue them below, or run <code>make provision</code>.
          </p>
        </div>
      )}

      {note && <div className="note">{note}</div>}

      <h3>Downloads</h3>
      <p className="muted">
        Queued here and fetched by the downloader container, one at a time.
        Progress is measured from disk, so it is exact for single-file models
        and an estimate for repos whose blobs are shared.
      </p>
      <DownloadList rows={downloads.data?.downloads ?? []} busy={busy} onCancel={cancel} />

      <h3>Required</h3>
      <ModelTable rows={required} busy={busy} onQueue={queue} active={d.active} />

      <h3>Optional</h3>
      <ModelTable rows={optional} busy={busy} onQueue={queue} active={d.active} />

      <h3>Brains — text generation</h3>
      <p className="muted">
        Tokens and speed exist only here. An image model draws and never
        tokenises; see the table below it.
      </p>
      <div style={{ overflowX: 'auto' }}>
        <table>
          <thead>
            <tr>
              <th>Brain</th><th>Calls</th><th>Failed</th><th>Output tokens</th>
              <th>Avg tok/s</th><th>p50 gen</th><th>p95 gen</th>
              <th>Avg cold load</th><th>Last used</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(d.brains).length === 0 && (
              <tr><td colSpan={9} className="muted">No brain calls recorded yet.</td></tr>
            )}
            {Object.entries(d.brains).map(([name, s]) => (
              <tr key={name}>
                <td>{name}{d.active === `brain:${name}` && ' '}
                  {d.active === `brain:${name}` && <Pill state="ok">active</Pill>}</td>
                <td>{int(s.calls)}</td>
                <td>{s.failures ? <Pill state="warn">{s.failures}</Pill> : '0'}</td>
                <td>{int(s.out_tokens)}</td>
                <td>{num(s.avg_tok_s)}</td>
                <td>{num(s.p50_generate_s, 2)}s</td>
                <td>{num(s.p95_generate_s, 2)}s</td>
                {/* Averaged over cold calls only: load_s is 0 on a warm one,
                    and including those would hide the real switch cost. */}
                <td>{num(s.avg_cold_load_s)}s</td>
                <td className="muted">{ago(s.last_used)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3>Image models</h3>
      <div style={{ overflowX: 'auto' }}>
        <table>
          <thead>
            <tr>
              <th>Model</th><th>Calls</th><th>Failed</th><th>From cache</th>
              <th>p50</th><th>p95</th><th>Last used</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(d.images).length === 0 && (
              <tr><td colSpan={7} className="muted">No image generations recorded yet.</td></tr>
            )}
            {Object.entries(d.images)
              .sort((a, b) => b[1].calls - a[1].calls)
              .map(([name, s]) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td>{int(s.calls)}</td>
                  <td>{s.failures ? <Pill state="warn">{s.failures}</Pill> : '0'}</td>
                  <td>{int(s.cache_hits)}</td>
                  <td>{s.p50_ms ? `${(s.p50_ms / 1000).toFixed(1)}s` : '—'}</td>
                  <td>{s.p95_ms ? `${(s.p95_ms / 1000).toFixed(1)}s` : '—'}</td>
                  <td className="muted">{ago(s.last_used)}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>

      <h3>Audio</h3>
      <p className="muted">{d.audio_note}.</p>
      <div style={{ overflowX: 'auto' }}>
        <table>
          <thead><tr><th>Kind</th><th>Calls</th><th>Failed</th><th>Avg</th></tr></thead>
          <tbody>
            {d.audio.map((a) => (
              <tr key={a.kind}>
                <td>{a.kind}</td>
                <td>{int(a.calls)}</td>
                <td>{a.failures ? <Pill state="warn">{a.failures}</Pill> : '0'}</td>
                <td>{a.avg_ms ? `${(a.avg_ms / 1000).toFixed(1)}s` : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3>Over time</h3>
      <div className="row" style={{ gap: '.5rem', marginBottom: '.75rem' }}>
        {/* Filters in one row above the charts. */}
        {/* aria-pressed, not opacity alone: a 0.6 vs 1.0 difference is not a
            state a screen reader or a low-contrast display conveys. */}
        {[7, 30, 90].map((n) => (
          <button key={n} className="btn" aria-pressed={days === n}
                  onClick={() => setDays(n)}
                  style={{
                    opacity: days === n ? 1 : 0.6,
                    borderColor: days === n ? 'var(--accent)' : undefined,
                  }}>
            {n} days
          </button>
        ))}
      </div>
      <LineChart title="Generations per day" points={points} series={SERIES}
                 format={(n) => int(n)} />
      <LineChart title="Brain tokens per day"
                 points={tokenPoints}
                 series={[SERIES[1]]} oneSeries
                 format={(n) => (n >= 1000 ? `${(n / 1000).toFixed(0)}k` : String(n))} />

      <p className="muted" style={{ fontSize: '.85rem' }}>
        MODELS_DIR <code>{d.models_dir}</code> · manifest{' '}
        <code>{d.manifest_path}</code> · {d.dead.length} entries are explicitly
        dead and never fetched.
      </p>
    </div>
  )
}

function Tile({ label, value, sub, state }: {
  label: string; value: string; sub?: string; state?: 'ok' | 'bad'
}) {
  return (
    <div className="card" style={{ minWidth: 170 }}>
      <div className="muted" style={{ fontSize: '.8rem' }}>{label}</div>
      {/* The size steps down for long values. A model id is far longer than a
          count, and at 1.5rem "qwen-image-edit" broke across three lines in a
          tile that was otherwise half empty. */}
      <div style={{
        fontSize: value.length > 22 ? '.95rem' : value.length > 13 ? '1.15rem' : '1.5rem',
        fontWeight: 600, lineHeight: 1.25,
        color: state === 'bad' ? 'var(--danger)' : 'var(--text)',
        overflowWrap: 'anywhere',
      }}>{value}</div>
      {sub && <div className="muted" style={{ fontSize: '.78rem' }}>{sub}</div>}
    </div>
  )
}

function ModelTable({ rows, busy, onQueue, active }: {
  rows: ModelEntry[]
  busy: string | null
  onQueue: (entry: string) => void
  active: string | null
}) {
  if (!rows.length) return <p className="muted">None declared.</p>
  return (
    <div style={{ overflowX: 'auto' }}>
      <table>
        <thead>
          <tr>
            <th>Entry</th><th>Kind</th><th>Size</th><th>State</th><th>Source</th><th />
          </tr>
        </thead>
        <tbody>
          {rows.map((m) => (
            <tr key={m.name}>
              <td>
                {m.name}
                {m.repo && active === m.repo && <> <Pill state="ok">active</Pill></>}
                {summary(m.why) && <div className="muted" style={{ fontSize: '.78rem' }}>
                  {summary(m.why)}
                </div>}
              </td>
              {/* nowrap: these columns are narrow enough that the browser
                  broke "hf-cache" into "hf-/cache" and "16.51 GB" across two
                  lines, which reads as corruption rather than wrapping. */}
              <td className="muted" style={{ whiteSpace: 'nowrap' }}>{m.kind}</td>
              <td style={{ whiteSpace: 'nowrap' }}>{gib(m.size_bytes)}</td>
              <td>
                {m.present ? <Pill state="ok">on disk</Pill>
                  : m.kind === 'local' ? <Pill state="idle">trained here</Pill>
                  : m.tier === 'required' ? <Pill state="bad">missing</Pill>
                  : <Pill state="warn">not fetched</Pill>}
                {m.gated && <div className="muted" style={{ fontSize: '.78rem' }}>
                  gated — needs accepted terms
                </div>}
              </td>
              <td className="muted" style={{ fontSize: '.8rem' }}>
                {m.repo ?? m.dest ?? '—'}
                {m.provenance === 'substitute' && <div style={{ color: 'var(--warn)' }}>
                  substitute build
                </div>}
              </td>
              <td>
                {m.downloadable ? (
                  <button className="btn" disabled={busy === m.name}
                          onClick={() => onQueue(m.name)}>
                    {busy === m.name ? 'Queueing…' : 'Download'}
                  </button>
                ) : m.kind === 'local' ? (
                  <span className="muted" style={{ fontSize: '.78rem' }}>copy by hand</span>
                ) : m.fetch_self ? (
                  <span className="muted" style={{ fontSize: '.78rem' }}>fetched on first use</span>
                ) : null}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function DownloadList({ rows, busy, onCancel }: {
  rows: ModelDownload[]
  busy: string | null
  onCancel: (id: string) => void
}) {
  if (!rows.length) return <p className="muted">Nothing queued.</p>
  return (
    <div style={{ overflowX: 'auto' }}>
      <table>
        <thead>
          <tr><th>Entry</th><th>State</th><th>Progress</th><th>By</th><th /></tr>
        </thead>
        <tbody>
          {rows.slice(0, 12).map((r) => (
            <tr key={r.id}>
              <td>{r.entry}</td>
              <td>
                {r.status === 'done' ? <Pill state="ok">done</Pill>
                  : r.status === 'running' ? <Pill state="warn">downloading</Pill>
                  : r.status === 'failed' ? <Pill state="bad">failed</Pill>
                  : <Pill state="idle">{r.status}</Pill>}
                {r.error && <div className="muted" style={{ fontSize: '.78rem' }}>{r.error}</div>}
              </td>
              <td style={{ minWidth: 160 }}>
                {r.progress_pct === null ? (
                  <span className="muted">{gib(r.bytes_done)}</span>
                ) : (
                  <>
                    <div style={{
                      height: 6, borderRadius: 3, background: 'var(--border)',
                      overflow: 'hidden',
                    }}>
                      <div style={{
                        width: `${r.progress_pct}%`, height: '100%', borderRadius: 3,
                        background: r.status === 'failed' ? 'var(--danger)' : 'var(--accent)',
                      }} />
                    </div>
                    {/* A finished row shows the SIZE, not a ratio. `bytes_done`
                        is a disk measurement and a repo whose blobs were
                        already present measures far below its declared total -
                        one read "0.00 GB / 0.04 GB · 100%", which is three
                        numbers disagreeing with each other. */}
                    <span className="muted" style={{ fontSize: '.78rem' }}>
                      {r.status === 'done'
                        ? gib(r.bytes_total ?? r.bytes_done)
                        : `${gib(r.bytes_done)} / ${gib(r.bytes_total)} · ${r.progress_pct}%`}
                    </span>
                  </>
                )}
              </td>
              <td className="muted" style={{ fontSize: '.8rem' }}>{r.requested_by}</td>
              <td>
                {r.status === 'queued' && (
                  <button className="btn" disabled={busy === r.id}
                          onClick={() => onCancel(r.id)}>Cancel</button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
