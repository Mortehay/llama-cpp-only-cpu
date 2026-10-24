import { useCallback, useEffect, useRef, useState } from 'react'
import { api, isAudioKind } from '../api'
import type { ActivityItem, AudioInfo, JobProgress } from '../api'
import { useAsync, usePoll } from '../hooks'

const PAGE = 50

/**
 * Who has asked this machine to generate, what it did, and what is still
 * waiting.
 *
 * NOT CALLED "ACTIONS", and that is not a style preference. `actions.py` and
 * `action_prompts.json` already mean animation actions - walk, idle, attack -
 * and `.ai/domain.md` records "core", "task" and "map" as words this codebase
 * has already overloaded. A fourth would have collided with the one that
 * appears in every spritesheet request.
 *
 * Three producers feed this, and it needs all three or "pending" lies: an API
 * call, a queued job, and a browser task all compete for the same
 * one-at-a-time GPU worker. Showing only the API ledger would report an idle
 * machine while a two-hour sheet held the card.
 *
 * TWO AXES, NOT ONE. `source` answers who asked (api | internal | unknown) and
 * `lane` answers whether a `jobs` row owns the work. They were one column until
 * `.ai/decisions/0017`, where it was really neither - it reported which table
 * owned the artefact, so this browser was filed under "api" (it holds a key,
 * so it writes a ledger row) and the "This browser" option matched only
 * pre-migration-018 history, frozen at 264 rows.
 */
export default function Activity() {
  const [requestedBy, setRequestedBy] = useState('')
  const [lane, setLane] = useState('')
  const [status, setStatus] = useState('')
  const [model, setModel] = useState('')
  const [offset, setOffset] = useState(0)

  const feed = useAsync(
    () =>
      api.activity({
        requested_by: requestedBy || undefined,
        lane: lane || undefined,
        status: status || undefined,
        model: model || undefined,
        limit: PAGE,
        offset,
      }),
    [requestedBy, lane, status, model, offset],
  )

  // ALWAYS POLLING, at two speeds. The condition used to be `busy` itself,
  // which self-latches: `usePoll` creates no interval while its flag is false,
  // and the only thing that could raise the flag was the poll that therefore
  // did not exist. Open this tab on an idle GPU and it never refreshed again -
  // something2 would generate twenty sprites from another machine and the page
  // showed none of them. That is the bug this tab was reported for; the rows
  // were always in the database.
  //
  // The comment this replaces argued a machine at rest should not be asked
  // every four seconds. True, and still honoured by the slow interval - but it
  // was already moot: <ModelGateway /> is mounted outside the tab switch and
  // polls every three seconds on every tab. Same shape as Models.tsx.
  const busy = (feed.data?.active.length ?? 0) > 0
  usePoll(() => feed.reload(), busy ? 4000 : 20000, true)

  const total = feed.data?.total ?? 0
  const active = feed.data?.active ?? []

  return (
    <div className="card">
      <h2>Activity</h2>
      <p className="hint">
        Every generation this machine has been asked for.{' '}
        <strong>Requested by</strong> is the API key that made the call — on this
        network that is the only identity that separates callers, so mint one key
        per consumer. This browser holds a key like any other consumer, so it
        appears by name here rather than as a separate source.
      </p>

      {feed.error && <div className="note err">{feed.error}</div>}

      <PendingPanel active={active} />

      <div className="row tight" style={{ marginBottom: 14 }}>
        {/* WHO ASKED. A dropdown of key names, not a three-value enum: the
            enum could not express "which of my consumers", and something2
            alone holds three keys. Built from `principals`, which ignores the
            current filter - options narrowed by the filter could not be used
            to leave it. */}
        <select
          value={requestedBy}
          onChange={(e) => {
            setRequestedBy(e.target.value)
            setOffset(0)
          }}
          title="The API key that made the call"
        >
          <option value="">Anyone</option>
          {(feed.data?.principals ?? []).map((p) => (
            <option key={p.name} value={p.name}>
              {p.name} ({p.n})
            </option>
          ))}
        </select>

        {/* WHAT RAN IT - a different question from who asked, and the one the
            old `job` source was really answering. */}
        <select
          value={lane}
          onChange={(e) => {
            setLane(e.target.value)
            setOffset(0)
          }}
          title="Whether a job row owns the work (202 and poll), or not"
        >
          <option value="">Any lane</option>
          <option value="job">Job queue</option>
          <option value="sync">Direct call</option>
        </select>

        <select
          value={status}
          onChange={(e) => {
            setStatus(e.target.value)
            setOffset(0)
          }}
        >
          <option value="">Any status</option>
          <option value="running">Running</option>
          <option value="queued">Queued</option>
          <option value="done">Done</option>
          <option value="failed">Failed</option>
          <option value="cancelled">Cancelled</option>
        </select>

        {/* GROUPED, not relabelled. This list holds both rosters and the
            biggest single entry in it is a BRAIN - the tooltip used to read
            "not a language model" above 2060 rows of `qwen3.6-35b-a3b`. The
            Models tab keeps brains, image models and audio in three tables on
            purpose; optgroups are the same rule in one control. */}
        <select
          value={model}
          onChange={(e) => {
            setModel(e.target.value)
            setOffset(0)
          }}
          title="The model that produced it: an image checkpoint, or a brain for text rows"
        >
          <option value="">Any model</option>
          {(['image', 'brain'] as const).map((group) => {
            const rows = (feed.data?.models ?? []).filter(
              (m) => m.model_kind === group,
            )
            if (rows.length === 0) return null
            return (
              <optgroup
                key={group}
                label={group === 'image' ? 'Image models' : 'Brains'}
              >
                {rows.map((m) => (
                  <option key={m.model} value={m.model}>
                    {shortModel(m.model)} ({m.n})
                  </option>
                ))}
              </optgroup>
            )
          })}
        </select>

        <div style={{ flex: "1 1 auto" }} />
        <span className="muted">
          {total} {total === 1 ? 'entry' : 'entries'}
        </span>
      </div>

      {feed.data?.items.length === 0 && (
        <div className="empty">Nothing matches that filter.</div>
      )}

      {feed.data && feed.data.items.length > 0 && (
        <table>
          <thead>
            <tr>
              <th style={{ width: 56 }} />
              <th>What</th>
              <th style={{ width: 110 }}>Source</th>
              <th style={{ width: 110 }}>Status</th>
              <th style={{ width: 170 }} title="The image model that drew it (llm_name) - not a language model">
                Model
              </th>
              <th style={{ width: 200 }}>Requested by</th>
              <th style={{ width: 90 }}>Took</th>
              <th style={{ width: 150 }}>When</th>
            </tr>
          </thead>
          <tbody>
            {feed.data.items.map((it) => (
              <Row key={`${it.source}:${it.id}`} item={it} />
            ))}
          </tbody>
        </table>
      )}

      <div className="row tight" style={{ marginTop: 14 }}>
        <button
          className="btn ghost sm"
          disabled={offset === 0}
          onClick={() => setOffset(Math.max(0, offset - PAGE))}
        >
          ← Newer
        </button>
        <button
          className="btn ghost sm"
          disabled={offset + PAGE >= total}
          onClick={() => setOffset(offset + PAGE)}
        >
          Older →
        </button>
        <div style={{ flex: "1 1 auto" }} />
        <button className="btn ghost sm" onClick={() => feed.reload()}>
          Refresh
        </button>
      </div>
    </div>
  )
}

/**
 * What the GPU is doing and what is behind it.
 *
 * Oldest first, because that is the order the single worker will reach them
 * and therefore the only order that answers "when is mine". It ignores the
 * filters below on purpose: "what is the card busy with" is not a question
 * about the page you happen to be looking at.
 */
function PendingPanel({ active }: { active: ActivityItem[] }) {
  if (active.length === 0) {
    return (
      <div className="note ok">
        Nothing queued or running — the GPU is idle.
      </div>
    )
  }

  const [head, ...rest] = active
  return (
    <div className="note info">
      <strong>
        {active.length} pending
      </strong>{' '}
      — <em>{head.title}</em> ({head.kind}, asked by {head.requested_by}) is
      {head.status === 'running' ? ' on the worker' : ' next in line'}.
      {head.progress && <ProgressNote p={head.progress} />}
      {rest.length > 0 && (
        <ul style={{ margin: '8px 0 0', paddingLeft: 18 }}>
          {rest.map((it) => (
            <li key={`${it.source}:${it.id}`}>
              {it.kind}
              {it.name ? ` "${it.name}"` : ''} — {it.title.slice(0, 80)}
              <span className="muted"> · {it.requested_by}</span>
              {it.progress && <ProgressNote p={it.progress} />}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

/**
 * `58% · cell 75/208`, with the build stage in the tooltip.
 *
 * Both halves come from the worker's own row - the percentage it wrote and the
 * message it wrote - so this cannot drift from what the job reports. Rendered
 * only while a row is unfinished; afterwards the duration column is the better
 * answer and a stale "cell 208/208" would just be noise.
 */
function ProgressNote({ p }: { p: JobProgress }) {
  const bits = [p.pct != null ? `${p.pct}%` : null, p.msg || null].filter(Boolean)
  if (bits.length === 0) return null
  return (
    <div
      className="muted"
      style={{ fontSize: 11, marginTop: 2 }}
      title={p.stage ? `stage: ${p.stage}` : undefined}
    >
      {bits.join(' · ')}
    </div>
  )
}

/** Loop points arrive as sample offsets; shown as seconds, which is what a
 * listener scrubbing to the seam needs. */
function AudioDetails({ a }: { a: AudioInfo }) {
  const secs = (n: number | null) =>
    n != null && a.sample_rate ? `${(n / a.sample_rate).toFixed(2)}s` : '—'
  return (
    <div>
      <strong>Audio:</strong> {a.style ?? '—'}
      {a.bpm != null && ` · ${a.bpm} bpm`}
      {a.time_signature != null && ` · ${a.time_signature}/4`}
      {a.duration_s != null && ` · ${a.duration_s.toFixed(1)}s`}
      {a.sample_rate != null && ` · ${a.sample_rate} Hz`}
      {a.seed != null && ` · seed ${a.seed}`}
      <br />
      <strong>Loop:</strong> {secs(a.loop_start)} → {secs(a.loop_end)}
      {a.seam_rms_jump_db != null && ` · seam jump ${a.seam_rms_jump_db.toFixed(2)} dB`}
      {a.author && (
        <>
          <br />
          <strong>Style chosen by:</strong> {a.author}
        </>
      )}
    </div>
  )
}

/** Why a row says what it says in the Source column. */
function sourceHint(source: string): string {
  if (source === 'api') return 'An API key made this call'
  if (source === 'internal')
    return 'This machine asked itself - a bench, or a brain call from inside a job'
  return 'Recorded before this route was attributed; the caller is not knowable'
}

/**
 * `"<base>+<lora>"` repo ids run past 60 characters; the org prefix is what a
 * row can lose. The full string stays in the tooltip and the details panel.
 */
function shortModel(m: string): string {
  return m
    .split('+')
    .map((p) => p.split('/').pop() || p)
    .join(' + ')
}

/** The preview box, and therefore the footprint the placement math clamps
 *  against. Matches `.preview-pop .shot` in styles.css. */
const PREVIEW = 360

/**
 * The 40px row thumbnail: a preview on hover, a zoomable viewer on click.
 *
 * The ledger stores no separate thumbnail - `item.url` IS the full image, so
 * the row has already downloaded it and the popup costs a layout, not a
 * request. At 40px a sheet cell, a failed cut-out and a contact sheet all look
 * alike, which is the whole reason this column needed one. Both sizes sit on a
 * checkerboard: an opaque backdrop is invisible against a dark cell.
 *
 * `position: fixed` with measured coordinates rather than a CSS-hover sibling:
 * the popup is six times the width of its 56px column, and a table cell clips
 * it. Escape or a scroll closes it - a fixed popup does not travel with the
 * row it belongs to.
 *
 * The hover popup deliberately does NOT zoom. It is a glance, and a 336-cell
 * sheet scaled into 360px is unreadable whatever you do to it; clicking hands
 * that case to <Lightbox>, which is also the only route a touch device has.
 */
function Thumb({ url, alt }: { url: string; alt: string }) {
  const [at, setAt] = useState<{ x: number; y: number } | null>(null)
  const [zoomed, setZoomed] = useState(false)
  const [size, setSize] = useState<string | null>(null)
  const ref = useRef<HTMLButtonElement>(null)

  const open = useCallback(() => {
    const el = ref.current
    if (!el) return
    const r = el.getBoundingClientRect()
    const box = PREVIEW + 18 // the shot plus the card's padding and border
    // Right of the cell by default, flipped to the left when that would run
    // off the viewport.
    let x = r.right + 12
    if (x + box > window.innerWidth) x = Math.max(8, r.left - box - 12)
    const y = Math.max(
      8,
      Math.min(r.top + r.height / 2 - box / 2, window.innerHeight - box - 8),
    )
    setAt({ x, y })
  }, [])

  const close = useCallback(() => setAt(null), [])

  useEffect(() => {
    if (!at) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close()
    }
    window.addEventListener('keydown', onKey)
    // Capture phase, so an inner scroller counts too, not just the page.
    window.addEventListener('scroll', close, true)
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('scroll', close, true)
    }
  }, [at, close])

  return (
    <>
      <button
        ref={ref}
        type="button"
        className="rowthumb"
        onMouseEnter={open}
        onMouseLeave={close}
        onFocus={open}
        onBlur={close}
        onClick={() => {
          close()
          setZoomed(true)
        }}
        title="Click to open it full size"
      >
        <img src={url} alt="" loading="lazy" />
      </button>

      {at && (
        <div
          className="preview-pop"
          style={{
            left: at.x,
            top: at.y,
            // Hover decoration only: taking the pointer would let it steal the
            // mouseleave that closes it.
            pointerEvents: 'none',
          }}
        >
          <div className="shot">
            <img
              src={url}
              alt={alt}
              onLoad={(e) =>
                setSize(`${e.currentTarget.naturalWidth} x ${e.currentTarget.naturalHeight}`)
              }
            />
          </div>
          <div className="cap">{size ? `${size} - click to zoom` : 'loading...'}</div>
        </div>
      )}

      {zoomed && <Lightbox url={url} alt={alt} onClose={() => setZoomed(false)} />}
    </>
  )
}

/** Zoom ceiling, in multiples of the fit scale. 24x is enough to put one
 *  48x64 cell of a 336-cell sheet across the stage. */
const MAX_ZOOM = 24

/**
 * The full-size viewer: wheel to zoom, drag to pan, buttons for the rest.
 *
 * Sized in SCALE, not in CSS `object-fit`, because the two questions asked of
 * these images need different scales and only one of them is "fit". A sheet is
 * 10k pixels wide and has to be zoomed INTO before a cell is legible; a 48x64
 * sprite has to be zoomed OUT of, to 1:1, before you can tell whether the
 * pixel grid survived. So `fit` is measured from the stage and the natural
 * size, and `zoom` multiplies it - which also makes "1:1" exact rather than
 * approximately right.
 *
 * Zoom is anchored at the pointer: wheeling over a cell keeps that cell under
 * the cursor instead of flying off toward the centre, which on a 336-cell
 * sheet is the difference between inspecting a frame and hunting for it.
 */
function Lightbox({
  url,
  alt,
  onClose,
}: {
  url: string
  alt: string
  onClose: () => void
}) {
  const stage = useRef<HTMLDivElement>(null)
  const [nat, setNat] = useState<{ w: number; h: number } | null>(null)
  const [box, setBox] = useState({ w: 0, h: 0 })
  const [zoom, setZoom] = useState(1)
  const [off, setOff] = useState({ x: 0, y: 0 })
  const [drag, setDrag] = useState(false)
  const from = useRef<{ x: number; y: number; ox: number; oy: number } | null>(null)

  // The stage is sized in vw/vh, so its pixel size is only knowable after
  // layout - and it is what `fit` divides by.
  useEffect(() => {
    const el = stage.current
    if (!el) return
    const measure = () => setBox({ w: el.clientWidth, h: el.clientHeight })
    measure()
    window.addEventListener('resize', measure)
    return () => window.removeEventListener('resize', measure)
  }, [])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const fit = nat && box.w ? Math.min(box.w / nat.w, box.h / nat.h) : 1
  // Zooming out past the fit is pointless EXCEPT for a sprite small enough
  // that fitting it means magnifying it - there, 1:1 is below the fit.
  const minZoom = Math.min(1, 1 / fit)
  const scale = fit * zoom
  const dispW = nat ? nat.w * scale : 0
  const dispH = nat ? nat.h * scale : 0

  /** Keep the image overlapping the stage: pan stops at its own edges. */
  const clampOff = useCallback(
    (o: { x: number; y: number }, w: number, h: number) => {
      const mx = Math.max(0, (w - box.w) / 2)
      const my = Math.max(0, (h - box.h) / 2)
      return {
        x: Math.max(-mx, Math.min(mx, o.x)),
        y: Math.max(-my, Math.min(my, o.y)),
      }
    },
    [box],
  )

  /** Zoom to `next`, holding the image point under (sx, sy) in place. */
  const zoomAt = useCallback(
    (next: number, sx: number, sy: number) => {
      const el = stage.current
      if (!el || !nat) return
      const r = el.getBoundingClientRect()
      const nz = Math.max(minZoom, Math.min(MAX_ZOOM, next))
      const k = nz / zoom
      const d = { x: sx - (r.left + r.width / 2), y: sy - (r.top + r.height / 2) }
      setZoom(nz)
      setOff(
        clampOff(
          { x: d.x - k * (d.x - off.x), y: d.y - k * (d.y - off.y) },
          nat.w * fit * nz,
          nat.h * fit * nz,
        ),
      )
    },
    [clampOff, fit, minZoom, nat, off, zoom],
  )

  /** Buttons and double-click zoom about the stage centre. */
  const zoomBy = useCallback(
    (factor: number) => {
      const el = stage.current
      if (!el) return
      const r = el.getBoundingClientRect()
      zoomAt(zoom * factor, r.left + r.width / 2, r.top + r.height / 2)
    },
    [zoom, zoomAt],
  )

  // Native listener, not onWheel: React's wheel handler is passive, so
  // preventDefault there is ignored and the page scrolls behind the overlay
  // while you are trying to zoom.
  useEffect(() => {
    const el = stage.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      // Exponential, so a notch is the same proportional step at every zoom.
      zoomAt(zoom * Math.exp(-e.deltaY / 400), e.clientX, e.clientY)
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [zoom, zoomAt])

  function reset() {
    setZoom(1)
    setOff({ x: 0, y: 0 })
  }

  function oneToOne() {
    if (!nat) return
    setZoom(Math.max(minZoom, Math.min(MAX_ZOOM, 1 / fit)))
    setOff({ x: 0, y: 0 })
  }

  return (
    <div
      className="lightbox"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose()
      }}
    >
      <div className="panel">
        <div className="toolbar">
          <span>
            {nat ? `${nat.w} x ${nat.h}` : 'loading...'}
            {nat && ` · ${Math.round(scale * 100)}%`}
          </span>
          <div style={{ flex: '1 1 auto' }} />
          <button className="btn ghost zoom" onClick={() => zoomBy(1 / 1.5)} title="Zoom out">
            −
          </button>
          <button className="btn ghost zoom" onClick={() => zoomBy(1.5)} title="Zoom in">
            +
          </button>
          <button className="btn ghost" onClick={reset} title="Fit to the window">
            Fit
          </button>
          <button className="btn ghost" onClick={oneToOne} title="One image pixel per screen pixel">
            1:1
          </button>
          <button className="btn ghost" onClick={onClose}>
            Close
          </button>
        </div>

        <div
          ref={stage}
          className={`stage${drag ? ' dragging' : ''}`}
          onPointerDown={(e) => {
            // Capture, so a drag that leaves the stage keeps delivering moves
            // instead of stranding the image mid-pan.
            ;(e.currentTarget as Element).setPointerCapture?.(e.pointerId)
            from.current = { x: e.clientX, y: e.clientY, ox: off.x, oy: off.y }
            setDrag(true)
          }}
          onPointerMove={(e) => {
            const f = from.current
            if (!f) return
            setOff(
              clampOff(
                { x: f.ox + (e.clientX - f.x), y: f.oy + (e.clientY - f.y) },
                dispW,
                dispH,
              ),
            )
          }}
          onPointerUp={() => {
            from.current = null
            setDrag(false)
          }}
          onPointerCancel={() => {
            from.current = null
            setDrag(false)
          }}
          onDoubleClick={(e) => zoomAt(zoom > 1.01 ? 1 : 1 / fit, e.clientX, e.clientY)}
        >
          <img
            src={url}
            alt={alt}
            draggable={false}
            onLoad={(e) =>
              setNat({
                w: e.currentTarget.naturalWidth,
                h: e.currentTarget.naturalHeight,
              })
            }
            style={
              nat
                ? {
                    width: dispW,
                    height: dispH,
                    left: (box.w - dispW) / 2 + off.x,
                    top: (box.h - dispH) / 2 + off.y,
                  }
                : { visibility: 'hidden' }
            }
          />
        </div>

        <div className="cap">
          Scroll to zoom · drag to pan · double-click for 1:1 · Esc closes
        </div>
      </div>
    </div>
  )
}

function Row({ item }: { item: ActivityItem }) {
  const [open, setOpen] = useState(false)

  return (
    <>
      <tr>
        <td>
          {item.url && isAudioKind(item.kind) ? (
            /* The open /audio mount, so no bearer is needed. preload="none":
               a page of 50 rows must not fetch 50 tracks to render. */
            <audio controls preload="none" src={item.url} style={{ width: 220, height: 32 }} />
          ) : item.url ? (
            <Thumb url={item.url} alt={item.title} />
          ) : (
            <span className="muted">—</span>
          )}
        </td>
        <td>
          <button
            className="btn ghost sm"
            style={{ padding: 0, background: 'none', border: 'none' }}
            onClick={() => setOpen(!open)}
            title="Show the details"
          >
            {item.name ? <strong>{item.name}</strong> : null}
            {item.name ? ' — ' : ''}
            {item.title.length > 90 ? `${item.title.slice(0, 90)}…` : item.title}
          </button>{' '}
          <span className="tag neutral">{item.kind}</span>{' '}
          {item.served_from === 'cache' && (
            /* A cache read cost no GPU. Without saying so, the duration column
               below reads as a suspiciously fast generation. */
            <span className="tag ok" title="Served from disk — no GPU time">
              cached
            </span>
          )}
        </td>
        {/* Two facts, one narrow column. `source` alone said "api" for almost
            everything once the browser got a key, and the lane is what someone
            scanning for a long build is looking for. `unknown` is not a gap in
            the data - those rows predate attribution and cannot acquire it. */}
        <td className="muted" title={sourceHint(item.source)}>
          {item.source}
          {item.lane === 'job' && (
            <>
              {' '}
              <span className="tag neutral">job</span>
            </>
          )}
        </td>
        <td>
          <span
            className={`tag ${
              item.status === 'done'
                ? 'ok'
                : item.status === 'failed' || item.status === 'cancelled'
                  ? 'no'
                  : 'neutral'
            }`}
          >
            {item.status}
          </span>
          {/* A sheet job is hours of GPU time and this column showed it as the
              bare word "running" for all of them - which reads exactly like
              wedged. The counter comes from the job row; nothing here derives
              it. */}
          {item.progress && <ProgressNote p={item.progress} />}
        </td>
        <td className="muted" title={item.model ?? undefined}>
          {item.model ? shortModel(item.model) : '—'}
        </td>
        <td className="muted">{item.requested_by}</td>
        <td className="muted">
          {item.duration_ms != null ? `${(item.duration_ms / 1000).toFixed(1)}s` : '—'}
        </td>
        <td className="muted">
          {item.created_at ? new Date(item.created_at).toLocaleString() : '—'}
        </td>
      </tr>

      {open && (
        <tr>
          <td colSpan={8} style={{ background: 'rgba(255,255,255,.02)' }}>
            <div style={{ fontSize: 12, lineHeight: 1.6 }}>
              <div>
                <strong>Prompt:</strong> {item.title}
              </div>
              {item.model && (
                <div>
                  <strong>Model:</strong> {item.model}
                </div>
              )}
              <div>
                <strong>Requested by:</strong> {item.requested_by}
              </div>
              <div>
                {/* Never labelled "IP". Measured: every external request
                    reaches this process as 172.18.0.1, the docker bridge
                    gateway, and LAN traffic is rewritten again by the Windows
                    portproxy. Two machines on the Wi-Fi are indistinguishable
                    here, so the key above is the identity that counts. */}
                <strong>Seen from:</strong>{' '}
                {item.client_addr ?? '—'}{' '}
                <span className="muted">
                  (gateway address, not the caller&apos;s — every LAN client
                  arrives through the same NAT hop)
                </span>
              </div>
              {item.forwarded_for && (
                <div>
                  {/* Shown only when the caller sent it, and labelled as
                      claimed rather than observed: no proxy sits in front of
                      this service, so nothing verifies this header. */}
                  <strong>Claimed origin:</strong> {item.forwarded_for}{' '}
                  <span className="muted">
                    (X-Forwarded-For, sent by the caller and not verified)
                  </span>
                </div>
              )}
              {item.job_id && (
                <div>
                  <strong>Job:</strong> {item.job_id}
                </div>
              )}
              {item.audio && <AudioDetails a={item.audio} />}
              {item.error && (
                <div className="note err" style={{ marginTop: 8 }}>
                  {item.error}
                </div>
              )}
            </div>
          </td>
        </tr>
      )}
    </>
  )
}
