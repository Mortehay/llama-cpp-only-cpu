import { useState } from 'react'
import { api, isAudioKind } from '../api'
import type { ActivityItem, AudioInfo } from '../api'
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
 */
export default function Activity() {
  const [source, setSource] = useState('')
  const [status, setStatus] = useState('')
  const [model, setModel] = useState('')
  const [offset, setOffset] = useState(0)

  const feed = useAsync(
    () =>
      api.activity({
        source: source || undefined,
        status: status || undefined,
        model: model || undefined,
        limit: PAGE,
        offset,
      }),
    [source, status, model, offset],
  )

  // Poll only while something is moving. A machine at rest does not need a
  // request every four seconds, and the GPU worker is the thing that changes.
  const busy = (feed.data?.active.length ?? 0) > 0
  usePoll(() => feed.reload(), 4000, busy)

  const total = feed.data?.total ?? 0
  const active = feed.data?.active ?? []

  return (
    <div className="card">
      <h2>Activity</h2>
      <p className="hint">
        Every generation this machine has been asked for, from the API, the job
        queue and this browser. <strong>Requested by</strong> is the API key that
        made the call — on this network that is the only identity that separates
        callers, so mint one key per consumer.
      </p>

      {feed.error && <div className="note err">{feed.error}</div>}

      <PendingPanel active={active} />

      <div className="row tight" style={{ marginBottom: 14 }}>
        <select
          value={source}
          onChange={(e) => {
            setSource(e.target.value)
            setOffset(0)
          }}
        >
          <option value="">All sources</option>
          <option value="api">API (something2 and scripts)</option>
          <option value="job">Job queue</option>
          <option value="ui">This browser</option>
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

        <select
          value={model}
          onChange={(e) => {
            setModel(e.target.value)
            setOffset(0)
          }}
          title="The image model that drew it (llm_name) - not a language model"
        >
          <option value="">Any model</option>
          {(feed.data?.models ?? []).map((m) => (
            <option key={m.model} value={m.model}>
              {shortModel(m.model)} ({m.n})
            </option>
          ))}
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
      {rest.length > 0 && (
        <ul style={{ margin: '8px 0 0', paddingLeft: 18 }}>
          {rest.map((it) => (
            <li key={`${it.source}:${it.id}`}>
              {it.kind}
              {it.name ? ` "${it.name}"` : ''} — {it.title.slice(0, 80)}
              <span className="muted"> · {it.requested_by}</span>
            </li>
          ))}
        </ul>
      )}
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
            <img
              src={item.url}
              alt=""
              loading="lazy"
              style={{ width: 40, height: 40, objectFit: 'contain' }}
            />
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
        <td className="muted">{item.source}</td>
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
