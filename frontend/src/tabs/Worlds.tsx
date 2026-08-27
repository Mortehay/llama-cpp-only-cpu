import { useState } from 'react'
import { api, getToken, type WorldReport } from '../api'
import { useAsync, useAuthedObjectUrl } from '../hooks'

/**
 * something2 world/region specs.
 *
 * The thing this screen exists to settle is "will it feel empty", and it
 * answers that BEFORE anything is seeded. Every world is previewed as one
 * screenful, so the dots you see are literally the creatures-per-screen number
 * next to it.
 *
 * Density is solved, not named. something2's `normal` tier is ~4 creatures per
 * screen and `biomes.creature_density` multiplies it - Meadow halves it to ~2,
 * which is where "too much empty space" comes from. Asking for a target and
 * letting the generator pick the tier per biome is the fix.
 */

/** Fetch with the bearer, then hand the browser a file. A plain link 401s. */
async function download(path: string, filename: string) {
  const res = await fetch(path, {
    headers: getToken() ? { Authorization: `Bearer ${getToken()}` } : {},
  })
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`)
  const url = URL.createObjectURL(await res.blob())
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

export default function Worlds() {
  const list = useAsync(() => api.worlds(), [])

  const [name, setName] = useState('emerald-reach')
  const [count, setCount] = useState(6)
  const [target, setTarget] = useState(6)
  const [size, setSize] = useState(128)
  const [theme, setTheme] = useState('a green river valley giving way to cold highlands')
  const [author, setAuthor] = useState<'rules' | 'llm'>('rules')

  const [made, setMade] = useState<string | null>(null)
  const [report, setReport] = useState<WorldReport | null>(null)
  const [note, setNote] = useState<string | null>(null)
  const [seedWith, setSeedWith] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Edit state. Separate from the create form on purpose: the create form
  // describes a region that does not exist yet, while these describe changes
  // to one that does, and conflating them made it unclear which button would
  // overwrite what.
  const [editable, setEditable] = useState(true)
  const [eTarget, setETarget] = useState<number | ''>('')
  const [eWorlds, setEWorlds] = useState<number | ''>('')
  const [eSize, setESize] = useState<number | ''>('')
  const [reauthor, setReauthor] = useState(false)

  // Cache-buster for the preview: a PATCH rewrites the PNG at the same URL, so
  // without this the browser shows the old region's picture next to the new
  // region's numbers - which is worse than showing nothing.
  const [previewNonce, setPreviewNonce] = useState(0)

  const preview = useAuthedObjectUrl(
    made ? `/api/worlds/${made}/preview.png?v=${previewNonce}` : null,
  )

  async function generate(overwrite = false) {
    setBusy(true)
    setError(null)
    try {
      const res = await api.createWorld({
        name: name.trim(),
        worlds: count,
        target_per_screen: target,
        size,
        theme: theme.trim() || null,
        author,
        overwrite,
      })
      setMade(res.name)
      setReport(res.report)
      setNote(res.author)
      setSeedWith(res.seed_with)
      setEditable(true)
      setETarget('')
      setEWorlds('')
      setESize('')
      setPreviewNonce((n) => n + 1)
      list.reload()
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      setError(msg)
    } finally {
      setBusy(false)
    }
  }

  /** PATCH: send only the fields actually filled in. */
  async function applyEdit() {
    if (!made) return
    setBusy(true)
    setError(null)
    try {
      const res = await api.editWorld(made, {
        ...(eTarget === '' ? {} : { target_per_screen: Number(eTarget) }),
        ...(eWorlds === '' ? {} : { worlds: Number(eWorlds) }),
        ...(eSize === '' ? {} : { size: Number(eSize) }),
        ...(reauthor ? { reauthor: true } : {}),
      })
      setReport(res.report)
      setNote(
        res.changed.length
          ? `${res.author} — changed ${res.changed.join(', ')}`
          : 'nothing named, so nothing changed',
      )
      setPreviewNonce((n) => n + 1)
      setETarget('')
      setEWorlds('')
      setESize('')
      setReauthor(false)
      list.reload()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  /** Open an existing region from the list, prefilled from its stored params. */
  function open(w: { name: string; editable?: boolean; params?: Record<string, unknown> | null }) {
    setMade(w.name)
    setNote(null)
    setError(null)
    setEditable(w.editable !== false)
    setSeedWith(`make seed-map SPEC=${w.name}`)
    setPreviewNonce((n) => n + 1)
    setETarget('')
    setEWorlds('')
    setESize('')
    void api.worldReport(w.name).then(setReport).catch((e) =>
      setError(e instanceof Error ? e.message : String(e)),
    )
  }

  const band =
    target < 3 ? 'reads as empty space' : target > 14 ? 'a crowd' : 'lively'

  return (
    <>
      <div className="card">
        <h2>World / region generator</h2>
        <p className="hint">
          Produces a <code>*.map.json</code> something2 seeds with{' '}
          <code>make seed-map</code>. Density is <strong>solved, not named</strong>:
          their <code>normal</code> tier is ~4 creatures per screen and a biome
          multiplier scales it — Meadow halves it to ~2, which is exactly where
          empty space comes from. Ask for a target and each world gets the tier
          that actually hits it.
        </p>

        {error && <div className="note err">{error}</div>}

        <div className="row tight">
          <div style={{ flex: '2 1 200px' }}>
            <label htmlFor="w-name">Region name</label>
            <input id="w-name" value={name} onChange={(e) => setName(e.target.value)} />
            <span className="muted">Becomes SPEC=&lt;name&gt;.</span>
          </div>
          <div style={{ flex: '1 1 110px' }}>
            <label htmlFor="w-n">Worlds</label>
            <input
              id="w-n"
              type="number"
              min={1}
              max={36}
              value={count}
              onChange={(e) => setCount(Number(e.target.value))}
            />
          </div>
          <div style={{ flex: '1 1 110px' }}>
            <label htmlFor="w-size">World size</label>
            <input
              id="w-size"
              type="number"
              min={32}
              max={224}
              step={32}
              value={size}
              onChange={(e) => setSize(Number(e.target.value))}
            />
          </div>
          <div style={{ flex: '1 1 150px' }}>
            <label htmlFor="w-author">Biome author</label>
            <select
              id="w-author"
              value={author}
              onChange={(e) => setAuthor(e.target.value as 'rules' | 'llm')}
            >
              <option value="rules">rules (deterministic)</option>
              <option value="llm">LLM (falls back if absent)</option>
            </select>
          </div>
        </div>

        {/* The slider gets a row to itself. Sharing one with the theme field
            squeezed a full-width control and a free-text input into the same
            line, and both ended up too small to use. */}
        <div style={{ marginTop: 14 }}>
          <label htmlFor="w-target">
            Creatures per screen: <strong>{target}</strong> — {band}
          </label>
          <input
            id="w-target"
            type="range"
            min={0.5}
            max={25}
            step={0.5}
            value={target}
            onChange={(e) => setTarget(Number(e.target.value))}
            style={{ width: '100%', display: 'block' }}
          />
          <span className="muted">
            their tiers: sparse ≈ 2 · normal ≈ 4 · dense ≈ 8 · horde ≈ 14 · swarm ≈ 20
            {' · '}under 3 reads as empty space
          </span>
        </div>

        <div style={{ marginTop: 14 }}>
          <label htmlFor="w-theme">Theme (used only by the LLM author)</label>
          <input
            id="w-theme"
            value={theme}
            onChange={(e) => setTheme(e.target.value)}
            style={{ width: '100%' }}
            disabled={author !== 'llm'}
            placeholder="a green river valley that climbs into frozen highlands"
          />
        </div>

        <div className="row" style={{ marginTop: 12 }}>
          <button className="btn" disabled={busy} onClick={() => void generate(false)}>
            {busy ? 'Generating…' : 'Generate region'}
          </button>
          <button className="btn ghost" disabled={busy} onClick={() => void generate(true)}>
            Regenerate (overwrite)
          </button>
        </div>
      </div>

      {report && made && (
        <div className="card">
          <h2>{made}</h2>
          {note && <div className="note">{note}</div>}

          <div className="row tight" style={{ marginBottom: 10 }}>
            <span className={`tag ${report.ok ? 'ok' : 'no'}`}>
              {report.ok ? 'no problems' : `${report.problems.length} problem(s)`}
            </span>{' '}
            <span className="muted">
              {report.totals.worlds} worlds · {report.totals.creatures} creatures ·{' '}
              {report.totals.min_per_screen}–{report.totals.max_per_screen} per screen
              {report.totals.empty_worlds > 0 &&
                ` · ${report.totals.empty_worlds} EMPTY`}
            </span>
          </div>

          {report.problems.map((p, i) => (
            <div className="note err" key={i}>
              {p}
            </div>
          ))}

          {/* Edit. Only fields you fill are sent; everything else carries
              over, biomes included — so raising the target does not redraw the
              region's character. */}
          {editable ? (
            <div className="row tight" style={{ marginBottom: 12 }}>
              <div style={{ flex: '1 1 150px' }}>
                <label htmlFor="e-target">Change /screen</label>
                <input
                  id="e-target"
                  type="number"
                  min={0.5}
                  max={25}
                  step={0.5}
                  placeholder="unchanged"
                  value={eTarget}
                  onChange={(e) =>
                    setETarget(e.target.value === '' ? '' : Number(e.target.value))
                  }
                />
              </div>
              <div style={{ flex: '1 1 120px' }}>
                <label htmlFor="e-worlds">Change worlds</label>
                <input
                  id="e-worlds"
                  type="number"
                  min={1}
                  max={36}
                  placeholder="unchanged"
                  value={eWorlds}
                  onChange={(e) =>
                    setEWorlds(e.target.value === '' ? '' : Number(e.target.value))
                  }
                />
              </div>
              <div style={{ flex: '1 1 120px' }}>
                <label htmlFor="e-size">Change size</label>
                <input
                  id="e-size"
                  type="number"
                  min={32}
                  max={224}
                  step={32}
                  placeholder="unchanged"
                  value={eSize}
                  onChange={(e) =>
                    setESize(e.target.value === '' ? '' : Number(e.target.value))
                  }
                />
              </div>
              <div style={{ flex: '0 0 auto', alignSelf: 'flex-end' }}>
                <label style={{ display: 'block' }}>
                  <input
                    type="checkbox"
                    checked={reauthor}
                    onChange={(e) => setReauthor(e.target.checked)}
                  />{' '}
                  re-pick biomes
                </label>
                <button className="btn" disabled={busy} onClick={() => void applyEdit()}>
                  {busy ? 'Applying…' : 'Apply change'}
                </button>
              </div>
            </div>
          ) : (
            <div className="note">
              Generated before edits were supported, so it has no stored
              parameters. It can be downloaded and replaced, but not patched —
              regenerate it above to make it editable.
            </div>
          )}

          {preview.error && <div className="note err">{preview.error}</div>}
          {preview.url && (
            <div style={{ overflow: 'auto', marginTop: 10 }}>
              <img src={preview.url} alt={made} style={{ maxWidth: '100%' }} />
            </div>
          )}

          <div style={{ overflowX: 'auto', marginTop: 12 }}>
            <table className="rows">
              <thead>
                <tr>
                  <th>world</th>
                  <th>depth</th>
                  <th>biomes</th>
                  <th>tier</th>
                  <th>×biome</th>
                  <th>/screen</th>
                  <th>creatures</th>
                  <th>kinds</th>
                  <th>flora</th>
                  <th>KiB/s</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {report.worlds.map((w) => (
                  <tr key={w.key}>
                    <td>{w.name}</td>
                    <td className="muted">{w.depth ?? '—'}</td>
                    <td>{w.biomes.join(', ')}</td>
                    <td>{w.density}</td>
                    <td>×{w.biome_multiplier}</td>
                    <td>
                      <strong>{w.per_screen}</strong>
                    </td>
                    <td>{w.creatures}</td>
                    <td className="muted">{w.variety}</td>
                    <td className="muted">{w.flora.length}</td>
                    <td className="muted">{w.socket_kib_s}</td>
                    <td>
                      {w.verdict !== 'ok' && (
                        <span className="tag no">{w.verdict}</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {report.notes.map((n, i) => (
            <p className="muted" key={i}>
              {n}
            </p>
          ))}

          <div className="row" style={{ marginTop: 12 }}>
            <button
              className="btn"
              onClick={() =>
                void download(`/api/worlds/${made}?download=true`, `${made}.map.json`)
              }
            >
              Download spec
            </button>
            <button
              className="btn ghost"
              onClick={() =>
                void download(`/api/worlds/${made}/preview.png`, `${made}.preview.png`)
              }
            >
              Download preview
            </button>
          </div>
          {seedWith && (
            <p className="muted">
              Then, in something2: <code>{seedWith}</code>
            </p>
          )}
        </div>
      )}

      <div className="card">
        <h2>Generated regions</h2>
        <p className="hint">
          something2 reads these over the API — <code>GET /api/worlds</code> to list,{' '}
          <code>/api/worlds/&lt;name&gt;</code> for the spec,{' '}
          <code>/preview.png</code> to look first.
        </p>
        {/* A failed listing must SAY so. Rendering only on `list.data` left an
            unauthenticated tab looking like an empty one, which is the same
            picture as "you have generated nothing" and a different problem. */}
        {list.error && <div className="note err">{list.error}</div>}
        {list.loading && !list.data && <div className="empty">Loading…</div>}
        {list.data && list.data.items.length === 0 && (
          <div className="empty">Nothing generated yet.</div>
        )}
        <div style={{ overflowX: 'auto' }}>
          <table className="rows">
            <tbody>
              {list.data?.items.map((w) => (
                <tr key={w.name}>
                  <td>
                    <strong>{w.name}</strong>
                  </td>
                  <td className="muted">{w.worlds} worlds</td>
                  <td className="muted">{w.creatures} creatures</td>
                  <td className="muted">{w.mean_per_screen}/screen</td>
                  <td>
                    {w.empty_worlds ? (
                      <span className="tag no">{w.empty_worlds} empty</span>
                    ) : (
                      <span className="tag neutral">ok</span>
                    )}
                  </td>
                  <td>
                    <button className="btn ghost sm" onClick={() => open(w)}>
                      Open
                    </button>
                  </td>
                  <td>
                    <button
                      className="btn danger sm"
                      onClick={() => {
                        if (!window.confirm(`Delete region ${w.name}?`)) return
                        void api.deleteWorld(w.name).then(() => {
                          if (made === w.name) {
                            setMade(null)
                            setReport(null)
                          }
                          list.reload()
                        })
                      }}
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  )
}
