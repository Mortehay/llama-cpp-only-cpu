import { useState } from 'react'
import { api, getToken, setToken, type NewApiKey } from '../api'
import { useAsync } from '../hooks'

const SCOPES = ['read', 'generate', 'admin'] as const

/**
 * API keys, the stored browser token, and machine info.
 *
 * This tab was permanently blank in the old UI: its container carried an inline
 * `style="display:none"` that the `.active` class could never override.
 */
export default function Settings({ onModeChange }: { onModeChange?: () => void }) {
  const mode = useAsync(() => api.authMode(), [])
  const compute = useAsync(() => api.computeInfo(), [])
  const [token, setTok] = useState(getToken())
  const [saved, setSaved] = useState(false)

  return (
    <>
      <div className={`note ${mode.data?.enforced ? 'ok' : 'warn'}`}>
        {mode.data?.message ?? 'Checking authentication…'}
      </div>

      <ApiKeys
        onChange={() => {
          mode.reload()
          onModeChange?.()
        }}
      />

      <div className="card">
        <h2>This browser's token</h2>
        <p className="hint">
          Stored in localStorage on this machine only, and sent as a bearer token with
          every request. Once you create a key the API stops answering without one —
          paste it here so this UI keeps working.
        </p>
        <div className="row tight">
          <div style={{ flex: '3 1 280px' }}>
            <label htmlFor="tok">Bearer token</label>
            <input
              id="tok"
              type="text"
              value={token}
              placeholder="sk_…"
              onChange={(e) => {
                setTok(e.target.value)
                setSaved(false)
              }}
            />
          </div>
          <div style={{ flex: '0 0 auto' }}>
            <button
              className="btn"
              onClick={() => {
                setToken(token.trim())
                setSaved(true)
                mode.reload()
              }}
            >
              Save
            </button>
          </div>
        </div>
        {saved && <div className="note ok" style={{ marginTop: 10 }}>Token saved.</div>}
      </div>

      <div className="card">
        <h2>Compute</h2>
        {compute.error && <div className="note err">{compute.error}</div>}
        {compute.data && (
          <table>
            <tbody>
              {Object.entries(compute.data).map(([k, v]) => (
                <tr key={k}>
                  <th style={{ width: '38%' }}>{k}</th>
                  <td>{typeof v === 'object' ? JSON.stringify(v) : String(v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  )
}

function ApiKeys({ onChange }: { onChange: () => void }) {
  const keys = useAsync(() => api.listKeys(), [])
  const [name, setName] = useState('')
  const [scopes, setScopes] = useState<string[]>(['read', 'generate'])
  const [minted, setMinted] = useState<NewApiKey | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function create() {
    setBusy(true)
    setError(null)
    try {
      const k = await api.createKey(name.trim(), scopes)
      setMinted(k)
      setName('')
      keys.reload()
      onChange()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  async function revoke(id: string) {
    setBusy(true)
    setError(null)
    try {
      await api.revokeKey(id)
      keys.reload()
      onChange()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="card">
      <h2>API keys</h2>
      <p className="hint">
        Creating the first key switches the API from open to enforced. The first key
        always gets the <code>admin</code> scope, because a first key that cannot
        manage keys locks you out of your own API.
      </p>

      {error && <div className="note err">{error}</div>}

      {minted && (
        <div className="note ok">
          <div>
            <strong>{minted.name}</strong> created
            {minted.bootstrap && ' — admin scope added automatically'}.
          </div>
          <div style={{ margin: '8px 0' }}>
            <code>{minted.token}</code>
          </div>
          <div>
            Copy it now: it is hashed on the server and cannot be shown again. Paste it
            into “This browser's token” below so the UI keeps working.
          </div>
        </div>
      )}

      <div className="row tight">
        <div style={{ flex: '2 1 200px' }}>
          <label htmlFor="k-name">Name</label>
          <input
            id="k-name"
            type="text"
            value={name}
            placeholder="something2"
            onChange={(e) => setName(e.target.value)}
          />
        </div>
        <div style={{ flex: '2 1 240px' }}>
          <label>Scopes</label>
          <div className="row tight">
            {SCOPES.map((s) => (
              <label className="check" key={s} style={{ flex: '0 0 auto' }}>
                <input
                  type="checkbox"
                  checked={scopes.includes(s)}
                  onChange={(e) =>
                    setScopes((cur) =>
                      e.target.checked ? [...cur, s] : cur.filter((x) => x !== s),
                    )
                  }
                />
                {s}
              </label>
            ))}
          </div>
        </div>
        <div style={{ flex: '0 0 auto' }}>
          <button className="btn" disabled={busy || !name.trim() || !scopes.length} onClick={() => void create()}>
            Create key
          </button>
        </div>
      </div>

      <div className="spacer" />

      {keys.error && <div className="note err">{keys.error}</div>}
      {keys.data && keys.data.keys.length === 0 && (
        <div className="empty">No keys yet — the API is open.</div>
      )}
      {keys.data && keys.data.keys.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Prefix</th>
              <th>Scopes</th>
              <th>Last used</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {keys.data.keys.map((k) => (
              <tr key={k.id} style={k.revoked ? { opacity: 0.45 } : undefined}>
                <td>{k.name}</td>
                <td><code>{k.key_prefix}…</code></td>
                <td>{k.scopes.join(', ')}</td>
                <td className="muted">
                  {k.last_used_at ? new Date(k.last_used_at).toLocaleString() : 'never'}
                </td>
                <td style={{ textAlign: 'right' }}>
                  {k.revoked ? (
                    <span className="tag no">revoked</span>
                  ) : (
                    <button className="btn danger sm" disabled={busy} onClick={() => void revoke(k.id)}>
                      Revoke
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
