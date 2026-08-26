import { useEffect, useState } from 'react'
import { api, getToken, setToken, type NewApiKey } from '../api'
import { useAsync } from '../hooks'

const SCOPES = ['read', 'generate', 'admin'] as const

/** Scopes the browser's own token is minted with. See `generate` below. */
const BROWSER_SCOPES = ['read', 'generate', 'admin']

/**
 * Copy that also works off localhost.
 *
 * `navigator.clipboard` exists only in a secure context, and this UI is reached
 * over plain http on a LAN address as often as not - which is precisely when
 * you need to move a token to another machine. So the deprecated execCommand
 * path is not a legacy fallback here, it is the one that runs.
 */
async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text)
      return true
    }
  } catch {
    /* denied or insecure context - try the old way before giving up */
  }
  try {
    const ta = document.createElement('textarea')
    ta.value = text
    ta.setAttribute('readonly', '')
    ta.style.position = 'fixed'
    ta.style.opacity = '0'
    document.body.appendChild(ta)
    ta.select()
    const ok = document.execCommand('copy')
    document.body.removeChild(ta)
    return ok
  } catch {
    return false
  }
}

/** Copies `value`, and says so for two seconds. */
function CopyButton({ value, className = 'btn ghost' }: { value: string; className?: string }) {
  const [state, setState] = useState<'idle' | 'ok' | 'fail'>('idle')

  // Reset through an effect rather than a bare setTimeout, so a button that
  // unmounts while "Copied" is showing does not set state on a dead component.
  useEffect(() => {
    if (state === 'idle') return
    const t = window.setTimeout(() => setState('idle'), 2000)
    return () => window.clearTimeout(t)
  }, [state])

  return (
    <button
      className={className}
      disabled={!value}
      onClick={() => void copyText(value).then((ok) => setState(ok ? 'ok' : 'fail'))}
    >
      {state === 'ok' ? 'Copied' : state === 'fail' ? 'Copy failed' : 'Copy'}
    </button>
  )
}

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
  // What is actually in localStorage, which is not the same as what is in the
  // box: Copy must hand over the token in use, never an unsaved edit of it.
  const [stored, setStored] = useState(getToken())
  const [saved, setSaved] = useState(false)
  const [minting, setMinting] = useState(false)
  const [mintError, setMintError] = useState<string | null>(null)
  // Bumped so the key table above re-reads after a token is minted down here.
  const [keyNonce, setKeyNonce] = useState(0)

  /** Mint a key and adopt it as this browser's token in one step. */
  async function generate() {
    setMinting(true)
    setMintError(null)
    try {
      const stamp = new Date().toISOString().slice(0, 16).replace('T', ' ')
      const k = await api.createKey(`browser · ${stamp}`, BROWSER_SCOPES)
      // Saved to localStorage immediately, not on a second click of Save: the
      // server hashes the token and will never show it again, so a generated
      // token that is only sitting in a text box is one refresh from lost.
      setTok(k.token)
      setToken(k.token)
      setStored(k.token)
      setSaved(true)
      setKeyNonce((n) => n + 1)
      mode.reload()
      onModeChange?.()
    } catch (e) {
      setMintError(e instanceof Error ? e.message : String(e))
    } finally {
      setMinting(false)
    }
  }

  return (
    <>
      <div className={`note ${mode.data?.enforced ? 'ok' : 'warn'}`}>
        {mode.data?.message ?? 'Checking authentication…'}
      </div>

      <ApiKeys
        nonce={keyNonce}
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
        {mintError && <div className="note err">{mintError}</div>}

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
              disabled={minting}
              onClick={() => {
                setToken(token.trim())
                setStored(token.trim())
                setSaved(true)
                mode.reload()
              }}
            >
              Save
            </button>
          </div>
          <div style={{ flex: '0 0 auto' }}>
            <button className="btn ghost" disabled={minting} onClick={() => void generate()}>
              {minting ? 'Generating…' : 'Generate'}
            </button>
          </div>
          {/* Only once something is stored: a Copy button over an empty box is
              a button that can only disappoint. */}
          {stored !== '' && (
            <div style={{ flex: '0 0 auto' }}>
              <CopyButton value={stored} />
            </div>
          )}
        </div>

        {saved && (
          <div className="note ok" style={{ marginTop: 10 }}>
            Token saved to this browser. <strong>Copy</strong> puts it on the clipboard
            for another service.
          </div>
        )}

        <div className="muted" style={{ marginTop: 10 }}>
          <strong>Generate</strong> mints a new key with{' '}
          <code>{BROWSER_SCOPES.join(', ')}</code> and stores it here. It includes{' '}
          <code>admin</code> because this same token is what the tab above uses to list
          and revoke keys — a narrower one would enforce auth and then lock this page
          out. To hand another service something weaker, mint it in{' '}
          <strong>API keys</strong> above and copy it from there. Generating does not
          revoke the old token; revoke it in the table above if you are replacing it.
        </div>
      </div>

      <WarmPanel />

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

function ApiKeys({ nonce, onChange }: { nonce: number; onChange: () => void }) {
  // `nonce` so that minting a token from the card below shows up here too,
  // rather than leaving a table that quietly disagrees with reality.
  const keys = useAsync(() => api.listKeys(), [nonce])
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
          <div className="row tight" style={{ margin: '8px 0', alignItems: 'center' }}>
            <div style={{ flex: '1 1 260px', minWidth: 0 }}>
              <code>{minted.token}</code>
            </div>
            <div style={{ flex: '0 0 auto' }}>
              <CopyButton value={minted.token} className="btn ghost sm" />
            </div>
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

/**
 * Pre-load a checkpoint so the first generation is not also a model load.
 *
 * Non-blocking on the server: warming an uncached checkpoint can take far
 * longer than an HTTP request should, so this queues and returns.
 */
function WarmPanel() {
  const models = useAsync(() => api.coreModels(), [])
  const [model, setModel] = useState('')
  const [status, setStatus] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const available = models.data?.models.filter((m) => m.available) ?? []
  const chosen = model || available[0]?.value || ''

  return (
    <div className="card">
      <h2>Warm a model</h2>
      <p className="hint">
        Loads a checkpoint into VRAM ahead of time. The first generation after a
        restart otherwise pays for the load as well, which reads as a hang.
      </p>
      {error && <div className="note err">{error}</div>}
      {status && <div className="note ok">{status}</div>}
      <div className="row tight">
        <div style={{ flex: '3 1 280px' }}>
          <label htmlFor="warm-model">Model</label>
          <select id="warm-model" value={chosen} onChange={(e) => setModel(e.target.value)}>
            {available.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </select>
        </div>
        <div style={{ flex: '0 0 auto' }}>
          <button
            className="btn ghost"
            disabled={busy || !chosen}
            onClick={() => {
              setBusy(true)
              setError(null)
              setStatus(null)
              api
                .warm(chosen)
                .then(() => setStatus('Queued — the worker loads it in the background.'))
                .catch((e) => setError(e instanceof Error ? e.message : String(e)))
                .finally(() => setBusy(false))
            }}
          >
            {busy ? 'Queuing…' : 'Warm'}
          </button>
        </div>
      </div>
    </div>
  )
}
