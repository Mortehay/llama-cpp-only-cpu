import { useState } from 'react'
import { api, type StyleProfile } from '../api'
import { useAsync } from '../hooks'

/**
 * Turn measured references into the constraints the conveyor applies.
 *
 * Deriving is an explicit button rather than something an upload does, so that
 * adding one odd tile cannot silently repoint the camera for every job that
 * follows.
 */
export default function StyleProfilePanel({ onChange }: { onChange?: () => void }) {
  const profiles = useAsync(() => api.profiles(), [])
  const [name, setName] = useState('something2')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<StyleProfile | null>(null)

  async function derive() {
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const p = await api.deriveProfile(name.trim())
      setResult(p)
      profiles.reload()
      onChange?.()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  async function remove(id: string) {
    setBusy(true)
    try {
      await api.deleteProfile(id)
      profiles.reload()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="card">
      <h2>Style profile</h2>
      <p className="hint">
        Collapses every usable reference into one set of hard constraints: palette,
        cell size, outline and camera. Measurement works from about three examples —
        training a style needs roughly twenty, and is a separate step.
      </p>

      {error && <div className="note err">{error}</div>}

      <div className="row tight">
        <div style={{ flex: '2 1 220px' }}>
          <label htmlFor="profile-name">Profile name</label>
          <input
            id="profile-name"
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="something2"
          />
        </div>
        <div style={{ flex: '0 0 auto' }}>
          <button className="btn" disabled={busy || !name.trim()} onClick={() => void derive()}>
            {busy ? 'Deriving…' : 'Derive from references'}
          </button>
        </div>
      </div>

      {result && (
        <>
          <div className="spacer" />
          <div className="note ok">
            Derived <strong>{result.name}</strong> from {(result as { from_references?: number }).from_references} reference(s).
          </div>
          <Summary p={result} />
          {result.gaps?.map((g) => (
            <div className="note warn" key={g}>
              {g}
            </div>
          ))}
        </>
      )}

      <div className="spacer" />
      {profiles.data && profiles.data.items.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Camera</th>
              <th>Cell</th>
              <th>Colours</th>
              <th>LoRA</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {profiles.data.items.map((p) => (
              <tr key={p.id}>
                <td>{p.name}</td>
                <td>
                  {p.elevation ?? '—'}
                  {p.projection_ratio ? ` (${p.projection_ratio}:1)` : ''}
                </td>
                <td>{p.cell_w && p.cell_h ? `${p.cell_w}×${p.cell_h}` : '—'}</td>
                <td>{p.colors ?? '—'}</td>
                <td>{p.lora_path ? 'trained' : <span className="muted">not trained</span>}</td>
                <td style={{ textAlign: 'right' }}>
                  <button className="btn danger sm" disabled={busy} onClick={() => void remove(p.id)}>
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {profiles.data?.items.length === 0 && (
        <div className="empty">No style profiles yet.</div>
      )}
    </div>
  )
}

function Summary({ p }: { p: StyleProfile }) {
  return (
    <div className="row" style={{ marginBottom: 10 }}>
      <div>
        <label>Camera</label>
        <div>
          {p.elevation ?? '—'}
          {p.projection_ratio ? ` · ${p.projection_ratio}:1` : ''}
        </div>
      </div>
      <div>
        <label>Cell</label>
        <div>{p.cell_w && p.cell_h ? `${p.cell_w}×${p.cell_h}` : '—'}</div>
      </div>
      <div>
        <label>Outline</label>
        <div>{p.outline ? `${p.outline.width}px ${p.outline.color}` : 'none'}</div>
      </div>
      <div style={{ flex: '1 1 100%' }}>
        <label>Palette ({p.palette?.length ?? 0})</label>
        <div className="swatches">
          {(p.palette ?? []).map((c) => (
            <span key={c} className="swatch" style={{ background: c }} title={c} />
          ))}
        </div>
      </div>
    </div>
  )
}
