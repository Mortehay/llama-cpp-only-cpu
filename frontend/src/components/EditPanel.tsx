import { useState } from 'react'
import { api, imageUrl } from '../api'
import { useAsync } from '../hooks'

/**
 * Instruction-driven edit of an existing concept.
 *
 * Not img2img: you say WHAT to change ("remove the shield"), not how far to
 * deviate. The wording matters enough that the capability note from the server
 * is shown rather than paraphrased.
 */
export default function EditPanel({ onDone }: { onDone?: () => void }) {
  const caps = useAsync(() => api.editCapabilities(), [])
  // 200 is the server's page cap: a <select> of prompt text is cheap, and a
  // truncated list here means an older concept simply cannot be edited.
  const cores = useAsync(() => api.cores({ limit: 200 }), [])

  const [source, setSource] = useState('')
  const [instruction, setInstruction] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [status, setStatus] = useState<string | null>(null)

  async function submit() {
    setBusy(true)
    setError(null)
    setStatus(null)
    try {
      await api.edit(source, instruction.trim())
      setStatus('Queued. It appears in Recent tasks below when it lands.')
      onDone?.()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const disabled = caps.data?.enabled === false

  return (
    <div className="card">
      <h2>Edit a concept</h2>
      <p className="hint">
        {caps.data?.note ??
          'Instruction-driven editing: say what to change, not how far to deviate.'}
      </p>

      {disabled && (
        <div className="note warn">
          Editing unavailable: {caps.data?.reason ?? 'the edit model is not loaded'}
        </div>
      )}
      {error && <div className="note err">{error}</div>}
      {status && <div className="note ok">{status}</div>}

      <div className="row">
        <div style={{ flex: '1 1 240px' }}>
          <label htmlFor="edit-src">Source image</label>
          <select id="edit-src" value={source} onChange={(e) => setSource(e.target.value)}>
            <option value="">Choose…</option>
            {cores.data?.items.map((c) => {
              const name = c.file_path.split('/').pop() ?? ''
              return (
                <option key={c.id} value={name}>
                  {c.prompt?.slice(0, 60) || name}
                </option>
              )
            })}
          </select>
        </div>
        <div style={{ flex: '2 1 300px' }}>
          <label htmlFor="edit-inst">Instruction</label>
          <input
            id="edit-inst"
            type="text"
            value={instruction}
            onChange={(e) => setInstruction(e.target.value)}
            placeholder="remove the shield, give them a red cloak"
          />
        </div>
      </div>

      {source && (
        <div className="thumb" style={{ maxWidth: 180, marginTop: 12 }}>
          <div className="pic">
            <img src={imageUrl(`/app/images/${source}`)} alt="source" />
          </div>
        </div>
      )}

      <div className="spacer" />
      <button
        className="btn"
        disabled={busy || disabled || !source || !instruction.trim()}
        onClick={() => void submit()}
      >
        {busy ? 'Queuing…' : 'Apply edit'}
      </button>
    </div>
  )
}
