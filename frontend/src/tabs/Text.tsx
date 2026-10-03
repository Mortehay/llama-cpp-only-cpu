import { useState } from 'react'
import { ApiError, textApi } from '../api'
import type { TextAnswer } from '../api'
import { useAsync } from '../hooks'

/**
 * Ask the brain anything, by hand.
 *
 * Drives the same `POST /api/text` something2's Text provider calls, so what
 * works here works there (decisions/0013). Two things shape the page:
 *
 * - The brain goes through the model gateway. While another model holds the
 *   card the answer is an immediate 503 (or 409 mid-switch), shown with its
 *   retry hint - it is never queued, so nothing is retried automatically.
 * - A schema makes the answer grammar-constrained JSON. An impossible schema is
 *   a 422 before any GPU time is spent.
 *
 * There is no history here: every call is a row on the Activity tab.
 */
export default function Text() {
  const models = useAsync(() => textApi.models(), [])
  const [model, setModel] = useState('')
  const [prompt, setPrompt] = useState('')
  const [system, setSystem] = useState('')
  const [schema, setSchema] = useState('')
  const [temperature, setTemperature] = useState('0.7')
  const [maxTokens, setMaxTokens] = useState('512')
  const [busy, setBusy] = useState(false)
  const [answer, setAnswer] = useState<TextAnswer | null>(null)
  const [error, setError] = useState<string | null>(null)

  const brains = models.data?.data ?? []

  async function ask() {
    setBusy(true)
    setError(null)
    setAnswer(null)
    let parsed: unknown = undefined
    if (schema.trim()) {
      try {
        parsed = JSON.parse(schema)
      } catch (e) {
        setError(`Schema is not valid JSON: ${e instanceof Error ? e.message : e}`)
        setBusy(false)
        return
      }
    }
    try {
      setAnswer(
        await textApi.complete({
          prompt,
          system: system.trim() || undefined,
          schema: parsed,
          temperature: Number(temperature),
          max_tokens: Number(maxTokens),
          model: model || undefined,
        }),
      )
    } catch (e) {
      const status = e instanceof ApiError ? `${e.status}: ` : ''
      setError(status + (e instanceof Error ? e.message : String(e)))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="card">
      <h2>Text — ask the brain</h2>
      <p className="hint">
        The same endpoint something2&apos;s Text provider uses. The brain shares the card through
        the model gateway, so while an image or audio model holds it the answer is
        &ldquo;busy&rdquo; at once. A cold brain loads first (about 40 s for the 8B, longer for
        the 35B). Every call is listed on the Activity tab.
      </p>
      {models.error && <div className="note err">{models.error}</div>}

      <div className="row tight">
        <div style={{ flex: '2 1 260px' }}>
          <label htmlFor="t-model">Brain</label>
          <select id="t-model" value={model} onChange={(e) => setModel(e.target.value)}>
            <option value="">default</option>
            {brains.map((b) => (
              <option key={b.id} value={b.id} disabled={!b.available}>
                {b.label}
                {b.default ? ' (default)' : ''}
                {b.available ? '' : ' (not on disk)'}
              </option>
            ))}
          </select>
        </div>
        <div style={{ flex: '1 1 100px' }}>
          <label htmlFor="t-temp">Temperature</label>
          <input id="t-temp" type="number" min={0} max={2} step={0.05}
            value={temperature} onChange={(e) => setTemperature(e.target.value)} />
        </div>
        <div style={{ flex: '1 1 100px' }}>
          <label htmlFor="t-max">Max tokens</label>
          <input id="t-max" type="number" min={1} max={4096}
            value={maxTokens} onChange={(e) => setMaxTokens(e.target.value)} />
        </div>
      </div>

      <label htmlFor="t-system" style={{ marginTop: 10 }}>System prompt (optional)</label>
      <textarea id="t-system" rows={2} value={system} onChange={(e) => setSystem(e.target.value)} />

      <label htmlFor="t-prompt">Prompt</label>
      <textarea id="t-prompt" rows={5} value={prompt} onChange={(e) => setPrompt(e.target.value)} />

      <label htmlFor="t-schema">JSON Schema (optional — the answer is then JSON that fits it)</label>
      <textarea id="t-schema" rows={4} spellCheck={false} value={schema}
        placeholder='{"type":"object","properties":{"style":{"type":"string","enum":["tavern","dungeon"]}},"required":["style"]}'
        onChange={(e) => setSchema(e.target.value)} />

      <div className="row tight" style={{ marginTop: 10 }}>
        <button className="btn" disabled={busy || !prompt.trim()} onClick={ask}>
          {busy ? 'Asking…' : 'Ask'}
        </button>
      </div>

      {error && <div className="note err">{error}</div>}
      {answer && (
        <div style={{ marginTop: 12 }}>
          <p className="hint" style={{ marginTop: 0 }}>
            <strong>{answer.model}</strong>
            {answer.usage?.completion_tokens !== undefined &&
              ` · ${answer.usage.completion_tokens} tokens`}
            {answer.timings?.generate_s !== undefined && ` · ${answer.timings.generate_s} s`}
            {answer.timings?.load_s ? ` (+${answer.timings.load_s} s load)` : ''}
            {answer.timings?.decode_tok_s ? ` · ${answer.timings.decode_tok_s} tok/s` : ''}
          </p>
          <pre style={{ whiteSpace: 'pre-wrap' }}>
            {answer.json !== undefined ? JSON.stringify(answer.json, null, 2) : answer.text}
          </pre>
          <button className="btn ghost" onClick={() => navigator.clipboard?.writeText(
            answer.json !== undefined ? JSON.stringify(answer.json, null, 2) : answer.text)}>
            Copy
          </button>
        </div>
      )}
    </div>
  )
}
