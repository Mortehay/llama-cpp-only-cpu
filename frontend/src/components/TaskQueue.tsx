import { useState } from 'react'
import { api, imageUrl } from '../api'
import { useAsync, usePoll } from '../hooks'

/**
 * Recent single-image tasks, with retry and delete.
 *
 * Separate from the job queue: `sprite_images` rows are step-1 concepts and
 * edits, which finish in seconds, while `jobs` rows are hour-long sheet builds.
 * Showing them in one list made both harder to read.
 */
export default function TaskQueue() {
  const tasks = useAsync(() => api.recentTasks(), [])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const working = tasks.data?.some((t) => !t.file_path && !t.error)
  usePoll(() => tasks.reload(), 4000, !!working)

  async function act(fn: () => Promise<unknown>) {
    setBusy(true)
    setError(null)
    try {
      await fn()
      tasks.reload()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="card">
      <h2>Recent tasks</h2>
      <p className="hint">
        Single-image work — concepts, edits and crops. Long spritesheet builds live
        on the Spritesheet tab, because an hour-long job and a nine-second one do not
        belong in the same list.
      </p>

      {error && <div className="note err">{error}</div>}
      {tasks.error && <div className="note err">{tasks.error}</div>}
      {tasks.data?.length === 0 && <div className="empty">Nothing yet.</div>}

      {tasks.data && tasks.data.length > 0 && (
        <table>
          <thead>
            <tr>
              <th style={{ width: 64 }} />
              <th>Prompt</th>
              <th>State</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {tasks.data.map((t) => {
              const done = !!t.file_path
              const failed = !!t.error
              return (
                <tr key={t.id}>
                  <td>
                    {done && (
                      <img
                        src={imageUrl(t.file_path)}
                        alt=""
                        style={{
                          width: 48,
                          height: 48,
                          objectFit: 'contain',
                          imageRendering: 'pixelated',
                        }}
                      />
                    )}
                  </td>
                  <td>
                    <div className="name" title={t.prompt}>
                      {t.prompt?.slice(0, 70) || '—'}
                    </div>
                    <div className="muted">
                      {t.llm_name ?? ''}{' '}
                      {t.duration_ms ? `· ${(t.duration_ms / 1000).toFixed(1)}s` : ''}
                    </div>
                  </td>
                  <td style={{ minWidth: 130 }}>
                    {failed ? (
                      <>
                        <span className="tag no">failed</span>
                        <div className="why">{t.error.slice(0, 120)}</div>
                      </>
                    ) : done ? (
                      <span className="tag ok">done</span>
                    ) : (
                      <>
                        <span className="tag neutral">{t.progress_msg ?? 'working'}</span>
                        <div className="bar">
                          <i style={{ width: `${t.progress_pct ?? 0}%` }} />
                        </div>
                      </>
                    )}
                  </td>
                  <td style={{ textAlign: 'right', whiteSpace: 'nowrap' }}>
                    {failed && (
                      <button
                        className="btn ghost sm"
                        disabled={busy}
                        onClick={() => void act(() => api.retryTask(t.id))}
                      >
                        Retry
                      </button>
                    )}{' '}
                    <button
                      className="btn danger sm"
                      disabled={busy}
                      onClick={() => void act(() => api.deleteTask(t.id))}
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
    </div>
  )
}
