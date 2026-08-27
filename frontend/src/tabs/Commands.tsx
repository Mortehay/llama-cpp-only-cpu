import { useState } from 'react'
import { api, type Command, type CommandStatus } from '../api'
import { useAsync, usePoll } from '../hooks'

/**
 * Operator commands.
 *
 * These existed only as Makefile targets, so running the audit that explains
 * why a reference was excluded needed a terminal, a checkout and Docker. The
 * REASONS were already visible - `ReferenceTab` renders `trainable_why` - but
 * producing them was not.
 *
 * The list comes from the server. Nothing here builds a command line: a button
 * sends a KEY and the server owns the argv, which is what keeps an endpoint on
 * a port that is unauthenticated by default from being a remote shell.
 */

const WRITE_LABEL: Record<string, string> = {
  nothing: 'reads only',
  files: 'writes new files',
  database: 'writes to the database',
}

const TERMINAL = ['SUCCESS', 'FAILURE', 'REVOKED']

export default function Commands() {
  const listing = useAsync(() => api.commands(), [])
  const [taskId, setTaskId] = useState<string | null>(null)
  const [status, setStatus] = useState<CommandStatus | null>(null)
  const [running, setRunning] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  // Which database-writing command is waiting for a second click. Held by name
  // rather than as a boolean so arming one cannot arm another.
  const [armed, setArmed] = useState<string | null>(null)

  const polling = !!taskId && (!status || !TERMINAL.includes(status.status))

  usePoll(
    () => {
      if (!taskId) return
      api
        .commandStatus(taskId)
        .then((s) => {
          setStatus(s)
          if (TERMINAL.includes(s.status)) {
            setRunning(null)
            // A command that wrote to the database has invalidated the row
            // count shown beside the button, so re-read it rather than leaving
            // a stale number on screen next to a confirmation.
            listing.reload()
          }
        })
        .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
    },
    1500,
    polling,
  )

  async function start(cmd: Command) {
    if (cmd.writes === 'database' && armed !== cmd.name) {
      setArmed(cmd.name)
      return
    }
    setArmed(null)
    setError(null)
    setStatus(null)
    setRunning(cmd.name)
    try {
      const { task_id } = await api.runCommand(cmd.name, cmd.writes === 'database')
      setTaskId(task_id)
    } catch (e: unknown) {
      setRunning(null)
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const stakes = listing.data?.stakes
  const busy = !!running

  return (
    <>
      <div className="card">
        <h2>Commands</h2>
        <p className="muted tight">
          The curation and recovery tools, without a terminal. Each one is a
          fixed command on the server — this page chooses which, never what.
        </p>

        {listing.error && <p className="err">Could not load commands: {listing.error}</p>}
        {error && <p className="err">{error}</p>}

        {listing.data?.shares_worker && (
          <p className="note">
            These run on the same single worker as image generation. A
            five-minute audit delays the next sheet by five minutes.
          </p>
        )}
      </div>

      {listing.data?.groups.map((group) => {
        const items = listing.data!.commands.filter((c) => c.group === group)
        if (!items.length) return null
        return (
          <div className="card" key={group}>
            <h3>{group}</h3>
            {items.map((cmd) => {
              const isArmed = armed === cmd.name
              const isRunning = running === cmd.name
              return (
                <div key={cmd.name} className="rows">
                  <div className="row">
                    <div>
                      <strong>{cmd.label}</strong>{' '}
                      <span className={`tag ${cmd.writes === 'database' ? 'no' : 'ok'}`}>
                        {WRITE_LABEL[cmd.writes] ?? cmd.writes}
                      </span>{' '}
                      <span className="muted sm">~{cmd.minutes} min</span>
                    </div>
                    <div className="spacer" />
                    <button
                      className={isArmed ? 'btn danger' : 'btn'}
                      disabled={busy || !cmd.available}
                      title={cmd.unavailable_why ?? undefined}
                      onClick={() => void start(cmd)}
                    >
                      {isRunning ? 'Running…' : isArmed ? 'Yes, write it' : 'Run'}
                    </button>
                    {isArmed && (
                      <button className="btn ghost" onClick={() => setArmed(null)}>
                        Cancel
                      </button>
                    )}
                  </div>

                  <p className="muted tight">{cmd.summary}</p>
                  <p className="hint tight">{cmd.detail}</p>

                  {/* Shown, not just disabled. A greyed button with no reason
                      reads as a bug in the page rather than a missing package,
                      and the reason names what to install. */}
                  {!cmd.available && cmd.unavailable_why && (
                    <p className="warn">Cannot run here: {cmd.unavailable_why}.</p>
                  )}

                  {/* The row count lives with the confirmation rather than in a
                      dialog title, because the number IS the decision — "are
                      you sure?" without it asks nothing answerable. */}
                  {isArmed && (
                    <p className="warn">
                      This sets <code>trainable = false</code> on every rejected
                      reference and records why.{' '}
                      {stakes?.trainable != null ? (
                        <>
                          <strong>{stakes.trainable}</strong> live references are
                          currently trainable
                          {stakes.with_reason != null && (
                            <>, {stakes.with_reason} already carry a reason</>
                          )}
                          .
                        </>
                      ) : (
                        <>
                          The row count could not be read
                          {stakes?.why ? ` (${stakes.why})` : ''}, so this would run
                          without showing you the stakes.
                        </>
                      )}{' '}
                      Click again to run it.
                    </p>
                  )}
                </div>
              )
            })}
          </div>
        )
      })}

      {status && <Output status={status} />}
    </>
  )
}

function Output({ status }: { status: CommandStatus }) {
  const done = TERMINAL.includes(status.status)
  // A non-zero exit is not a crash, and the difference matters: a failing test
  // suite has done its job and its output is the point. Only `crashed` means
  // the task itself died, in which case there may be nothing useful below.
  const failed = status.crashed || (status.exit_code != null && status.exit_code !== 0)

  return (
    <div className="card">
      <h3>
        {status.name ?? 'Command'}{' '}
        <span className={`tag ${!done ? 'neutral' : failed ? 'no' : 'ok'}`}>
          {!done ? status.status : failed ? 'finished with failures' : 'finished'}
        </span>
      </h3>

      {!done && status.message && <p className="muted tight">{status.message}</p>}
      {status.crashed && <p className="err">The command did not run: {status.error}</p>}
      {!status.crashed && status.exit_code != null && status.exit_code !== 0 && (
        <p className="warn">
          Exit code {status.exit_code}. For a test suite that is a result rather
          than an error — the failing checks are below.
        </p>
      )}

      {status.lines.length > 0 && (
        <pre
          style={{
            background: 'rgba(0,0,0,.25)',
            padding: 10,
            borderRadius: 6,
            fontSize: 11,
            lineHeight: 1.5,
            overflowX: 'auto',
            whiteSpace: 'pre-wrap',
          }}
        >
          {status.lines.join('\n')}
        </pre>
      )}
      {done && !status.lines.length && !status.crashed && (
        <p className="muted">The command produced no output.</p>
      )}
    </div>
  )
}
