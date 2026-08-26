import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError, fetchObjectUrl } from './api'

export interface AsyncState<T> {
  data: T | null
  error: string | null
  loading: boolean
  reload: () => void
}

/**
 * Load something, keep the error, expose a reload.
 *
 * The error is a *string* rather than swallowed: the old UI's most common
 * failure was a panel that simply stayed empty, because a rejected fetch had
 * nowhere to go. Anything using this renders the reason instead.
 */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[] = []): AsyncState<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [tick, setTick] = useState(0)

  // A slow response that lands after the component unmounts (or after the
  // inputs change) must not overwrite newer state.
  const alive = useRef(true)
  useEffect(() => {
    alive.current = true
    return () => {
      alive.current = false
    }
  }, [])

  useEffect(() => {
    setLoading(true)
    fn()
      .then((d) => {
        if (!alive.current) return
        setData(d)
        setError(null)
      })
      .catch((e: unknown) => {
        if (!alive.current) return
        setError(e instanceof ApiError || e instanceof Error ? e.message : String(e))
      })
      .finally(() => {
        if (alive.current) setLoading(false)
      })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick])

  const reload = useCallback(() => setTick((t) => t + 1), [])
  return { data, error, loading, reload }
}

/**
 * An object URL for a protected binary route, fetched with the bearer token.
 *
 * Use this anywhere an `<img src>` or an `<a href>` would otherwise point at an
 * authenticated endpoint: the browser sends no Authorization header on either,
 * so those 401 as soon as a key exists. See `fetchObjectUrl` in api.ts.
 *
 * Pass `null` to skip fetching (e.g. while a job is still running).
 *
 * The URL is revoked when it changes or the component unmounts. Without that,
 * every poll of a finished job would leak a blob for the lifetime of the tab.
 */
export function useAuthedObjectUrl(path: string | null): {
  url: string | null
  error: string | null
} {
  const [url, setUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!path) {
      setUrl(null)
      setError(null)
      return
    }
    let revoked = false
    let created: string | null = null

    fetchObjectUrl(path)
      .then((u) => {
        // Unmounted (or `path` changed) while the fetch was in flight: revoke
        // immediately rather than setting state on a dead component and leaking
        // the blob.
        if (revoked) {
          URL.revokeObjectURL(u)
          return
        }
        created = u
        setUrl(u)
        setError(null)
      })
      .catch((e: unknown) => {
        if (revoked) return
        setError(e instanceof ApiError || e instanceof Error ? e.message : String(e))
        setUrl(null)
      })

    return () => {
      revoked = true
      if (created) URL.revokeObjectURL(created)
    }
  }, [path])

  return { url, error }
}

/** Re-run `fn` on an interval, but only while `active`. */
export function usePoll(fn: () => void, ms: number, active: boolean) {
  const saved = useRef(fn)
  saved.current = fn
  useEffect(() => {
    if (!active) return
    const id = setInterval(() => saved.current(), ms)
    return () => clearInterval(id)
  }, [ms, active])
}

/**
 * `value`, but only after it has stopped changing for `ms`.
 *
 * Search boxes here drive real queries — a keystroke-per-request list also
 * races: `useAsync` keeps whichever response lands last, which is not
 * necessarily the one for what is currently typed.
 */
export function useDebounced<T>(value: T, ms = 250): T {
  const [settled, setSettled] = useState(value)
  useEffect(() => {
    const id = setTimeout(() => setSettled(value), ms)
    return () => clearTimeout(id)
  }, [value, ms])
  return settled
}
