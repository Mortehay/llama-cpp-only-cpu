import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError } from './api'

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
