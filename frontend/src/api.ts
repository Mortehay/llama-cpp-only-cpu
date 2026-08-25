/**
 * The single place that talks to the backend.
 *
 * Every call goes through `request`, so the bearer token, error shape and JSON
 * handling are decided once. The old UI did `fetch` inline in ~40 places and
 * each site invented its own error handling; a 500 usually surfaced as a
 * silently empty panel.
 */

const TOKEN_KEY = 'sprite.apiToken'

export function getToken(): string {
  try {
    return localStorage.getItem(TOKEN_KEY) ?? ''
  } catch {
    return ''
  }
}

export function setToken(token: string) {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token)
    else localStorage.removeItem(TOKEN_KEY)
  } catch {
    /* private browsing - the app still works, it just cannot remember */
  }
}

/** An error that carries the server's own explanation, not just a status. */
export class ApiError extends Error {
  constructor(readonly status: number, message: string) {
    super(message)
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers)
  const token = getToken()
  if (token) headers.set('Authorization', `Bearer ${token}`)
  // FormData sets its own multipart boundary; forcing JSON here breaks uploads.
  if (init.body && !(init.body instanceof FormData)) {
    headers.set('Content-Type', 'application/json')
  }

  const res = await fetch(path, { ...init, headers })
  const text = await res.text()
  let body: unknown = null
  if (text) {
    try {
      body = JSON.parse(text)
    } catch {
      body = text
    }
  }

  if (!res.ok) {
    // FastAPI puts the useful part in `detail`, and it is usually a sentence
    // written for exactly this moment. Prefer it over "Request failed".
    const detail =
      (body && typeof body === 'object' && 'detail' in body
        ? String((body as { detail: unknown }).detail)
        : typeof body === 'string' && body
          ? body
          : `${res.status} ${res.statusText}`)
    throw new ApiError(res.status, detail)
  }
  return body as T
}

// --- types ----------------------------------------------------------------

export interface AuthMode {
  enforced: boolean
  active_keys: number
  legacy_token: boolean
  message: string
}

export interface ApiKey {
  id: string
  name: string
  key_prefix: string
  scopes: string[]
  created_at: string | null
  last_used_at: string | null
  revoked: boolean
}

export interface NewApiKey extends Omit<ApiKey, 'revoked' | 'last_used_at'> {
  token: string
  bootstrap: boolean
}

export interface Asset {
  id: string
  source: 'image' | 'job'
  kind: string
  title: string
  url: string | null
  created_at: string | null
  job_id: string | null
  model: string | null
  atlas_url: string | null
}

export type ReferenceKind = 'core' | 'sprite' | 'tile'

export interface Reference {
  id: string
  kind: ReferenceKind
  label: string
  url: string | null
  usable: boolean | null
  why: string
  metrics: Record<string, unknown>
  created_at: string | null
}

export interface ReferenceList {
  items: Reference[]
  total: number
  usable: number
  enough_to_train: boolean
  enough_to_measure: boolean
}

export interface StyleProfile {
  id: string
  name: string
  palette: string[] | null
  cell_w: number | null
  cell_h: number | null
  colors: number | null
  outline: { width: number; color: string } | null
  projection_ratio: number | null
  elevation: string | null
  lora_path: string | null
  trigger_token: string | null
  gaps?: string[]
}

export interface Job {
  id: string
  status: 'queued' | 'running' | 'done' | 'failed' | 'cancelled'
  kind?: string
  progress_pct: number
  progress_msg: string | null
  stage: string | null
  error: string | null
  sheet_url?: string
  created_at: string | null
  updated_at: string | null
}

export interface CoreModel {
  value: string
  label: string
  default: boolean
  available: boolean
  reason: string | null
  missing: string[]
}

export interface ActionCatalog {
  actions: { id: string; label: string; max_frames: number }[]
  directions: { id: string; family: string }[]
}

export interface Core {
  id: number
  file_path: string
  prompt: string
}

// --- calls ----------------------------------------------------------------

export const api = {
  authMode: () => request<AuthMode>('/api/auth/mode'),
  listKeys: () => request<{ keys: ApiKey[]; mode: AuthMode }>('/api/auth/keys'),
  createKey: (name: string, scopes: string[]) =>
    request<NewApiKey>('/api/auth/keys', {
      method: 'POST',
      body: JSON.stringify({ name, scopes }),
    }),
  revokeKey: (id: string) =>
    request<unknown>(`/api/auth/keys/${id}`, { method: 'DELETE' }),

  assets: (params: { kind?: string; source?: string; q?: string; limit?: number; offset?: number }) => {
    const qs = new URLSearchParams()
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== '') qs.set(k, String(v))
    })
    return request<{ total: number; limit: number; offset: number; items: Asset[] }>(
      `/api/assets?${qs}`,
    )
  },
  assetKinds: () =>
    request<{ groups: { source: string; kind: string; n: number }[] }>('/api/assets/kinds'),
  deleteAsset: (source: string, id: string) =>
    request<unknown>(`/api/assets/${source}/${id}`, { method: 'DELETE' }),

  references: (kind?: ReferenceKind) =>
    request<ReferenceList>(`/api/references${kind ? `?kind=${kind}` : ''}`),
  uploadReference: (kind: ReferenceKind, file: File, label?: string) => {
    const fd = new FormData()
    fd.set('kind', kind)
    fd.set('file', file)
    if (label) fd.set('label', label)
    return request<Reference>('/api/references', { method: 'POST', body: fd })
  },
  deleteReference: (id: string) =>
    request<unknown>(`/api/references/${id}`, { method: 'DELETE' }),
  remeasure: (id: string) =>
    request<Reference>(`/api/references/${id}/remeasure`, { method: 'POST' }),

  profiles: () => request<{ items: StyleProfile[]; total: number }>('/api/style-profiles'),
  deriveProfile: (name: string, reference_ids: string[] = []) =>
    request<StyleProfile>('/api/style-profiles/derive', {
      method: 'POST',
      body: JSON.stringify({ name, reference_ids }),
    }),
  deleteProfile: (id: string) =>
    request<unknown>(`/api/style-profiles/${id}`, { method: 'DELETE' }),

  jobs: (limit = 25) => request<{ jobs: Job[] }>(`/api/jobs?limit=${limit}`),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  createJob: (spec: Record<string, unknown>) =>
    request<{ job_id: string; status: string }>('/api/jobs', {
      method: 'POST',
      body: JSON.stringify(spec),
    }),
  cancelJob: (id: string) => request<unknown>(`/api/jobs/${id}`, { method: 'DELETE' }),

  coreModels: () => request<{ models: CoreModel[] }>('/api/core-models'),
  cores: () => request<Core[]>('/api/cores'),
  actionCatalog: () => request<ActionCatalog>('/api/action-catalog'),
  computeInfo: () => request<Record<string, unknown>>('/api/compute-info'),

  generateCore: (prompt: string, llm_name: string) => {
    const fd = new FormData()
    fd.set('prompt', prompt)
    fd.set('llm_name', llm_name)
    return request<{ task_id?: string; status?: string }>('/api/generate_core', {
      method: 'POST',
      body: fd,
    })
  },
}

/** `/app/images/x.png` -> `/images/x.png`, matching the static mount. */
export function imageUrl(filePath: string | null): string {
  if (!filePath) return ''
  const name = filePath.split('/').pop()
  return name ? `/images/${name}` : ''
}
