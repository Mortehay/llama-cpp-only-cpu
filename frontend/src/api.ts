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

/**
 * Fetch a PROTECTED binary resource and hand back an object URL.
 *
 * WHY THIS EXISTS - THE ONE THING BEARER AUTH CANNOT DO
 *
 * A browser sends no `Authorization` header when it loads `<img src>`, follows
 * `<a href>`, or navigates to a page. So the moment the API starts enforcing
 * keys, every plain `<img src="/api/jobs/{id}/sheet">` 401s - and it looks like
 * a broken image, not an auth failure, which is a miserable thing to debug.
 *
 * This was latent in the repo for a while: `/api/jobs/{id}/sheet` has always
 * called `auth.require`, and it only worked because no key existed yet, so the
 * API was in open mode and authorised everyone. Minting the first key would
 * have broken the sheet and tile previews with no warning.
 *
 * Fetching here instead puts the request back on the XHR path, where the token
 * IS sent. The caller owns the returned URL and must `URL.revokeObjectURL` it -
 * see `useAuthedObjectUrl` in hooks.ts, which does that on unmount.
 *
 * Note this is NOT needed for `/images/*`: that is a StaticFiles mount and is
 * deliberately left unauthenticated. See `.ai/specs/api-auth-lockdown/plan.md`
 * for why that is safe (unguessable names, no directory listing, and every
 * endpoint that enumerates names requires a key).
 */
export async function fetchObjectUrl(path: string): Promise<string> {
  const headers = new Headers()
  const token = getToken()
  if (token) headers.set('Authorization', `Bearer ${token}`)

  const res = await fetch(path, { headers })
  if (!res.ok) {
    // A binary route's error body is JSON from FastAPI, but do not assume it:
    // a proxy or a crash can put HTML here, and `.json()` would then throw and
    // mask the real status.
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body && typeof body === 'object' && 'detail' in body) {
        detail = String((body as { detail: unknown }).detail)
      }
    } catch {
      /* keep the status line */
    }
    throw new ApiError(res.status, detail)
  }
  return URL.createObjectURL(await res.blob())
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

export type ReferenceKind = 'core' | 'sprite' | 'tile' | 'map'

/**
 * One ground type in a map.
 *
 * `color` is not a display hint - it is the colour the biome painting is FORCED
 * to, so quantising is a lookup rather than a nearest-match. `tile` reuses an
 * existing tile and costs no GPU; without it, `prompt` generates one.
 */
export interface Terrain {
  name: string
  color: string
  prompt?: string | null
  tile?: string | null
}

/**
 * How a map's missing props are getting on.
 *
 * `final` is the field to branch on, not `state`. A map can sit at
 * `complete: false` because the resolver is still working or because it died,
 * and those look identical from the outside - so a UI without this either
 * spins forever or gives up on a map that was about to finish.
 *
 * `null` on a map means nothing was ever missing.
 */
export interface PropsStatus {
  state: 'working' | 'partial' | 'failed' | 'lost' | string
  final: boolean
  progress_pct: number | null
  detail: string | null
  job_id: string
}

export interface MapSummary {
  job_id: string
  name: string | null
  status: string
  size: number | null
  terrains: string[]
  picture_url: string | null
  map_url: string | null
  props: PropsStatus | null
  created_at: string | null
}

/**
 * One world's verdict.
 *
 * `per_screen` is the only number here a human can feel: something2's canvas is
 * a fixed 1280x720 and a tile projects to a 128x64 diamond, so a screen is ~225
 * tiles. Under 3 and the world reads as empty space.
 */
export interface WorldRow {
  key: string
  name: string
  biomes: string[]
  density: string
  per_screen: number
  creatures: number
  area: number
  biome_multiplier: number
  flora: string[]
  creature_types: string[]
  variety: number
  leaders: number
  /** surface | deep — a region descends, so this should ramp outward. */
  depth: string | null
  /** Projected JSON per second down ONE socket for a parked player. */
  socket_kib_s: number
  verdict: 'ok' | 'EMPTY' | 'CROWDED'
}

export interface WorldReport {
  worlds: WorldRow[]
  problems: string[]
  ok: boolean
  totals: {
    worlds: number
    creatures: number
    mean_per_screen: number
    min_per_screen: number
    max_per_screen: number
    empty_worlds: number
  }
  notes: string[]
}

export interface WorldListItem {
  name: string
  region?: string
  worlds?: number
  creatures?: number
  mean_per_screen?: number
  empty_worlds?: number
  ok?: boolean
  bytes?: number
  created_at?: number
  spec_url?: string
  preview_url?: string
  error?: string
}

export interface NewMap {
  job_id: string
  name: string
  grid: { w: number; h: number }
  tile: { w: number; h: number; ratio: number }
  picture: { w: number; h: number }
  tiles: { reused: number; to_generate: number }
  projection: string
  poll: string
  map: string
}

export interface Reference {
  id: string
  kind: ReferenceKind
  label: string
  url: string | null
  /** Measurement-grade: palette-locked, hard alpha, isolated. Gates profiles. */
  usable: boolean | null
  why: string
  /** Good enough to train a style on. Far more permissive. Gates training. */
  trainable: boolean | null
  trainable_why: string | null
  /** Small PNG for the grid. Falls back to `url` when not yet generated. */
  thumb_url: string | null
  metrics: Record<string, unknown>
  created_at: string | null
}

export interface ReferenceList {
  items: Reference[]
  total: number
  usable: number
  trainable: number
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
  /**
   * The server calls this `job_id`, not `id` — see `_row_to_job` in jobs.py.
   * It is the same name `POST /api/jobs` returns and the name something2 polls
   * on, so the contract is right and the client was wrong. Declaring `id` here
   * typed a field that never arrives, so `job.id.slice(0, 8)` threw on the
   * first poll of every job and took the whole tab down with it.
   */
  job_id: string
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
  /** True for adapters trained on this machine, which sort to the top. */
  trained?: boolean
  trigger?: string | null
}

export interface TrainingRun {
  id: string
  base_model: string
  config: Record<string, unknown>
  dataset_size: number | null
  status: 'queued' | 'running' | 'done' | 'failed'
  steps_done: number
  steps_total: number | null
  loss: number | null
  output_path: string | null
  error: string | null
  created_at: string | null
  started_at: string | null
  finished_at: string | null
}

export interface TrainingReadiness {
  ready: boolean
  usable_references: number
  min_images: number
  busy: boolean
  why: string
  kinds: string[]
  per_kind: Record<string, number>
  /** References no successful run has consumed - what incremental would use. */
  new_references: number
  new_per_kind: Record<string, number>
  min_new_images: number
}

export interface ActionCatalog {
  actions: { id: string; label: string; max_frames: number }[]
  directions: { id: string; family: string }[]
}

export interface RecentTask {
  id: number
  timestamp: string
  prompt: string
  file_path: string | null
  duration_ms: number | null
  error: string
  task_id: string | null
  progress_pct: number
  progress_msg: string | null
  image_type: string | null
  llm_name?: string
}

export interface EditCapabilities {
  model: string
  note: string
  enabled?: boolean
  reason?: string | null
}

export interface Core {
  id: number
  file_path: string
  prompt: string
  created_at?: string | null
}

export interface CoreList {
  items: Core[]
  total: number
  limit: number
  offset: number
  /** Tags mined from every core prompt, most used first, for the search box. */
  suggestions: string[]
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
  /**
   * Hide an asset, or with `purge` delete its file from disk too.
   *
   * The job row survives either way - something2 may still be polling that id.
   * `purge` is what reclaims the disk, and it is the half that cannot be undone.
   */
  deleteAsset: (source: string, id: string, purge = false) =>
    request<{ deleted: { source: string; id: string }; purged: string[] }>(
      `/api/assets/${source}/${id}${purge ? '?purge=true' : ''}`,
      { method: 'DELETE' },
    ),

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
    request<Reference>(`/api/references/${id}/remeasure`, { method: "POST" }),
  remeasureAll: (kind?: ReferenceKind) =>
    request<{
      remeasured: number
      by_kind: { kind: string; total: number; usable: number; trainable: number }[]
    }>(`/api/references/remeasure-all${kind ? `?kind=${kind}` : ""}`, { method: "POST" }),

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

  coreModels: () => request<{ models: CoreModel[] }>("/api/core-models"),

  training: () =>
    request<{ items: TrainingRun[]; total: number; min_images: number }>("/api/training"),
  trainingReadiness: (kinds: string[] = ["sprite", "core"]) =>
    request<TrainingReadiness>(`/api/training/readiness?kinds=${kinds.join(",")}`),
  startTraining: (body: Record<string, unknown>) =>
    request<{
      run_id: string
      status: string
      dataset_size: number
      trigger: string
      kinds: string[]
      mixed_kinds: boolean
      note: string | null
      mode: "incremental" | "full"
      resuming: boolean
    }>(
      "/api/training",
      { method: "POST", body: JSON.stringify(body) },
    ),
  deleteRun: (id: string) =>
    request<unknown>(`/api/training/${id}`, { method: "DELETE" }),

  recentTasks: () => request<RecentTask[]>("/api/tasks/recent"),
  retryTask: (id: number) =>
    request<unknown>(`/api/task/${id}/retry`, { method: "POST" }),
  deleteTask: (id: number) =>
    request<unknown>(`/api/task/${id}`, { method: "DELETE" }),
  warm: (model: string) => {
    const fd = new FormData()
    fd.set("model", model)
    return request<unknown>("/api/warm", { method: "POST", body: fd })
  },
  editCapabilities: () => request<EditCapabilities>("/api/edit-capabilities"),
  edit: (source: string, instruction: string) => {
    const fd = new FormData()
    fd.set("source", source)
    fd.set("instruction", instruction)
    return request<{ task_id?: string; status?: string }>("/api/edit", {
      method: "POST",
      body: fd,
    })
  },
  crop: (source_id: number, x: number, y: number, w: number, h: number) =>
    request<{ status?: string; url?: string }>("/api/crop", {
      method: "POST",
      body: JSON.stringify({ source_id, x, y, w, h }),
    }),

  createTile: (body: Record<string, unknown>) =>
    request<{
      job_id: string
      tile: { w: number; h: number; ratio: number }
      projection: string
    }>("/api/tiles", { method: "POST", body: JSON.stringify(body) }),
  /**
   * Entities, newest first. Paged since the server stopped hard-capping at 24
   * rows; pass `q` to filter by prompt substring.
   */
  cores: (params: { q?: string; limit?: number; offset?: number } = {}) => {
    const qs = new URLSearchParams()
    if (params.q) qs.set('q', params.q)
    if (params.limit != null) qs.set('limit', String(params.limit))
    if (params.offset) qs.set('offset', String(params.offset))
    const s = qs.toString()
    return request<CoreList>(`/api/cores${s ? `?${s}` : ''}`)
  },
  createWorld: (body: {
    name: string
    worlds: number
    target_per_screen: number
    size: number
    theme?: string | null
    author: 'rules' | 'llm'
    overwrite?: boolean
  }) =>
    request<{
      name: string
      author: string
      report: WorldReport
      spec_url: string
      preview_url: string
      seed_with: string
      spec: unknown
    }>('/api/worlds', { method: 'POST', body: JSON.stringify(body) }),
  /**
   * Change one thing about a region and rebuild it.
   *
   * Fields not named are carried over, including the biome plan - so raising
   * the creature target does not re-roll which biomes the region is made of.
   * `reauthor` is the opt-in that does re-ask the LLM.
   */
  editWorld: (
    name: string,
    edit: {
      worlds?: number
      target_per_screen?: number
      size?: number
      theme?: string | null
      reauthor?: boolean
    },
  ) =>
    request<{
      name: string
      author: string
      params: Record<string, unknown>
      report: WorldReport
      changed: string[]
      spec: unknown
    }>(`/api/worlds/${name}`, { method: 'PATCH', body: JSON.stringify(edit) }),
  worlds: () => request<{ items: WorldListItem[]; total: number }>('/api/worlds'),
  worldReport: (name: string) => request<WorldReport>(`/api/worlds/${name}/report`),
  deleteWorld: (name: string) =>
    request<{ deleted: string[] }>(`/api/worlds/${name}`, { method: 'DELETE' }),
  createMap: (body: {
    name: string
    terrains: Terrain[]
    size: number
    prompt?: string | null
    painting_from?: string | null
    tile_w?: number
    style_profile?: string | null
    colors?: number
    seed?: number
    llm_name?: string | null
  }) => request<NewMap>('/api/maps', { method: 'POST', body: JSON.stringify(body) }),
  maps: () => request<{ items: MapSummary[]; total: number }>('/api/maps'),
  mapData: (jobId: string) =>
    request<Record<string, unknown>>(`/api/maps/${jobId}`),
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
