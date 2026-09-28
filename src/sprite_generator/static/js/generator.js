let pollInterval = null;
let selectedCoreId = null;
// True while a core generation is in flight. updateCoreModelState must not
// re-enable the Generate button underneath a running job.
let coreBusy = false;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    updateQueue();
    setInterval(updateQueue, 3000);
    // Model gateway: same cadence as the queue. See the gateway section below.
    updateGateway();
    setInterval(updateGateway, 3000);
    
    // Directions are in the markup; the action checkboxes are rendered from
    // /api/action-catalog and bind their own listeners in renderActions().
    document.querySelectorAll('input[name="direction"]')
        .forEach(cb => cb.addEventListener('change', updateGenSheetButtonState));
    // The frames box had no listener, so the estimate and the frame ceiling
    // only refreshed when an action or direction was toggled.
    const framesInput = document.getElementById('sheet-frames');
    if (framesInput) framesInput.addEventListener('input', updateSheetEstimate);
    updateGenSheetButtonState();
    // A running job outlives this page, so reattach to it instead of leaving it
    // orphaned with the button re-enabled as though nothing were happening.
    resumeSheetJob();
    loadCores(); // Fetch initial core images
    updateCoreModelState();
    loadActionCatalog();
    updateDiagnostics();
    // 15s, not 5s: each call is a Celery round-trip to the inference worker, and
    // polling it three times a minute is plenty for a static device readout.
    setInterval(updateDiagnostics, 15000);
});

async function saveAppSetting(key, value) {
    try {
        await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ [key]: value })
        });
        console.log(`Setting ${key} saved: ${value}`);
    } catch (e) {
        console.error("Error saving setting:", e);
    }
}

// Report the WORKER's compute device.
//
// The previous version of this read the browser's WebGL renderer and labelled
// it "Hardware" — that reports the machine viewing the page, which tells you
// nothing about where inference runs and reads as confirmation when it is not.
// /api/compute-info round-trips through Celery to the process that actually
// owns the GPU.
async function updateDiagnostics() {
    const deviceEl = document.getElementById('compute-device');
    const detailEl = document.getElementById('compute-detail');
    if (!deviceEl) return;

    try {
        const resp = await fetch('/api/compute-info');
        const info = await resp.json();

        if (!resp.ok || info.error) {
            deviceEl.innerHTML = `<strong style="color:#e06c5a;">Worker unreachable</strong>`;
            detailEl.innerText = info.error || `HTTP ${resp.status}`;
            return;
        }

        if (info.device === 'cuda') {
            // A stale reading means the worker is mid-task and cannot answer —
            // it is the one moment the panel matters most, so label it rather
            // than passing the figures off as live.
            const busy = info.stale
                ? ` <span style="color:#e0a45a;">· busy, last read ${info.snapshot_age_s}s ago</span>`
                : '';
            deviceEl.innerHTML = `<strong style="color:#5ac37d;">CUDA</strong> — ${info.gpu_name || 'GPU'}${busy}`;
            const used = info.vram_reserved_gb ?? 0;
            detailEl.innerText =
                `${used} / ${info.vram_total_gb} GiB VRAM reserved · ${info.dtype}` +
                ` · torch ${info.torch_version} (CUDA ${info.torch_cuda_build})` +
                (info.loaded_pipelines?.length
                    ? ` · loaded: ${info.loaded_pipelines.join(', ')}`
                    : ' · no pipeline loaded yet');
        } else {
            // Flag this rather than showing it neutrally: on this host CPU means
            // something is misconfigured, not that a slower mode was chosen.
            deviceEl.innerHTML = `<strong style="color:#e0a45a;">CPU</strong> — GPU not in use`;
            detailEl.innerText =
                `torch ${info.torch_version} (CUDA build: ${info.torch_cuda_build || 'none'})` +
                ` · cuda_available=${info.cuda_available}` +
                ` — check nvidia-container-toolkit and COMPUTE_DEVICE, then run \`make gpu-check\`.`;
        }
    } catch (e) {
        deviceEl.innerHTML = `<strong style="color:#e06c5a;">Diagnostics failed</strong>`;
        detailEl.innerText = String(e);
    }
}

function updateGenSheetButtonState() {
    const btn = document.getElementById('gen-sheet-btn');
    if (!btn) return;
    // An ACTION is required - zero actions is zero cells, a job that finishes
    // in seconds having written nothing. A DIRECTION is not: an empty
    // selection is read by /api/jobs as the front row, which is a sensible
    // sheet rather than an empty one, so requiring it only forced a choice
    // before anything could be pressed.
    const acts = document.querySelectorAll('input[name="action"]:checked');
    btn.disabled = (acts.length === 0);
    if (typeof updateSheetEstimate === 'function') updateSheetEstimate();
}

function switchTab(tabId) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    event.currentTarget.classList.add('active');
    document.getElementById('tab-' + tabId).classList.add('active');
    if (pollInterval) clearInterval(pollInterval);
    // Re-stat the model cache when the core tab is opened. A page left open
    // across an archive/restore would otherwise keep showing the availability
    // it was rendered with, which is exactly when it is most misleading.
    if (tabId === 'core') refreshCoreModels();
}

// --- core model availability ---------------------------------------------
//
// The dropdown is rendered server-side from core_models.roster(), which stats
// the shared /models cache. A checkpoint that has been archived to cold storage
// cannot be fetched (HF_HUB_OFFLINE=1), so it is rendered disabled rather than
// offered — the failure it used to produce was an opaque "Model failed to load
// on worker" arriving seconds later in the queue panel, with no mention of the
// cache at all.

function updateCoreModelState() {
    const sel = document.getElementById('core-llm');
    const warn = document.getElementById('core-model-warning');
    const btn = document.getElementById('gen-core-btn');
    if (!sel || !warn || !btn) return;

    const opts = Array.from(sel.options);
    const usable = opts.filter(o => o.dataset.available === 'true');

    if (usable.length === 0) {
        // Every option disabled: the browser still reports one as selected, so
        // check the roster rather than the selection.
        warn.hidden = false;
        warn.innerHTML =
            '<strong>No core model is on disk.</strong> Step 1 cannot run until one is '
            + 'restored.<br>' + esc(opts[0] ? (opts[0].dataset.reason || '') : '');
        btn.disabled = true;
        return;
    }

    // A disabled option can still be the selection after a refresh flips it.
    if (sel.selectedOptions[0] && sel.selectedOptions[0].dataset.available !== 'true') {
        sel.value = usable[0].value;
    }

    warn.hidden = true;
    warn.innerHTML = '';
    if (!coreBusy) btn.disabled = false;
}

async function refreshCoreModels() {
    const sel = document.getElementById('core-llm');
    if (!sel) return;
    try {
        const res = await fetch('/api/core-models');
        if (!res.ok) return;
        const { models } = await res.json();
        const by = new Map(models.map(m => [m.value, m]));
        // Patch the existing options in place rather than rebuilding the select:
        // rebuilding drops the user's selection and closes an open dropdown.
        Array.from(sel.options).forEach(o => {
            const m = by.get(o.value);
            if (!m) return;
            o.disabled = !m.available;
            o.dataset.available = m.available ? 'true' : 'false';
            o.dataset.reason = m.reason || '';
            const base = o.textContent.replace(/ — not on disk$/, '').trim();
            o.textContent = m.available ? base : base + ' — not on disk';
        });
        updateCoreModelState();
    } catch (e) {
        console.error('Could not refresh core models:', e);
    }
}

async function loadCores() {
    try {
        const res = await fetch('/api/cores');
        const cores = await res.json();
        const picker = document.getElementById('core-picker');
        if (cores.length === 0) {
            picker.innerHTML = '<span style="color:var(--muted); font-size: 13px;">No core images found. Generate one first!</span>';
            return;
        }
        picker.innerHTML = cores.map(c => `
            <div class="core-item" id="core-sel-${c.id}" onclick="selectCore(${c.id})">
                <img src="${esc(c.file_path.split('/app').pop())}" title="${esc(c.prompt)}" loading="lazy" onerror="retryImage(this)"/>
            </div>
        `).join('');
        
        // auto-select first
        if(!selectedCoreId && cores.length > 0) selectCore(cores[0].id);
    } catch(e) { console.error("Error loading cores:", e); }
}

function selectCore(id) {
    selectedCoreId = id;
    document.querySelectorAll('.core-item').forEach(e => e.classList.remove('selected'));
    const el = document.getElementById(`core-sel-${id}`);
    if(el) el.classList.add('selected');
}

// Escape before interpolating anything server-supplied into HTML.
//
// Not defensive style — this was load-bearing. Worker errors reach the queue
// verbatim, and PyTorch's allocator assert contains double quotes:
//   !handles_.at(i) INTERNAL ASSERT FAILED at "/__w/pytorch/.../..cpp":467
// which broke out of title="${t.error}" and left the parser to invent
// attributes out of the rest of the message. Prompts are user text and are the
// same hazard with a friendlier name.
function esc(v) {
  return String(v ?? '').replace(/[&<>"']/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

// Ids whose Delete button is showing "Confirm?" right now.
//
// The queue re-renders every 3s by replacing innerHTML wholesale, which
// destroys the very button the pointer is over. A click needs mousedown and
// mouseup on the SAME element, so a re-render landing between them silently
// eats the click — no request, no error, nothing in the access log. Pause the
// refresh while a button is armed.
const armedDeletes = new Set();

// --- sheet jobs in the queue panel ---------------------------------------
//
// Sheet jobs live in the `jobs` table; Live Tasks only ever read
// /api/tasks/recent, which is the `sprite_images` table. So a sheet was
// invisible the moment you navigated away from the page that submitted it -
// no progress, no result, and a finished sheet that existed only as a URL
// nobody was holding. The two are merged here rather than given a second
// panel: from the operator's side it is one queue of work on one GPU.

// The model gateway's worker writes progress_msg = "Deferred - <reason>" on a
// task or job row that is waiting for its model to become active. That row is
// waiting, not failed and not progressing, so it gets a quiet tag instead of a
// pulsing 0%.
const DEFERRED_TAG = '<span class="tag tag-waiting">Deferred</span>';
function isDeferredMsg(msg) { return /^Deferred\b/.test(msg || ''); }

const JOB_TAGS = {
  done: '<span class="tag tag-success">Done</span>',
  failed: '<span class="tag tag-danger">Failed</span>',
  cancelled: '<span class="tag tag-danger">Cancelled</span>',
};

function renderJobCard(j) {
  const spec = j.spec || {};
  const acts = (spec.actions || []).join(', ') || 'sheet';
  const dirs = (spec.directions || []).length;
  const title = `Sheet: ${acts} · ${dirs} dir · ${spec.frames || '?'}f`;
  const running = !JOB_TAGS[j.status];
  const deferred = running && isDeferredMsg(j.progress_msg);
  const tag = JOB_TAGS[j.status]
    || (deferred ? DEFERRED_TAG
                 : `<span class="tag tag-working pulse">${j.progress_pct || 0}%</span>`);

  let body = '';
  if (j.status === 'failed' && j.error) {
    body = `<div style="color: var(--danger); font-size: 11px; margin-top: 4px;`
         + ` overflow: hidden; text-overflow: ellipsis; white-space: nowrap;"`
         + ` title="${esc(j.error)}">${esc(j.error)}</div>`;
  } else if (j.sheet_url) {
    body = `<a href="${esc(j.sheet_url)}" target="_blank">`
         + `<img src="${esc(j.sheet_url)}" alt="sheet" loading="lazy"`
         + ` style="width:100%; margin-top:6px; border-radius:4px;`
         + ` image-rendering: pixelated; background: rgba(0,0,0,.25);" /></a>`;
  } else if (running) {
    body = `<span class="progress-info${deferred ? ' is-deferred' : ''}">`
         + `${esc(j.progress_msg || j.stage || 'queued')}</span>`
         + `<div class="progress-bg"><div class="progress-fill"`
         + ` style="width: ${j.progress_pct || 0}%"></div></div>`;
  }

  const when = (j.updated_at || j.created_at || '').split('T')[1] || '';
  return `
    <div class="task-item" id="live-job-${esc(j.job_id)}">
      <div class="prompt-clip">${esc(title)}</div>
      <div class="meta">
        <span>${tag}</span>
        <span>${esc(when.split('.')[0])}</span>
      </div>
      ${body}
      <div class="task-actions">
        ${j.atlas_url ? `<button class="btn-sm btn-retry-sm" onclick="window.open('${esc(j.atlas_url)}','_blank')">Atlas</button>` : ''}
        ${running ? `<button class="btn-sm btn-danger-sm" onclick="cancelSheetJob('${esc(j.job_id)}')">Cancel</button>` : ''}
      </div>
    </div>
  `;
}

async function cancelSheetJob(jobId) {
  try {
    await fetch(`/api/jobs/${jobId}`, { method: 'DELETE' });
    updateQueue();
  } catch (e) { console.error('cancel failed:', e); }
}

// Both feeds sort into one list by time. sprite_images uses `timestamp`, jobs
// use `updated_at`; neither is comparable across tables except as a string,
// which is fine because both are ISO-8601 UTC.
function queueSortKey(item) {
  return item._job ? (item.updated_at || item.created_at || '')
                   : (item.timestamp || '');
}

async function updateQueue() {
  if (armedDeletes.size > 0) return;
  try {
    // Settle both before rendering. Rendering on whichever returns first made
    // the panel flicker between one feed and both.
    const [taskRes, jobRes] = await Promise.allSettled([
      fetch('/api/tasks/recent'),
      fetch('/api/jobs?limit=25'),
    ]);

    const tasks = taskRes.status === 'fulfilled' && taskRes.value.ok
      ? await taskRes.value.json() : [];
    let jobs = [];
    if (jobRes.status === 'fulfilled' && jobRes.value.ok) {
      const body = await jobRes.value.json();
      jobs = (body.jobs || []).map(j => Object.assign({ _job: true }, j));
    }

    const queueDiv = document.getElementById('task-queue');
    const merged = tasks.concat(jobs)
                        .sort((a, b) => queueSortKey(b).localeCompare(queueSortKey(a)));

    if (merged.length === 0) {
      queueDiv.innerHTML = '<p style="font-size: 12px; color: var(--muted); text-align: center; padding: 40px 0;">No history yet.</p>';
      return;
    }

    queueDiv.innerHTML = merged.map(t => {
      if (t._job) return renderJobCard(t);
      let statusTag = '';
      let progressLine = '';
      
      if (t.error) {
         statusTag = '<span class="tag tag-danger">Failed</span>';
         progressLine = `<div style="color: var(--danger); font-size: 11px; margin-top: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${esc(t.error)}">${esc(t.error)}</div>`;
      } else if (t.file_path) {
         statusTag = '<span class="tag tag-success">Done</span>';
      } else {
         const deferred = isDeferredMsg(t.progress_msg);
         statusTag = deferred ? DEFERRED_TAG
                              : `<span class="tag tag-working pulse">${t.progress_pct}%</span>`;
         progressLine = `
          <span class="progress-info${deferred ? ' is-deferred' : ''}">${esc(t.progress_msg || 'Preparing...')}</span>
          <div class="progress-bg"><div class="progress-fill" style="width: ${t.progress_pct}%"></div></div>
         `;
      }
      
      const isCore = t.image_type === 'core';
      const imgUrl = t.file_path ? `/images/${t.file_path.split('/').pop()}` : null;

      // Retry is offered on failures, not only on successes. It used to be
      // gated on file_path, so it appeared exactly where it was least useful —
      // a completed sprite — and never on the failed cards, which were left
      // with Delete as their only action.
      return `
        <div class="task-item" id="live-task-${t.id}">
          <div class="prompt-clip">${esc(t.prompt)}</div>
          <div class="meta">
            <span>${statusTag}</span>
            <span>${esc(t.timestamp.split('T')[1].split('.')[0])}</span>
          </div>
          ${progressLine}
          <div class="task-error" id="task-error-${t.id}"></div>
          <div class="task-actions">
              ${t.file_path && isCore ? `<button class="btn-sm btn-retry-sm" onclick="openCropModal('${esc(imgUrl)}', ${t.id})" style="border-color: var(--accent); color: var(--accent2);">Crop</button>` : ''}
              ${t.file_path || t.error ? `<button class="btn-sm btn-retry-sm" onclick="retryLiveTask(${t.id})">Retry</button>` : ''}
              <button class="btn-sm btn-danger-sm" id="del-btn-${t.id}" onclick="deleteTask(${t.id})">Delete</button>
          </div>
        </div>
      `;
    }).join('');
  } catch (e) { console.error(e); }
}

// Report a failure ON the card rather than through alert().
//
// alert()/confirm() are the wrong channel here for a reason that bit this page:
// Chrome offers "prevent this page from creating additional dialogs" after a
// couple of them, and once that box is ticked every later confirm() returns
// false for the lifetime of the page — no dialog, no request, no console error.
// Delete then looks broken and leaves nothing to diagnose.
function showTaskError(id, msg) {
    const el = document.getElementById(`task-error-${id}`);
    if (el) {
        el.textContent = msg;
        el.style.display = 'block';
    }
    console.error(`task ${id}: ${msg}`);
}

// FastAPI reports refusals as {"detail": "..."}; show the sentence, not the JSON.
async function errText(res) {
    const body = await res.text();
    try { return JSON.parse(body).detail ?? body; } catch { return body; }
}

async function retryLiveTask(id) {
    try {
        const res = await fetch('/api/task/' + id + '/retry', { method: 'POST' });
        if (!res.ok) {
            showTaskError(id, `Retry failed (HTTP ${res.status}): ${await errText(res)}`);
            return;
        }
        const data = await res.json();
        const mode = data.image_type === 'spritesheet' ? 'sheet' : data.image_type;
        pollTaskStatus(data.task_id, mode);
        updateQueue();
    } catch(e) { showTaskError(id, 'Retry failed: ' + e.message); }
}

// Two-step delete, confirmed on the button itself.
//
// First click arms it ("Confirm?"), second click sends the request, and it
// disarms itself after 4s. This replaces confirm() — see showTaskError — and
// also gives the queue refresh something to pause on, so the 3s re-render
// cannot delete the button out from under the second click.
async function deleteTask(id) {
    const btn = document.getElementById(`del-btn-${id}`);

    if (!armedDeletes.has(id)) {
        armedDeletes.add(id);
        if (btn) {
            btn.dataset.label = btn.textContent;
            btn.textContent = 'Confirm?';
            btn.classList.add('armed');
        }
        setTimeout(() => {
            if (!armedDeletes.delete(id)) return;
            const b = document.getElementById(`del-btn-${id}`);
            if (b) {
                b.textContent = b.dataset.label || 'Delete';
                b.classList.remove('armed');
            }
        }, 4000);
        return;
    }

    armedDeletes.delete(id);
    if (btn) { btn.disabled = true; btn.textContent = 'Deleting…'; }
    try {
        const res = await fetch('/api/task/' + id, { method: 'DELETE' });
        if (!res.ok) {
            showTaskError(id, `Delete failed (HTTP ${res.status}): ${await errText(res)}`);
            if (btn) { btn.disabled = false; btn.textContent = 'Delete'; }
            return;
        }
        const card = document.getElementById(`live-task-${id}`);
        if (card) card.remove();
        updateQueue();
    } catch(e) {
        showTaskError(id, 'Delete failed: ' + e.message);
        if (btn) { btn.disabled = false; btn.textContent = 'Delete'; }
    }
}

async function generateCore() {
  const promptElem = document.getElementById('core-prompt');
  if (!promptElem) return;
  const promptVal = promptElem.value.trim();
  if (!promptVal) return;
  const resultDiv = document.getElementById('core-result');
  const statusDiv = document.getElementById('core-status');
  const btn = document.getElementById('gen-core-btn');

  if (resultDiv) resultDiv.innerHTML = '<span class="preview-placeholder pulse">⏳ Sending task to worker...</span>';
  if (statusDiv) statusDiv.innerText = 'Initializing...';
  coreBusy = true;
  if (btn) btn.disabled = true;

  try {
    const llmElem = document.getElementById('core-llm');
    const fd = new FormData();
    fd.append('prompt', promptVal);
    // No select -> omit the field so the server applies the roster default.
    if (llmElem && llmElem.value) fd.append('llm_name', llmElem.value);
    const req = await fetch('/api/generate_core', { method: 'POST', body: fd });

    if (req.ok) {
      const data = await req.json();
      pollTaskStatus(data.task_id, 'core');
      updateQueue();
    } else {
      // FastAPI puts the reason in `detail`; printing the raw body showed the
      // operator {"detail":"..."} instead of the sentence inside it. A 409 is
      // the "model is not on disk" refusal, which is worth re-stating in the
      // dropdown too in case the cache changed under an open page.
      let msg = await req.text();
      try { msg = JSON.parse(msg).detail || msg; } catch (e) { /* not JSON */ }
      statusDiv.innerText = '❌ ' + msg;
      coreBusy = false;
      if (req.status === 409) refreshCoreModels();
      btn.disabled = false;
    }
  } catch (e) {
    statusDiv.innerText = '❌ Error: ' + e.message;
    coreBusy = false;
    btn.disabled = false;
  }
}

// --- sheet generation: async job API -------------------------------------
//
// Sheets go through POST /api/jobs, not the old /api/generate_sheet Celery
// task. The reason is duration, not tidiness: a full character is ~96 cells at
// roughly 33s each, so about an hour. Nothing survives that synchronously, and
// the old path also had no way to report which cell it was on.
//
// The job id is the durable handle. It resolves after a page reload, a worker
// restart or a broker flush, which is what lets this poller be re-attachable
// and is why something2 can poll the same endpoints.

let sheetJobId = null;
let sheetPoll = null;

function selectedActions() {
  return Array.from(document.querySelectorAll('input[name="action"]:checked'))
              .map(c => c.value);
}

function selectedDirections() {
  return Array.from(document.querySelectorAll('input[name="direction"]:checked'))
              .map(c => c.value);
}

// Shown before submitting. ~33s/cell measured, plus roughly four model loads
// at ~90s across the five build stages.
// --- pose library limits -------------------------------------------------
//
// A frame is a named pose in actions.py, not an interpolation, so the number of
// frames a sheet can have is bounded by the poses the chosen actions define.
// The input used to offer up to 8 against a library of 4; the excess was only
// discovered by the worker, which encoded keys 0..3, looked up 0..5, and died
// with KeyError: 'idle|s|4' after the turnaround pass had already run.
let actionCatalog = null;

async function loadActionCatalog() {
  try {
    const res = await fetch('/api/action-catalog');
    if (!res.ok) return;
    actionCatalog = await res.json();
    renderActions();
    applyFrameLimit();
  } catch (e) {
    console.error('Could not load the action catalog:', e);
  }
}

// The ceiling depends on WHICH actions are ticked, so recompute on every change
// rather than once at load.
// Walk stays ticked by default because it was the previous hardcoded default;
// everything else the library defines is offered unticked.
const DEFAULT_ACTION = 'walk';

function renderActions() {
  const grid = document.getElementById('actions-grid');
  if (!grid || !actionCatalog) return;
  const list = actionCatalog.actions || [];
  if (!list.length) {
    grid.innerHTML = '<span style="color:var(--danger); font-size:13px;">'
      + 'No actions defined in action_prompts.json.</span>';
    updateGenSheetButtonState();
    return;
  }
  // Preserve what was already ticked, so a catalog refresh does not silently
  // reset a selection the user made.
  const ticked = new Set(selectedActions());
  const anyTicked = ticked.size > 0;
  grid.innerHTML = list.map(a => {
    const on = anyTicked ? ticked.has(a.id) : a.id === DEFAULT_ACTION;
    return `<label class="action-cb"><input type="checkbox" name="action"`
         + ` value="${esc(a.id)}"${on ? ' checked' : ''} /> ${esc(a.label)}</label>`;
  }).join('');
  grid.querySelectorAll('input[name="action"]')
      .forEach(cb => cb.addEventListener('change', updateGenSheetButtonState));
  updateGenSheetButtonState();
}

function frameLimit() {
  if (!actionCatalog) return null;
  const per = actionCatalog.frames_by_action || {};
  const picked = selectedActions().filter(a => a in per);
  if (!picked.length) return actionCatalog.max_frames || null;
  return Math.min(...picked.map(a => per[a]));
}

function applyFrameLimit() {
  const el = document.getElementById('sheet-frames');
  const limit = frameLimit();
  if (!el || !limit) return;
  el.max = String(limit);
  // Clamp rather than leave an out-of-range value sitting in the box: a number
  // input does not refuse one, it just submits it.
  if ((parseInt(el.value, 10) || 0) > limit) el.value = String(limit);
}

function updateSheetEstimate() {
  const el = document.getElementById('sheet-estimate');
  if (!el) return;
  applyFrameLimit();
  const framesEl = document.getElementById('sheet-frames');
  const frames = framesEl ? parseInt(framesEl.value, 10) || 4 : 4;
  // No direction ticked builds the front row, so it is one direction, not zero.
  const dirCount = selectedDirections().length || 1;
  const cells = selectedActions().length * dirCount * frames;
  if (!cells) { el.innerText = ''; return; }
  const mins = Math.round((cells * 33 + 360) / 60);
  const limit = frameLimit();
  const cap = (limit && frames >= limit)
    ? ` ${limit} frames is the most these actions define.`
    : '';
  const front = selectedDirections().length ? '' : ' Front row only.';
  el.innerText = `${cells} cells — roughly ${mins} min on this card.${front} `
               + `The job keeps running if you close this page.${cap}`;
}

async function generateSheet() {
  if (!selectedCoreId) { alert("Please select a concept image first!"); return; }
  const actions = selectedActions();
  const directions = selectedDirections();
  if (!actions.length) { alert("Select at least one action!"); return; }

  const resultDiv = document.getElementById('sheet-result');
  const statusDiv = document.getElementById('sheet-status');
  const btn = document.getElementById('gen-sheet-btn');

  resultDiv.innerHTML = '<span class="preview-placeholder pulse">⏳ Queueing job...</span>';
  statusDiv.innerText = 'Submitting...';
  btn.disabled = true;

  try {
    // The job API takes the concept as a FILENAME under images/, not the
    // sprite_images row id the old endpoint used - a job has to stay
    // resolvable without a DB join.
    const coreRes = await fetch('/api/cores');
    const cores = await coreRes.json();
    const core = cores.find(c => c.id === selectedCoreId);
    if (!core) { throw new Error('selected concept no longer exists'); }

    const body = {
      concept_image: core.file_path.split('/').pop(),
      actions: actions,
      directions: directions,
      frames: parseInt(document.getElementById('sheet-frames').value, 10) || 4,
      cell: document.getElementById('sheet-cell').value,
      colors: parseInt(document.getElementById('sheet-colors').value, 10) || 24,
    };

    const req = await fetch('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!req.ok) {
      // FastAPI puts the reason in `detail`. A 400 here is a spec the pose
      // library cannot satisfy, which is worth reading rather than skimming
      // past as a JSON blob.
      let msg = await req.text();
      try { msg = JSON.parse(msg).detail || msg; } catch (e) { /* not JSON */ }
      statusDiv.innerText = '❌ ' + msg;
      btn.disabled = false;
      return;
    }

    const job = await req.json();
    sheetJobId = job.job_id;
    // Survive a reload. The job outlives the page, so the page should be able
    // to find its way back to it.
    try { localStorage.setItem('sheetJobId', sheetJobId); } catch (e) {}
    pollSheetJob(sheetJobId);
    updateQueue();
  } catch (e) {
    statusDiv.innerText = '❌ Error: ' + e.message;
    btn.disabled = false;
  }
}

function pollSheetJob(jobId) {
  const statusDiv = document.getElementById('sheet-status');
  const resultDiv = document.getElementById('sheet-result');
  const btn = document.getElementById('gen-sheet-btn');

  if (sheetPoll) clearInterval(sheetPoll);
  sheetPoll = setInterval(async () => {
    try {
      const res = await fetch(`/api/jobs/${jobId}`);
      if (res.status === 404) {
        clearInterval(sheetPoll);
        statusDiv.innerText = 'Job no longer exists.';
        btn.disabled = false;
        try { localStorage.removeItem('sheetJobId'); } catch (e) {}
        return;
      }
      const job = await res.json();

      if (job.status === 'done') {
        clearInterval(sheetPoll);
        // sheet_url is only present when finished, so its presence is the
        // readiness signal - no need to string-match on status.
        resultDiv.innerHTML = `
            <img src="${job.sheet_url}" alt="Spritesheet" onerror="retryImage(this)" />
            <div class="crop-btn-container">
                <a class="btn-crop" href="${job.atlas_url}" download>⬇ atlas.json</a>
            </div>`;
        statusDiv.innerText = '✅ Complete';
        btn.disabled = false;
        try { localStorage.removeItem('sheetJobId'); } catch (e) {}
        updateQueue();
        return;
      }

      if (job.status === 'failed' || job.status === 'cancelled') {
        clearInterval(sheetPoll);
        statusDiv.innerText = `❌ ${job.status}: ${job.error || ''}`;
        btn.disabled = false;
        try { localStorage.removeItem('sheetJobId'); } catch (e) {}
        updateQueue();
        return;
      }

      statusDiv.innerHTML = `
        <div style="font-weight: 700; color: var(--accent2);">
          ${job.stage || job.status} — ${job.progress_msg || ''}
        </div>
        <div class="progress-bg" style="width: 240px; margin: 8px auto;">
          <div class="progress-fill" style="width: ${job.progress_pct || 0}%"></div>
        </div>`;
    } catch (e) { console.error(e); }
    // 3s, not the 1.5s the old task poller used: these jobs run for an hour and
    // a cell takes ~33s, so a faster poll only adds requests.
  }, 3000);
}

// Re-attach after a reload rather than orphaning a running job.
function resumeSheetJob() {
  let saved = null;
  try { saved = localStorage.getItem('sheetJobId'); } catch (e) {}
  if (!saved) return;
  const btn = document.getElementById('gen-sheet-btn');
  if (btn) btn.disabled = true;
  sheetJobId = saved;
  pollSheetJob(saved);
}

function pollTaskStatus(taskId, mode) {
  const statusDiv = document.getElementById(`${mode}-status`);
  const resultDiv = document.getElementById(`${mode}-result`);
  const btn = document.getElementById(`gen-${mode}-btn`);

  if (pollInterval) clearInterval(pollInterval);
  pollInterval = setInterval(async () => {
    try {
      const resRecent = await fetch('/api/tasks/recent');
      const recentTasks = await resRecent.json();
      const me = recentTasks.find(t => t.task_id === taskId);

      if (me) {
        if (me.file_path) {
            clearInterval(pollInterval);
            const imgUrl = `/images/${me.file_path.split('/').pop()}`;
            resultDiv.innerHTML = `
                <img src="${imgUrl}" alt="Sprite" onerror="retryImage(this)" />
                <div class="crop-btn-container">
                    <button class="btn-crop" onclick="openCropModal('${imgUrl}', ${me.id})">✂️ Crop Character</button>
                </div>
            `;
            statusDiv.innerText = `✅ Success! Completed in ${me.duration_ms / 1000}s`;
            if (mode === 'core') coreBusy = false;
            btn.disabled = false;
            updateQueue();
            if (mode === 'core') loadCores(); // Refresh picker if we just made a core
            return;
        }
        if (me.error) {
            clearInterval(pollInterval);
            statusDiv.innerText = '❌ Error: ' + me.error;
            if (mode === 'core') coreBusy = false;
            btn.disabled = false;
            updateQueue();
            return;
        }
        statusDiv.innerHTML = `
          <div style="font-weight: 700; color: var(--accent2);">${me.progress_msg || 'Queued'}</div>
          <div class="progress-bg" style="width: 240px; margin: 8px auto;"><div class="progress-fill" style="width: ${me.progress_pct}%"></div></div>
        `;
      }
    } catch (e) { console.error(e); }
  }, 1500);
}

// --- model gateway ---------------------------------------------------------
//
// One active image model holds the 12 GB card; jobs for any other model are
// deferred by the worker instead of forcing a reload mid-queue. This shows the
// active model in the nav, offers the switch, and blocks the page with a popup
// while a switch is in progress. Spec: .ai/specs/model-gateway/plan.md.
//
// Auth: the rest of this file sends no bearer token (it predates enforcement).
// The gateway calls send the key the React UI stores under `sprite.apiToken`
// (Settings tab) - same origin, so the same localStorage. Without one, the
// endpoints answer 401 and the indicator says so rather than looking broken.

const GW_TOKEN_KEY = 'sprite.apiToken';
let gwState = null;          // last GET /api/model-gateway body
let gwFetchedAt = 0;         // Date.now() when gwState arrived
let gwHiddenSwitch = null;   // switching.started the user chose to hide
let gwTicker = null;         // 1s refresh of the popup's elapsed time
let gwSelectTouched = false; // do not reset a choice the user is making

function gwAuthHeaders(extra) {
  const h = Object.assign({}, extra || {});
  let token = '';
  try { token = localStorage.getItem(GW_TOKEN_KEY) || ''; } catch (e) {}
  if (token) h['Authorization'] = `Bearer ${token}`;
  return h;
}

function gwLabel(model) {
  if (!model) return '';
  const c = ((gwState && gwState.choices) || []).find(x => x.value === model);
  return c ? c.label : model;
}

function gwClock(s) {
  s = Math.max(0, Math.round(s || 0));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

async function updateGateway() {
  try {
    const res = await fetch('/api/model-gateway', { headers: gwAuthHeaders() });
    if (!res.ok) {
      renderGatewayIndicator(null, res.status === 401 || res.status === 403
        ? 'API key needed - set it in the React UI Settings tab'
        : `status unavailable (HTTP ${res.status})`);
      return;
    }
    gwState = await res.json();
    gwFetchedAt = Date.now();
    renderGatewayIndicator(gwState);
    renderSwitchingPopup();
    const overlay = document.getElementById('gw-overlay');
    if (overlay && overlay.style.display !== 'none') renderGatewayModal();
  } catch (e) {
    renderGatewayIndicator(null, 'status unavailable');
    console.error('model gateway:', e);
  }
}

function renderGatewayIndicator(st, problem) {
  const btn = document.getElementById('gw-indicator');
  const label = document.getElementById('gw-indicator-label');
  const extra = document.getElementById('gw-indicator-extra');
  if (!btn || !label || !extra) return;
  btn.classList.remove('is-active', 'is-switching', 'is-error');

  if (!st) {
    btn.classList.add('is-error');
    label.textContent = 'model: ?';
    extra.textContent = problem || '';
    btn.title = problem || '';
    return;
  }

  const a = st.active;
  if (st.switching) {
    btn.classList.add('is-switching');
    label.textContent = `switching to ${gwLabel(st.switching.to)}`;
  } else if (a) {
    btn.classList.add('is-active');
    label.textContent = gwLabel(a.model);
  } else {
    label.textContent = 'no model yet';
  }

  const bits = [];
  if (a && a.pinned && a.pin_release_in_s != null) {
    bits.push(`pinned · releases in ${gwClock(a.pin_release_in_s)}`);
  } else if (a && a.pinned) {
    bits.push('pinned');
  }
  let deferred = 0, queued = 0, running = 0;
  Object.values(st.by_model || {}).forEach(m => {
    deferred += m.deferred || 0; queued += m.queued || 0; running += m.running || 0;
  });
  if (running) bits.push(`${running} running`);
  if (queued) bits.push(`${queued} queued`);
  if (deferred) bits.push(`${deferred} deferred`);
  extra.textContent = bits.length ? '· ' + bits.join(' · ') : '';
  btn.title = 'Active image model - click to switch' + (a ? `\n${a.model}` : '');
}

function openGatewayModal() {
  const overlay = document.getElementById('gw-overlay');
  if (!overlay) return;
  gwSelectTouched = false;
  setGatewayMessage(null);
  overlay.style.display = 'flex';
  renderGatewayModal();
  updateGateway();
}

function closeGatewayModal() {
  const overlay = document.getElementById('gw-overlay');
  if (overlay) overlay.style.display = 'none';
}

function renderGatewayModal() {
  const cur = document.getElementById('gw-current');
  const sel = document.getElementById('gw-select');
  const list = document.getElementById('gw-pending');
  if (!cur || !sel || !list) return;
  const st = gwState;
  if (!st) {
    cur.textContent = 'Status not loaded yet.';
    return;
  }

  const a = st.active;
  cur.innerHTML = a
    ? `Active: <strong>${esc(gwLabel(a.model))}</strong>`
      + (a.pinned ? ` · pinned${a.pin_release_in_s != null
          ? ` · releases in ${gwClock(a.pin_release_in_s)}` : ''}` : '')
      + `<div class="gw-sub">idle ${gwClock(a.idle_for_s)} · deferred jobs for other `
      + `models run after ${gwClock(st.idle_s)} idle</div>`
    : 'No model is active yet.';

  // Rebuild the options only while the user is not mid-choice, and keep the
  // selection: a 3 s poll must not close the dropdown or reset it.
  if (!gwSelectTouched) {
    const keep = sel.value || (a && a.model) || '';
    const opt = c => `<option value="${esc(c.value)}"${c.available ? '' : ' disabled'}>`
      + `${esc(c.label)}${c.available ? '' : ' - not on disk'}</option>`;
    const choices = st.choices || [];
    const normal = choices.filter(c => !c.fixed).map(opt).join('');
    const fixed = choices.filter(c => c.fixed).map(opt).join('');
    sel.innerHTML = normal
      + (fixed ? `<optgroup label="Fixed-model jobs">${fixed}</optgroup>` : '');
    if (keep && choices.some(c => c.value === keep && c.available)) sel.value = keep;
    sel.onchange = () => { gwSelectTouched = true; setGatewayMessage(null); };
    sel.onfocus = () => { gwSelectTouched = true; };
  }

  const pending = st.pending || [];
  list.innerHTML = pending.length
    ? pending.map(p => {
        const state = p.running
          ? '<span class="tag tag-working">Running</span>'
          : p.deferred
            ? `<span class="tag tag-waiting" title="${esc(p.deferred)}">Deferred</span>`
            : '<span class="tag tag-core">Queued</span>';
        const why = p.deferred && !p.running
          ? `<div class="gw-sub">${esc(p.deferred)}</div>` : '';
        return `<div class="gw-pending-row"><div>`
          + `<div>${esc(p.task || 'task')}</div>`
          + `<div class="gw-sub">${esc(gwLabel(p.model))}</div>${why}</div>`
          + `<div>${state}</div></div>`;
      }).join('')
    : '<div class="gw-sub">Nothing queued.</div>';
}

function setGatewayMessage(html, isError) {
  const el = document.getElementById('gw-message');
  if (!el) return;
  if (!html) { el.hidden = true; el.innerHTML = ''; return; }
  el.hidden = false;
  el.classList.toggle('is-error', !!isError);
  el.innerHTML = html;
}

async function gatewaySwitch(force) {
  const sel = document.getElementById('gw-select');
  const model = sel ? sel.value : '';
  if (!model) { setGatewayMessage('Pick a model first.', true); return; }
  const label = gwLabel(model);

  if (force) {
    const idle = gwState ? `${gwState.idle_s}s` : 'the idle window';
    // confirm() can be silenced by the browser (see showTaskError); if it is,
    // this returns false and nothing is sent, which is the safe direction.
    if (!confirm(`Force switch to ${label}?\n\n`
        + 'It takes effect now. Jobs already queued for other models are '
        + `deferred: they wait until ${label} has been idle for ${idle}, `
        + 'then run. A job that is already running is not interrupted - '
        + 'the switch waits for it.')) return;
  }

  const btns = ['gw-switch-btn', 'gw-force-btn'].map(id => document.getElementById(id));
  btns.forEach(b => { if (b) b.disabled = true; });
  try {
    const res = await fetch('/api/model-gateway/switch', {
      method: 'POST',
      headers: gwAuthHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ model, force: !!force }),
    });
    if (res.status === 409) {
      setGatewayMessage(`${esc(await errText(res))}<br>`
        + `<button class="btn-sm" onclick="gatewaySwitch(true)">Force switch now</button>`);
      return;
    }
    if (!res.ok) {
      const hint = res.status === 401 || res.status === 403
        ? ' - this needs an API key with the generate scope (React UI, Settings tab).' : '';
      setGatewayMessage(`Switch failed (HTTP ${res.status}): ${esc(await errText(res))}${esc(hint)}`, true);
      return;
    }
    const body = await res.json();
    gwSelectTouched = false;
    if (body.status === 'already_active') {
      setGatewayMessage(`${esc(label)} is already the active model.`);
    } else {
      setGatewayMessage(null);
      closeGatewayModal();
      gwHiddenSwitch = null; // a switch the user just asked for is shown
    }
    updateGateway();
  } catch (e) {
    setGatewayMessage('Switch failed: ' + esc(e.message), true);
  } finally {
    btns.forEach(b => { if (b) b.disabled = false; });
  }
}

const GW_PHASES = {
  queued: 'Waiting for the worker to pick up the switch.',
  waiting: 'Waiting for the running job to finish - it is never interrupted.',
  loading: 'Loading the new model onto the GPU.',
};

function renderSwitchingPopup() {
  const pop = document.getElementById('gw-switching');
  if (!pop) return;
  const sw = gwState && gwState.switching;

  // Done only when `switching` is null - never on elapsed time.
  if (!sw || gwHiddenSwitch === sw.started) {
    pop.style.display = 'none';
    if (!sw) gwHiddenSwitch = null;
    if (gwTicker) { clearInterval(gwTicker); gwTicker = null; }
    return;
  }

  const expected = Number(sw.expected_s) || 10;
  const elapsed = (Number(sw.elapsed_s) || 0) + (Date.now() - gwFetchedAt) / 1000;
  const over = elapsed > expected;
  document.getElementById('gw-sw-title').textContent =
    `Switching model to ${gwLabel(sw.to)}, please wait`;
  document.getElementById('gw-sw-phase').textContent =
    (GW_PHASES[sw.phase] || sw.phase || '')
    + (sw.from ? ` (from ${gwLabel(sw.from)})` : '');
  document.getElementById('gw-sw-time').textContent =
    `elapsed ${Math.round(elapsed)}s / expected ~${Math.round(expected)}s`
    + (over ? ' - taking longer than the last load' : '');
  const bar = document.getElementById('gw-sw-bar');
  bar.classList.toggle('indeterminate', over);
  bar.style.width = over ? '' : `${Math.min(100, (elapsed / expected) * 100)}%`;
  pop.style.display = 'flex';

  // Tick the elapsed figure between 3 s polls.
  if (!gwTicker) gwTicker = setInterval(renderSwitchingPopup, 1000);
}

function hideSwitchingPopup() {
  const sw = gwState && gwState.switching;
  gwHiddenSwitch = sw ? sw.started : null;
  renderSwitchingPopup();
}
