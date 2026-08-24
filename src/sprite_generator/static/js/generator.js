let pollInterval = null;
let selectedCoreId = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    updateQueue();
    setInterval(updateQueue, 3000);
    
    // Both axes toggle the button and move the estimate.
    document.querySelectorAll('input[name="action"], input[name="direction"]')
        .forEach(cb => cb.addEventListener('change', updateGenSheetButtonState));
    updateGenSheetButtonState();
    // A running job outlives this page, so reattach to it instead of leaving it
    // orphaned with the button re-enabled as though nothing were happening.
    resumeSheetJob();
    loadCores(); // Fetch initial core images
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
    // Both axes are required now. An action with no direction - or a direction
    // with no action - is zero cells, which submits happily and produces a job
    // that finishes in seconds having written nothing.
    const acts = document.querySelectorAll('input[name="action"]:checked');
    const dirs = document.querySelectorAll('input[name="direction"]:checked');
    btn.disabled = (acts.length === 0 || dirs.length === 0);
    if (typeof updateSheetEstimate === 'function') updateSheetEstimate();
}

function switchTab(tabId) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    event.currentTarget.classList.add('active');
    document.getElementById('tab-' + tabId).classList.add('active');
    if (pollInterval) clearInterval(pollInterval);
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

async function updateQueue() {
  if (armedDeletes.size > 0) return;
  try {
    const res = await fetch('/api/tasks/recent');
    const tasks = await res.json();
    const queueDiv = document.getElementById('task-queue');

    if (tasks.length === 0) {
      queueDiv.innerHTML = '<p style="font-size: 12px; color: var(--muted); text-align: center; padding: 40px 0;">No history yet.</p>';
      return;
    }

    queueDiv.innerHTML = tasks.map(t => {
      let statusTag = '';
      let progressLine = '';
      
      if (t.error) {
         statusTag = '<span class="tag tag-danger">Failed</span>';
         progressLine = `<div style="color: var(--danger); font-size: 11px; margin-top: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${esc(t.error)}">${esc(t.error)}</div>`;
      } else if (t.file_path) {
         statusTag = '<span class="tag tag-success">Done</span>';
      } else {
         statusTag = `<span class="tag tag-working pulse">${t.progress_pct}%</span>`;
         progressLine = `
          <span class="progress-info">${esc(t.progress_msg || 'Preparing...')}</span>
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
  if (btn) btn.disabled = true;

  try {
    const llmElem = document.getElementById('core-llm');
    const llm_name = llmElem ? llmElem.value : 'stabilityai/sdxl-turbo';
    const fd = new FormData();
    fd.append('prompt', promptVal);
    fd.append('llm_name', llm_name);
    const req = await fetch('/api/generate_core', { method: 'POST', body: fd });

    if (req.ok) {
      const data = await req.json();
      pollTaskStatus(data.task_id, 'core');
      updateQueue();
    } else {
      statusDiv.innerText = '❌ Error: ' + await req.text();
      btn.disabled = false;
    }
  } catch (e) {
    statusDiv.innerText = '❌ Error: ' + e.message;
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
function updateSheetEstimate() {
  const el = document.getElementById('sheet-estimate');
  if (!el) return;
  const framesEl = document.getElementById('sheet-frames');
  const frames = framesEl ? parseInt(framesEl.value, 10) || 4 : 4;
  const cells = selectedActions().length * selectedDirections().length * frames;
  if (!cells) { el.innerText = ''; return; }
  const mins = Math.round((cells * 33 + 360) / 60);
  el.innerText = `${cells} cells — roughly ${mins} min on this card. `
               + `The job keeps running if you close this page.`;
}

async function generateSheet() {
  if (!selectedCoreId) { alert("Please select a concept image first!"); return; }
  const actions = selectedActions();
  const directions = selectedDirections();
  if (!actions.length) { alert("Select at least one action!"); return; }
  if (!directions.length) { alert("Select at least one direction!"); return; }

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
      statusDiv.innerText = '❌ Error: ' + await req.text();
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
            btn.disabled = false;
            updateQueue();
            if (mode === 'core') loadCores(); // Refresh picker if we just made a core
            return;
        }
        if (me.error) {
            clearInterval(pollInterval);
            statusDiv.innerText = '❌ Error: ' + me.error;
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
