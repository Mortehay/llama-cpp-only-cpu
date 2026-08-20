let pollInterval = null;
let selectedCoreId = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    updateQueue();
    setInterval(updateQueue, 3000);
    
    // Listen for action checkbox changes to toggle button
    const actionCbs = document.querySelectorAll('input[name="action"]');
    actionCbs.forEach(cb => {
        cb.addEventListener('change', updateGenSheetButtonState);
    });
    updateGenSheetButtonState();
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
            deviceEl.innerHTML = `<strong style="color:#5ac37d;">CUDA</strong> — ${info.gpu_name || 'GPU'}`;
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
    const checked = document.querySelectorAll('input[name="action"]:checked');
    btn.disabled = (checked.length === 0);
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
                <img src="${c.file_path.split('/app').pop()}" title="${c.prompt}"/>
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

async function updateQueue() {
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
         progressLine = `<div style="color: var(--danger); font-size: 11px; margin-top: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${t.error}">${t.error}</div>`;
      } else if (t.file_path) {
         statusTag = '<span class="tag tag-success">Done</span>';
      } else {
         statusTag = `<span class="tag tag-working pulse">${t.progress_pct}%</span>`;
         progressLine = `
          <span class="progress-info">${t.progress_msg || 'Preparing...'}</span>
          <div class="progress-bg"><div class="progress-fill" style="width: ${t.progress_pct}%"></div></div>
         `;
      }
      
      const isCore = t.image_type === 'core';
      const imgUrl = t.file_path ? `/images/${t.file_path.split('/').pop()}` : null;

      return `
        <div class="task-item" id="live-task-${t.id}">
          <div class="prompt-clip">${t.prompt}</div>
          <div class="meta">
            <span>${statusTag}</span>
            <span>${t.timestamp.split('T')[1].split('.')[0]}</span>
          </div>
          ${progressLine}
          <div class="task-actions">
              ${t.file_path ? `
                  ${isCore ? `<button class="btn-sm btn-retry-sm" onclick="openCropModal('${imgUrl}', ${t.id})" style="border-color: var(--accent); color: var(--accent2);">Crop</button>` : ''}
                  <button class="btn-sm btn-retry-sm" onclick="retryLiveTask(${t.id})">Retry</button>
              ` : ''}
              <button class="btn-sm btn-danger-sm" onclick="deleteTask(${t.id})">Delete</button>
          </div>
        </div>
      `;
    }).join('');
  } catch (e) { console.error(e); }
}

async function retryLiveTask(id) {
    if(!confirm('Duplicate and retry this task?')) return;
    try {
        const res = await fetch('/api/task/' + id + '/retry', { method: 'POST' });
        if (res.ok) {
            const data = await res.json();
            const mode = data.image_type === 'spritesheet' ? 'sheet' : data.image_type;
            pollTaskStatus(data.task_id, mode);
            updateQueue();
        } else {
            alert('Retry failed: ' + await res.text());
        }
    } catch(e) { alert('Retry failed: ' + e.message); }
}

async function deleteTask(id) {
    if(!confirm('Permanently cancel and remove this task?')) return;
    try {
        const res = await fetch('/api/task/' + id, { method: 'DELETE' });
        if (res.ok) updateQueue();
    } catch(e) { alert('Delete failed: ' + e.message); }
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

async function generateSheet() {
  if (!selectedCoreId) { alert("Please select a core image first!"); return; }
  const checkboxes = document.querySelectorAll('input[name="action"]:checked');
  if (checkboxes.length === 0) { alert("Select at least one action!"); return; }
  
  const actions = Array.from(checkboxes).map(c => c.value);
  
  const resultDiv = document.getElementById('sheet-result');
  const statusDiv = document.getElementById('sheet-status');
  const btn = document.getElementById('gen-sheet-btn');

  if (resultDiv) resultDiv.innerHTML = '<span class="preview-placeholder pulse">⏳ Sending task to worker...</span>';
  if (statusDiv) statusDiv.innerText = 'Initializing...';
  if (btn) btn.disabled = true;

  try {
    const llmElem = document.getElementById('sheet-llm');
    const llm_name = llmElem ? llmElem.value : 'stabilityai/sdxl-turbo';
    const sizeElem = document.getElementById('sheet-frame-size');
    const size = sizeElem ? sizeElem.value : 128;
    const motionElem = document.getElementById('sheet-motion-steps');
    const motion_steps = motionElem ? motionElem.value : 4;
    
    const fd = new FormData();
    fd.append('parent_id', selectedCoreId);
    fd.append('actions', JSON.stringify(actions));
    fd.append('llm_name', llm_name);
    fd.append('width', size);
    fd.append('height', size);
    fd.append('motion_steps', motion_steps);
    const req = await fetch('/api/generate_sheet', { method: 'POST', body: fd });

    if (req.ok) {
      const data = await req.json();
      pollTaskStatus(data.task_id, 'sheet');
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
                <img src="${imgUrl}" alt="Sprite" />
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
