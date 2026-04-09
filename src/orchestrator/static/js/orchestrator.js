function getIcon(name) {
  const n = name.toLowerCase();
  if (n.includes('llama')) return { icon: '🦙', cls: 'llama' };
  if (n.includes('mistral')) return { icon: '🌪️', cls: 'mistral' };
  if (n.includes('gemma')) return { icon: '💎', cls: 'gemma' };
  return { icon: '🤖', cls: 'other' };
}

function showToast(msg, type = 'success') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = `toast show ${type}`;
  setTimeout(() => t.className = 'toast', 3000);
}

async function loadModels() {
  const res = await fetch('/api/models');
  const models = await res.json();
  const grid = document.getElementById('models-grid');
  document.getElementById('model-count').textContent = models.length;
  const total = models.reduce((s, m) => s + m.size_mb, 0);
  document.getElementById('total-size').textContent = total >= 1024
    ? (total / 1024).toFixed(1) + ' GB'
    : total.toFixed(0) + ' MB';

  if (!models.length) {
    grid.innerHTML = `<div class="empty-state"><div class="icon">📂</div><p>No models downloaded yet.</p></div>`;
    return;
  }

  grid.innerHTML = models.map(m => {
    const { icon, cls } = getIcon(m.name);
    return `
      <div class="model-card" id="card-${encodeURIComponent(m.file)}">
        <div class="model-card-header">
          <div class="model-icon ${cls}">${icon}</div>
          <div style="flex:1">
            <div class="model-name">${m.name}</div>
            <div class="model-file">${m.file}</div>
          </div>
        </div>
        <div class="model-footer">
          <span class="size-badge">${m.size_mb >= 1024 ? (m.size_mb/1024).toFixed(1)+' GB' : m.size_mb+' MB'}</span>
          <button class="btn-delete" onclick="deleteModel('${m.file}', '${m.name}')">🗑 Remove</button>
        </div>
      </div>`;
  }).join('');
}

async function deleteModel(filename, name) {
  if (!confirm(`Remove "${name}"? This will delete the .gguf file from disk.`)) return;
  const res = await fetch(`/api/models/${encodeURIComponent(filename)}`, { method: 'DELETE' });
  const data = await res.json();
  if (data.status === 'deleted') {
    showToast(`Removed ${name}`, 'success');
    loadModels();
  } else {
    showToast('Could not delete model', 'error');
  }
}

async function startDownload() {
  const repo = document.getElementById('repo').value.trim();
  const file = document.getElementById('file').value.trim();
  if (!repo || !file) { showToast('Please fill in both fields', 'error'); return; }

  const btn = document.getElementById('dl-btn');
  const log = document.getElementById('log-box');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Downloading...';
  log.innerHTML = '';
  log.classList.add('active');

  const es = new EventSource(`/api/download/stream?repo=${encodeURIComponent(repo)}&file=${encodeURIComponent(file)}`);
  es.onmessage = (e) => {
    if (e.data === '__DONE__') {
      es.close();
      const line = document.createElement('div');
      line.className = 'log-line done';
      line.textContent = '✓ Download complete!';
      log.appendChild(line);
      log.scrollTop = log.scrollHeight;
      btn.disabled = false;
      btn.textContent = 'Download';
      showToast(`Downloaded ${file}!`, 'success');
      loadModels();
      return;
    }
    if (e.data.trim()) {
      const line = document.createElement('div');
      line.className = 'log-line';
      line.textContent = e.data;
      log.appendChild(line);
      log.scrollTop = log.scrollHeight;
    }
  };
  es.onerror = () => {
    es.close();
    btn.disabled = false;
    btn.textContent = 'Download';
    showToast('Download failed or connection lost', 'error');
  };
}

// Initial load
document.addEventListener('DOMContentLoaded', () => {
  loadModels();
  setInterval(loadModels, 10000);
});
