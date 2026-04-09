async function fetchStats() {
  const tbody = document.getElementById('stats-body');
  try {
    const res = await fetch('/api/stats');
    const data = await res.json();
    
    // Summary
    document.getElementById('total-reqs').textContent = data.total_count;
    document.getElementById('avg-tps').textContent = data.avg_tps.toFixed(1);
    document.getElementById('total-tokens').textContent = (data.total_tokens / 1000).toFixed(1) + 'k';

    if (!data.history.length) {
      tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; padding: 2rem; color: var(--muted)">No stats logged yet.</td></tr>';
      return;
    }

    tbody.innerHTML = data.history.map(row => `
      <tr>
        <td>${new Date(row.created_at).toLocaleTimeString()}</td>
        <td style="font-weight: 500">${row.model_name}</td>
        <td style="color: var(--accent)">${row.tokens_per_second.toFixed(1)}</td>
        <td>${row.total_tokens}</td>
        <td style="color: var(--muted)">${row.total_duration_ms.toFixed(0)}ms</td>
      </tr>
    `).join('');

  } catch (err) {
    console.error(err);
    tbody.innerHTML = '<tr><td colspan="5" style="color: var(--red); text-align:center">Error loading statistics.</td></tr>';
  }
}

document.addEventListener('DOMContentLoaded', () => {
  fetchStats();
  setInterval(fetchStats, 15000);
});
