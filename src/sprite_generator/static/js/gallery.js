async function deleteTask(id) {
    if(!confirm('Permanently remove this task and image?')) return;
    try {
        const res = await fetch('/api/task/' + id, { method: 'DELETE' });
        if (res.ok) {
            const card = document.getElementById('card-' + id);
            if (card) card.remove();
            
            // Check if gallery is empty after deletion
            const grid = document.getElementById('gallery-grid');
            if (grid && grid.querySelectorAll('.sprite-card').length === 0) {
                grid.innerHTML = '<div class="empty"><span>🖼️</span><p>No history.</p></div>';
            }
        }
    } catch(e) { alert('Delete failed: ' + e.message); }
}
