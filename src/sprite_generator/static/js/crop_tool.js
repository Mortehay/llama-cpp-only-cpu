let activeCropper = null;
let activeCropSourceId = null;

function openCropModal(url, sourceId) {
    const overlay = document.getElementById('crop-overlay');
    const img = document.getElementById('crop-image');
    activeCropSourceId = sourceId;
    img.src = url;
    if (overlay) overlay.style.display = 'flex';

    if (activeCropper) activeCropper.destroy();
    
    // Initialize activeCropper after a short delay to ensure image is loaded and modal is visible
    setTimeout(() => {
        activeCropper = new Cropper(img, {
            viewMode: 1,
            dragMode: 'move',
            autoCropArea: 0.5,
            restore: false,
            guides: true,
            center: true,
            highlight: false,
            cropBoxMovable: true,
            cropBoxResizable: true,
            toggleDragModeOnDblclick: false,
        });
    }, 100);
}

function closeCropModal() {
    const overlay = document.getElementById('crop-overlay');
    if (overlay) overlay.style.display = 'none';
    if (activeCropper) {
        activeCropper.destroy();
        activeCropper = null;
    }
}

async function saveCrop() {
    if (!activeCropper || !activeCropSourceId) return;
    
    const data = activeCropper.getData(true); // true for rounded integers
    const btn = document.getElementById('confirm-crop-btn');
    if (!btn) return;
    
    const originalText = btn.innerText;
    btn.disabled = true;
    btn.innerText = '⌛ Saving...';

    try {
        const res = await fetch('/api/crop', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                source_id: activeCropSourceId,
                x: data.x,
                y: data.y,
                w: data.width,
                h: data.height
            })
        });

        if (res.ok) {
            const result = await res.json();
            closeCropModal();
            
            // If we are on the generator page, refresh the cores picker
            if (typeof loadCores === 'function') {
                loadCores();
            }
            
            alert('Crop saved successfully! It is now available in Step 2.');
        } else {
            alert('Crop failed: ' + await res.text());
        }
    } catch (e) {
        alert('Crop error: ' + e.message);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerText = originalText;
        }
    }
}
