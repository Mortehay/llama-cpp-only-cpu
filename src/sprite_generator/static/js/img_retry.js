// One-shot recovery for images the browser refuses to load from its own cache.
//
// Chrome can answer an <img> request straight out of its disk cache, and when
// that entry is unreadable it fails the load with ERR_CACHE_READ_FAILURE -
// no network attempt, no retry, just a broken thumbnail. The /images mount now
// sends no-store so new responses never land in that cache, but entries stored
// before that fix are still poisoned and stay usable until they go stale. A
// cache-busted reload gives the image a fresh cache key, which forces the
// request onto the network and past the bad entry.
//
// Loaded from <head> in base.html on purpose: an inline onerror can fire while
// the page is still parsing, so this has to be defined before any <img> is.
function retryImage(img) {
    if (img.dataset.cacheRetried) return;   // one attempt: a real 404 must stay broken
    img.dataset.cacheRetried = '1';
    const src = img.getAttribute('src').split('?')[0];
    img.src = src + '?cb=' + Date.now();
}
