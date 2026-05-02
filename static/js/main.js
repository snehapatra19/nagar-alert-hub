// Nagar Alert Hub — Main JavaScript

// ── Toast Notifications ───────────────────────────────────────────────
const toastContainer = (() => {
  const el = document.createElement('div');
  el.className = 'toast-container';
  document.body.appendChild(el);
  return el;
})();

function showToast(msg, type = 'success', duration = 3500) {
  const icons = { success: '✅', error: '❌', warning: '⚠️' };
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.innerHTML = `<span>${icons[type]}</span> ${msg}`;
  toastContainer.appendChild(t);
  setTimeout(() => {
    t.style.opacity = '0';
    t.style.transform = 'translateY(-10px)';
    t.style.transition = 'all 0.3s ease';
    setTimeout(() => t.remove(), 300);
  }, duration);
}

// ── Format helpers ────────────────────────────────────────────────────
function timeAgo(dateStr) {
  const diff = (Date.now() - new Date(dateStr)) / 1000;
  if (diff < 60)   return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff/60)}m ago`;
  if (diff < 86400)return `${Math.floor(diff/3600)}h ago`;
  return `${Math.floor(diff/86400)}d ago`;
}

// ── Geolocation ───────────────────────────────────────────────────────
function getUserLocation() {
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) return reject('Geolocation not supported');
    navigator.geolocation.getCurrentPosition(
      pos => resolve({ lat: pos.coords.latitude, lng: pos.coords.longitude }),
      err => reject(err.message),
      { timeout: 8000 }
    );
  });
}

// Expose globally
window.showToast = showToast;
window.timeAgo = timeAgo;
window.getUserLocation = getUserLocation;
