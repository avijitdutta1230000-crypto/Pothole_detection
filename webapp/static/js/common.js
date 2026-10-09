/* Shared helpers for every page. */
const PW = {
  DEFAULT_CENTER: [12.9692, 79.1559], // VIT Vellore: the map starts here until GPS answers
  sevColor: { HIGH: "#b8321f", MEDIUM: "#b86a0e", LOW: "#2c7448" },
  statusLabel: { OPEN: "Open", IN_PROGRESS: "In progress", RESOLVED: "Fixed", REJECTED: "Closed" },

  esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  },

  fmtTime(iso) {
    if (!iso) return "";
    const d = new Date(iso);
    return isNaN(d) ? iso : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
  },

  fmtDist(m) {
    if (m >= 1000) return `${(m / 1000).toFixed(1).replace(/\.0$/, "")} km`;
    if (m >= 100) return `${Math.round(m / 10) * 10} m`;
    return `${Math.max(5, Math.round(m / 5) * 5)} m`;
  },

  async api(url, opts) {
    try {
      const r = await fetch(url, opts);
      let data;
      try { data = await r.json(); } catch { data = null; }
      if (!data) data = { ok: false, error: `The server answered with an error (${r.status}). Try again.` };
      if (!r.ok && data.ok !== false) data = { ok: false, error: data.error || `Request failed (${r.status}).` };
      return data;
    } catch {
      return { ok: false, error: "No connection to the server. Check your internet and try again." };
    }
  },

  tiles(map) {
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }).addTo(map);
  },

  haversine(a, b) {
    const R = 6371000, toRad = (x) => (x * Math.PI) / 180;
    const dLat = toRad(b.lat - a.lat), dLng = toRad(b.lng - a.lng);
    const h = Math.sin(dLat / 2) ** 2 + Math.cos(toRad(a.lat)) * Math.cos(toRad(b.lat)) * Math.sin(dLng / 2) ** 2;
    return 2 * R * Math.asin(Math.sqrt(h));
  },

  bearing(a, b) {
    const toRad = (x) => (x * Math.PI) / 180, toDeg = (x) => (x * 180) / Math.PI;
    const y = Math.sin(toRad(b.lng - a.lng)) * Math.cos(toRad(b.lat));
    const x = Math.cos(toRad(a.lat)) * Math.sin(toRad(b.lat)) -
      Math.sin(toRad(a.lat)) * Math.cos(toRad(b.lat)) * Math.cos(toRad(b.lng - a.lng));
    return (toDeg(Math.atan2(y, x)) + 360) % 360;
  },
};

document.querySelectorAll("time[data-iso]").forEach((t) => { t.textContent = PW.fmtTime(t.dataset.iso); });
