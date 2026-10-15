(() => {
  const listEl = document.getElementById("report-list");
  const detail = document.getElementById("detail");
  const detailBody = document.getElementById("detail-body");
  const sevSel = document.getElementById("f-sev");
  const qInput = document.getElementById("f-q");
  const segButtons = [...document.querySelectorAll(".seg button")];

  const state = { status: "ACTIVE", severity: "", q: "", reports: [], selected: null, knownIds: null };
  const map = L.map("dash-map").setView(PW.DEFAULT_CENTER, 13);
  PW.tiles(map);
  const layer = L.layerGroup().addTo(map);
  const markers = new Map();
  let fitted = false;

  const sevWord = (s) => s.charAt(0) + s.slice(1).toLowerCase();
  const label = (r) => `${sevWord(r.severity)}-severity ${r.primary_label.toLowerCase()}`;

  // ---------------------------------------------------------------- list + map
  async function load({ quiet = false } = {}) {
    const params = new URLSearchParams({ status: state.status, severity: state.severity, q: state.q });
    const data = await PW.api(`/api/admin/reports?${params}`);
    if (!data.ok) {
      if (!quiet) listEl.innerHTML = `<li class="empty-row">${PW.esc(data.error)}</li>`;
      return;
    }
    state.reports = data.reports;
    const c = data.counts || {};
    document.querySelector('[data-count="ACTIVE"]').textContent = (c.OPEN || 0) + (c.IN_PROGRESS || 0);
    document.querySelector('[data-count="IN_PROGRESS"]').textContent = c.IN_PROGRESS || 0;
    document.querySelector('[data-count="RESOLVED"]').textContent = c.RESOLVED || 0;
    detectNew(data.reports);
    renderList();
    renderMap();
  }

  function renderList() {
    if (!state.reports.length) {
      listEl.innerHTML = `<li class="empty-row">${state.status === "ACTIVE"
        ? "Nothing to do. New reports appear here as soon as citizens send them."
        : "No reports match these filters."}</li>`;
      return;
    }
    listEl.innerHTML = state.reports.map((r) => `
      <li><button type="button" data-id="${r.id}" ${state.selected === r.id ? 'aria-current="true"' : ""}>
        <span class="bar" style="background:${PW.sevColor[r.severity]}"></span>
        <span class="r-title">${PW.esc(label(r))}</span>
        <span class="r-status">${PW.esc(PW.statusLabel[r.status])}</span>
        <span class="r-addr">${PW.esc(r.address)}</span>
        <span class="r-meta">${PW.esc(r.code)} | ${PW.esc(PW.fmtTime(r.created_at))}${r.report_count > 1 ? ` | ${r.report_count} reports` : ""}</span>
      </button></li>`).join("");
  }

  function renderMap() {
    layer.clearLayers();
    markers.clear();
    for (const r of state.reports) {
      const done = r.status === "RESOLVED" || r.status === "REJECTED";
      const m = L.circleMarker([r.lat, r.lng], {
        radius: r.severity === "HIGH" ? 10 : 8,
        color: done ? PW.sevColor[r.severity] : "#fff",
        weight: 3,
        fillColor: PW.sevColor[r.severity],
        fillOpacity: done ? 0.15 : 1,
      }).bindTooltip(`${PW.esc(label(r))}<br>${PW.esc(r.address)}`).on("click", () => openDetail(r.id));
      m.addTo(layer);
      markers.set(r.id, m);
    }
    if (!fitted && state.reports.length) {
      map.fitBounds(L.latLngBounds(state.reports.map((r) => [r.lat, r.lng])).pad(0.2), { maxZoom: 16 });
      fitted = true;
    }
  }

  listEl.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-id]");
    if (b) openDetail(Number(b.dataset.id));
  });

  segButtons.forEach((b) => b.addEventListener("click", () => {
    segButtons.forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    state.status = b.dataset.status;
    fitted = false;
    load();
  }));
  sevSel.addEventListener("change", () => { state.severity = sevSel.value; load(); });
  let qTimer;
  qInput.addEventListener("input", () => {
    clearTimeout(qTimer);
    qTimer = setTimeout(() => { state.q = qInput.value.trim(); load(); }, 300);
  });

  // ---------------------------------------------------------------- new-report alerts for staff
  function detectNew(reports) {
    if (state.status !== "ACTIVE") return;
    const ids = new Set(reports.map((r) => r.id));
    if (state.knownIds) {
      const fresh = reports.filter((r) => !state.knownIds.has(r.id));
      for (const r of fresh) {
        if ("Notification" in window && Notification.permission === "granted") {
          new Notification(`New ${label(r)}`, { body: r.address, tag: r.code });
        }
      }
      if (fresh.length) document.title = `(${fresh.length} new) Reports · PotholeWatch`;
    }
    state.knownIds = ids;
  }
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") document.title = "Reports · PotholeWatch";
  });
  if ("Notification" in window && Notification.permission === "default") {
    const ask = document.createElement("button");
    ask.type = "button";
    ask.className = "btn btn--small";
    ask.textContent = "Alert me about new reports";
    ask.addEventListener("click", async () => { await Notification.requestPermission(); ask.remove(); });
    document.querySelector(".filters").appendChild(ask);
  }

  // ---------------------------------------------------------------- detail panel
  const NEXT = {
    OPEN: [["IN_PROGRESS", "Start repair", "btn--primary"], ["RESOLVED", "Mark fixed", ""], ["REJECTED", "Close without repair", "btn--danger"]],
    IN_PROGRESS: [["RESOLVED", "Mark fixed", "btn--primary"], ["OPEN", "Move back to open", ""], ["REJECTED", "Close without repair", "btn--danger"]],
    RESOLVED: [["OPEN", "Reopen", ""]],
    REJECTED: [["OPEN", "Reopen", ""]],
  };

  async function openDetail(id) {
    state.selected = id;
    renderList();
    const data = await PW.api(`/api/admin/reports/${id}`);
    if (!data.ok) { alert(data.error); return; }
    const r = data.report;
    const m = markers.get(id);
    if (m) map.setView(m.getLatLng(), Math.max(map.getZoom(), 16));
    history.replaceState(null, "", `?report=${encodeURIComponent(r.code)}`);
    detailBody.innerHTML = renderDetail(r);
    detail.hidden = false;
    wireActions(r);
  }

  function renderDetail(r) {
    const dets = (r.detections || []).map((d) =>
      `<li><span>${PW.esc(d.label)}</span><span class="sev sev--${d.severity.toLowerCase()}">${PW.esc(sevWord(d.severity))}, ${Math.round(d.confidence * 100)}%</span></li>`).join("");
    const events = (r.events || []).map((e) =>
      `<li><span>${PW.esc(PW.statusLabel[e.status] || e.status)}${e.note ? `: ${PW.esc(e.note)}` : ""}</span><span>${PW.esc(PW.fmtTime(e.created_at))}</span></li>`).join("");
    const msgs = (r.notifications || []).map((n) =>
      `<li><span>${n.channel === "email" ? "Email" : "SMS"} to ${PW.esc(n.audience)} (${PW.esc(n.recipient)})</span><span>${n.status === "sent" ? "Sent" : n.status === "failed" ? "Failed" : "Not set up"}</span></li>`).join("");
    const actions = (NEXT[r.status] || []).map(([s, text, cls]) =>
      `<button type="button" class="btn btn--small ${cls}" data-status="${s}">${text}</button>`).join("");
    const where = `https://www.google.com/maps?q=${r.lat},${r.lng}`;
    const locNote = r.location_source === "photo" ? "from photo"
      : r.location_source === "manual" ? "placed on map"
      : r.location_source === "device" ? "pothole box GPS"
      : r.accuracy_m ? `GPS ±${Math.round(r.accuracy_m)} m` : "GPS";
    const foundBy = r.source === "device"
      ? `Pothole box ${PW.esc(r.device_id || "")}${r.jolt_g ? `, jolt ${Number(r.jolt_g).toFixed(2)} g` : ""}${r.speed_kmh != null ? ` at ${Math.round(r.speed_kmh)} km/h` : ""}`
      : "Citizen report";
    return `
      <p class="sev-badge ${r.severity.toLowerCase()}">${PW.esc(sevWord(r.severity))} severity, score ${Number(r.severity_score).toFixed(2)}</p>
      <h2>${PW.esc(r.primary_label)} at ${PW.esc(r.address)}</h2>
      <dl>
        <dt>Code</dt><dd>${PW.esc(r.code)}</dd>
        <dt>Status</dt><dd>${PW.esc(PW.statusLabel[r.status])}</dd>
        <dt>Reported</dt><dd>${PW.esc(PW.fmtTime(r.created_at))}</dd>
        <dt>Found by</dt><dd>${foundBy}</dd>
        <dt>Reports</dt><dd>${r.report_count} (${r.subscriber_count} want updates)</dd>
        <dt>Assigned to</dt><dd>${PW.esc(r.authority ? r.authority.name : "No contact covers this area")}</dd>
        <dt>Location</dt><dd><a href="${where}" target="_blank" rel="noopener">${r.lat.toFixed(5)}, ${r.lng.toFixed(5)}</a>
          <span class="muted">${locNote}</span></dd>
      </dl>
      <div class="photos">
        <a href="${r.annotated_url}" target="_blank" rel="noopener"><img src="${r.annotated_url}" alt="Reported photo with detections"></a>
        ${r.resolution_image_url ? `<a href="${r.resolution_image_url}" target="_blank" rel="noopener"><img src="${r.resolution_image_url}" alt="Photo after repair"></a>` : ""}
      </div>
      ${dets ? `<h3>Detections</h3><ul class="det-list">${dets}</ul>` : ""}
      ${r.resolution_note ? `<p class="result-note">Note: ${PW.esc(r.resolution_note)}</p>` : ""}
      <form class="actions" id="status-form">
        <h3>Update status</h3>
        <label class="field"><span class="field-label">Note for the reporter <span class="optional">sent with the message</span></span>
          <textarea name="note" rows="3" maxlength="500" placeholder="Filled with hot-mix asphalt on 12 Oct."></textarea></label>
        <label class="field"><span class="field-label">Photo after repair <span class="optional">optional</span></span>
          <input type="file" name="after_photo" accept="image/*" capture="environment"></label>
        <div class="btn-row">${actions}</div>
        <p class="hint" id="status-msg" role="status"></p>
      </form>
      <h3>History</h3><ul class="msg-list">${events}</ul>
      ${msgs ? `<h3>Messages</h3><ul class="msg-list">${msgs}</ul>` : ""}`;
  }

  function wireActions(r) {
    const form = document.getElementById("status-form");
    const msg = document.getElementById("status-msg");
    form.addEventListener("click", async (e) => {
      const b = e.target.closest("button[data-status]");
      if (!b) return;
      const fd = new FormData(form);
      fd.set("status", b.dataset.status);
      form.querySelectorAll("button").forEach((x) => { x.disabled = true; });
      msg.textContent = "Saving…";
      const data = await PW.api(`/api/admin/reports/${r.id}/status`, { method: "POST", body: fd });
      form.querySelectorAll("button").forEach((x) => { x.disabled = false; });
      if (!data.ok) { msg.textContent = data.error; return; }
      await load({ quiet: true });
      await openDetail(r.id);
      const m2 = document.getElementById("status-msg");
      const who = data.notified
        ? `${data.notified} message${data.notified > 1 ? "s" : ""} to the reporter; delivery results are on the Notifications page.`
        : "Nobody left contact details, so no message was sent.";
      if (m2) m2.textContent = `Saved as ${PW.statusLabel[data.report.status].toLowerCase()}. ${who}`;
    });
  }

  document.getElementById("detail-close").addEventListener("click", () => {
    detail.hidden = true;
    state.selected = null;
    history.replaceState(null, "", location.pathname);
    renderList();
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !detail.hidden) document.getElementById("detail-close").click(); });

  // ---------------------------------------------------------------- start
  (async () => {
    await load();
    const code = new URLSearchParams(location.search).get("report");
    if (code) {
      let r = state.reports.find((x) => x.code === code.toUpperCase());
      if (!r) { // link to a fixed or closed report: widen the filter
        segButtons.forEach((x) => x.setAttribute("aria-pressed", String(x.dataset.status === "")));
        state.status = "";
        await load();
        r = state.reports.find((x) => x.code === code.toUpperCase());
      }
      if (r) openDetail(r.id);
    }
    setInterval(() => load({ quiet: true }), 30000);
  })();
})();
