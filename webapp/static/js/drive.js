(() => {
  const root = document.querySelector(".drive");
  const RADIUS = Number(root.dataset.radius) || 1000;
  const FETCH_RADIUS = RADIUS + 500;          // fetch a margin so new hazards appear before they matter
  const TIERS = [RADIUS, 300, 80];             // warn when entering each distance band
  const AHEAD_DEG = 50, BEHIND_DEG = 110;

  const alertEl = document.getElementById("alert");
  const distEl = document.getElementById("alert-distance");
  const textEl = document.getElementById("alert-text");
  const moreEl = document.getElementById("alert-more");
  const startBtn = document.getElementById("start-btn");
  const voiceEl = document.getElementById("voice");
  const followEl = document.getElementById("follow");
  const simEl = document.getElementById("sim");
  const gpsEl = document.getElementById("gps-status");

  const map = L.map("drive-map", { zoomControl: false }).setView(PW.DEFAULT_CENTER, 15);
  PW.tiles(map);
  const hazardLayer = L.layerGroup().addTo(map);
  const ring = L.circle(PW.DEFAULT_CENTER, { radius: RADIUS, color: "#1d4f9c", weight: 1.5, dashArray: "6 6", fillOpacity: 0.04 });
  const meIcon = L.divIcon({
    className: "",
    iconSize: [28, 28],
    iconAnchor: [14, 14],
    html: '<svg class="me-marker" viewBox="0 0 24 24" style="transform:rotate(0deg)"><circle cx="12" cy="12" r="9" fill="#1d4f9c" stroke="#fff" stroke-width="3"/><path d="M12 5l4 8.5h-8z" fill="#fff"/></svg>',
  });
  const me = L.marker(PW.DEFAULT_CENTER, { icon: meIcon, keyboard: false, interactive: false });

  let started = false, watchId = null, wakeLock = null;
  let pos = null, prev = null, heading = null;
  let hazards = [], lastFetch = { at: 0, pos: null }, fetching = false;
  const announced = new Map(); // code -> smallest tier already announced

  // ---------------------------------------------------------------- output channels
  function speak(text) {
    if (!voiceEl.checked || !("speechSynthesis" in window)) return;
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "en-IN";
    speechSynthesis.cancel();
    speechSynthesis.speak(u);
  }
  const buzz = () => navigator.vibrate?.([250, 120, 250]);
  function systemNotify(title, body, tag) {
    if (document.hidden && "Notification" in window && Notification.permission === "granted") {
      new Notification(title, { body, tag, renotify: true });
    }
  }
  const spokenDistance = (m) => (m >= 950 ? "1 kilometre" : `${Math.max(50, Math.round(m / 50) * 50)} metres`);
  const angleDiff = (a, b) => { const d = Math.abs(a - b) % 360; return d > 180 ? 360 - d : d; };
  const sevWord = (s) => s.charAt(0) + s.slice(1).toLowerCase();

  // ---------------------------------------------------------------- hazards
  function drawHazards() {
    hazardLayer.clearLayers();
    for (const h of hazards) {
      L.circleMarker([h.lat, h.lng], {
        radius: h.severity === "HIGH" ? 11 : 8, color: "#fff", weight: 3, fillColor: PW.sevColor[h.severity], fillOpacity: 1,
      }).bindPopup(`<strong>${PW.esc(sevWord(h.severity))}-severity ${PW.esc(h.label.toLowerCase())}</strong><br>${PW.esc(h.address)}<br>${PW.esc(PW.statusLabel[h.status])}${h.report_count > 1 ? `, ${h.report_count} reports` : ""}`)
        .addTo(hazardLayer);
    }
  }

  async function fetchHazards(center, radius = FETCH_RADIUS) {
    if (fetching) return;
    fetching = true;
    const data = await PW.api(`/api/hazards?lat=${center.lat}&lng=${center.lng}&radius=${radius}`);
    fetching = false;
    if (data.ok) {
      hazards = data.hazards;
      drawHazards();
    } else if (started) {
      gpsEl.textContent = data.error;
    }
  }

  function evaluate() {
    if (!pos) return;
    const list = hazards.map((h) => {
      const d = PW.haversine(pos, h), b = PW.bearing(pos, h);
      const diff = heading == null ? null : angleDiff(heading, b);
      return { ...h, d, ahead: diff != null && diff <= AHEAD_DEG, behind: diff != null && diff > BEHIND_DEG };
    }).filter((h) => (h.d <= RADIUS && !h.behind) || h.d <= 60).sort((a, b) => a.d - b.d);

    render(list);

    for (const h of list) {
      const tier = TIERS.filter((t) => h.d <= t).pop();
      if (tier !== undefined && tier < (announced.get(h.code) ?? Infinity)) {
        announced.set(h.code, tier);
        announce(h, tier);
        break; // one warning per update keeps the voice understandable
      }
    }
    for (const code of [...announced.keys()]) {
      const h = hazards.find((x) => x.code === code);
      if (!h || PW.haversine(pos, h) > RADIUS + 200) announced.delete(code); // passed it: warn again next time
    }
  }

  function announce(h, tier) {
    const what = h.type === "pothole" ? "pothole" : h.label.toLowerCase();
    const where = h.ahead ? "ahead" : "nearby";
    const text = tier <= 80
      ? `Caution. ${what} ${where}, ${spokenDistance(h.d)}. Slow down now.`
      : `Caution. ${sevWord(h.severity)} severity ${what} ${where} in ${spokenDistance(h.d)}. Drive carefully.`;
    speak(text);
    buzz();
    systemNotify(`${what.charAt(0).toUpperCase() + what.slice(1)} ${where}: ${PW.fmtDist(h.d)}`, h.address, h.code);
  }

  function render(list) {
    alertEl.classList.remove("alert--idle", "alert--clear", "alert--near", "alert--danger");
    if (!list.length) {
      alertEl.classList.add("alert--clear");
      distEl.textContent = "Clear";
      textEl.textContent = `No reported potholes within ${PW.fmtDist(RADIUS)}.`;
      moreEl.textContent = heading == null ? "Direction unknown until you start moving." : "";
      return;
    }
    const h = list[0];
    if (h.d <= 80) alertEl.classList.add("alert--danger");
    else if (h.d <= 300) alertEl.classList.add("alert--near");
    distEl.textContent = PW.fmtDist(h.d);
    textEl.textContent = `${sevWord(h.severity)}-severity ${h.label.toLowerCase()} ${h.ahead ? "ahead" : "nearby"}. Drive carefully.`;
    moreEl.textContent = list.length > 1
      ? `${h.address}. ${list.length - 1} more within ${PW.fmtDist(RADIUS)}.`
      : h.address;
  }

  // ---------------------------------------------------------------- position
  function onPosition(p) {
    const cur = { lat: p.coords.latitude, lng: p.coords.longitude };
    const gpsHeading = p.coords.heading;
    if (gpsHeading != null && !Number.isNaN(gpsHeading) && (p.coords.speed ?? 0) > 1.5) {
      heading = gpsHeading;
    }
    if (!prev) prev = cur;
    else if (PW.haversine(prev, cur) > 8) {
      if (gpsHeading == null || Number.isNaN(gpsHeading) || (p.coords.speed ?? 0) <= 1.5) heading = PW.bearing(prev, cur);
      prev = cur;
    }
    pos = cur;

    me.setLatLng(cur).addTo(map);
    ring.setLatLng(cur).addTo(map);
    const svg = me.getElement()?.querySelector("svg");
    if (svg) svg.style.transform = `rotate(${heading ?? 0}deg)`;
    if (followEl.checked) map.panTo(cur, { animate: true });

    const parts = [simEl.checked ? "Simulated position" : `GPS accurate to ${Math.round(p.coords.accuracy)} m`];
    if (p.coords.speed != null && p.coords.speed > 0.5) parts.push(`${Math.round(p.coords.speed * 3.6)} km/h`);
    gpsEl.textContent = parts.join(", ");

    const moved = lastFetch.pos ? PW.haversine(lastFetch.pos, pos) : Infinity;
    if (moved > 150 || Date.now() - lastFetch.at > 30000) {
      lastFetch = { at: Date.now(), pos: { ...pos } };
      fetchHazards(pos).then(evaluate);
    } else {
      evaluate();
    }
  }

  function onGpsError(err) {
    gpsEl.textContent = err.code === 1
      ? "Location access is blocked. Allow it in the browser, or use Simulate."
      : "Waiting for a GPS fix… Move to open sky if this takes long.";
  }

  function startGps() {
    if (!("geolocation" in navigator)) { gpsEl.textContent = "This browser has no GPS access. Use Simulate."; return; }
    if (!window.isSecureContext) { gpsEl.textContent = "GPS needs an https address. Use Simulate, or open the site over https."; return; }
    gpsEl.textContent = "Waiting for GPS…";
    watchId = navigator.geolocation.watchPosition(onPosition, onGpsError, { enableHighAccuracy: true, maximumAge: 2000, timeout: 20000 });
  }
  function stopGps() {
    if (watchId != null) navigator.geolocation.clearWatch(watchId);
    watchId = null;
  }

  async function keepAwake() {
    try { wakeLock = await navigator.wakeLock?.request("screen"); } catch { wakeLock = null; }
  }
  document.addEventListener("visibilitychange", () => { if (started && document.visibilityState === "visible") keepAwake(); });

  function start() {
    started = true;
    startBtn.textContent = "Stop drive mode";
    speak("Drive mode on.");
    if ("Notification" in window && Notification.permission === "default") Notification.requestPermission();
    keepAwake();
    textEl.textContent = "Looking for reported potholes near you…";
    if (simEl.checked) gpsEl.textContent = "Simulating: tap the map to move.";
    else startGps();
  }
  function stop() {
    started = false;
    stopGps();
    wakeLock?.release?.();
    window.speechSynthesis?.cancel?.();
    startBtn.textContent = "Start drive mode";
    alertEl.className = "alert alert--idle";
    distEl.textContent = PW.fmtDist(RADIUS);
    textEl.textContent = `Start drive mode to be warned about reported potholes within ${PW.fmtDist(RADIUS)}.`;
    moreEl.textContent = "";
    gpsEl.textContent = "GPS off";
    announced.clear();
    pos = prev = heading = null;
  }
  startBtn.addEventListener("click", () => (started ? stop() : start()));

  simEl.addEventListener("change", () => {
    if (!started) return;
    if (simEl.checked) { stopGps(); gpsEl.textContent = "Simulating: tap the map to move."; }
    else startGps();
  });
  map.on("click", (e) => {
    if (!simEl.checked) return;
    if (!started) start();
    onPosition({ coords: { latitude: e.latlng.lat, longitude: e.latlng.lng, accuracy: 5, heading: null, speed: null } });
  });

  // before starting, show reported potholes around whatever part of the map is on screen
  map.on("moveend", () => {
    if (started) return;
    const c = map.getCenter();
    fetchHazards({ lat: c.lat, lng: c.lng }, 5000);
  });
  fetchHazards({ lat: PW.DEFAULT_CENTER[0], lng: PW.DEFAULT_CENTER[1] }, 5000);
})();
