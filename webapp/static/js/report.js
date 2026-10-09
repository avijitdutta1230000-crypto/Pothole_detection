(() => {
  const form = document.getElementById("report-form");
  const photo = document.getElementById("photo");
  const preview = document.getElementById("preview");
  const dzText = document.getElementById("dz-text");
  const dropzone = document.getElementById("dropzone");
  const locStatus = document.getElementById("loc-status");
  const latEl = document.getElementById("lat"), lngEl = document.getElementById("lng");
  const accEl = document.getElementById("accuracy"), srcEl = document.getElementById("location_source");
  const submitBtn = document.getElementById("submit-btn");
  const errorEl = document.getElementById("form-error");
  const result = document.getElementById("result");

  // ---------------------------------------------------------------- photo
  function showPreview(file) {
    if (!file) return;
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
    dzText.hidden = true;
  }
  photo.addEventListener("change", () => showPreview(photo.files[0]));
  ["dragenter", "dragover"].forEach((e) => dropzone.addEventListener(e, (ev) => { ev.preventDefault(); dropzone.classList.add("is-over"); }));
  ["dragleave", "drop"].forEach((e) => dropzone.addEventListener(e, () => dropzone.classList.remove("is-over")));
  dropzone.addEventListener("drop", (ev) => {
    ev.preventDefault();
    if (ev.dataTransfer.files.length) {
      const dt = new DataTransfer();
      dt.items.add(ev.dataTransfer.files[0]);
      photo.files = dt.files;
      showPreview(photo.files[0]);
    }
  });

  // ---------------------------------------------------------------- location
  const map = L.map("pick-map", { scrollWheelZoom: false }).setView(PW.DEFAULT_CENTER, 13);
  PW.tiles(map);
  let marker = null;

  function setPoint(lat, lng, source, accuracy) {
    latEl.value = lat.toFixed(7);
    lngEl.value = lng.toFixed(7);
    srcEl.value = source;
    accEl.value = accuracy ? Math.round(accuracy) : "";
    if (!marker) {
      marker = L.marker([lat, lng], { draggable: true, autoPan: true }).addTo(map);
      marker.on("dragend", () => {
        const p = marker.getLatLng();
        setPoint(p.lat, p.lng, "manual");
      });
    } else {
      marker.setLatLng([lat, lng]);
    }
    if (source === "manual") {
      locStatus.textContent = "Pin placed on the map.";
      locStatus.className = "loc-status is-good";
    }
  }

  map.on("click", (e) => setPoint(e.latlng.lat, e.latlng.lng, "manual"));

  function askLocation() {
    if (!("geolocation" in navigator)) {
      locStatus.textContent = "This browser can't share location. Tap the map to place the pothole.";
      locStatus.className = "loc-status is-bad";
      return;
    }
    if (!window.isSecureContext) {
      locStatus.textContent = "Location only works on https pages. Tap the map to place the pothole.";
      locStatus.className = "loc-status is-bad";
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (p) => {
        if (srcEl.value === "manual" && latEl.value) return; // user already placed it
        setPoint(p.coords.latitude, p.coords.longitude, "gps", p.coords.accuracy);
        map.setView([p.coords.latitude, p.coords.longitude], 18);
        locStatus.textContent = `Location found, accurate to about ${Math.round(p.coords.accuracy)} m.`;
        locStatus.className = "loc-status is-good";
      },
      (err) => {
        locStatus.textContent = err.code === 1
          ? "Location access is blocked. Tap the map to place the pothole, or allow location and reload."
          : "Couldn't get a GPS fix. Tap the map to place the pothole.";
        locStatus.className = "loc-status is-bad";
      },
      { enableHighAccuracy: true, timeout: 15000, maximumAge: 10000 },
    );
  }
  askLocation();

  // ---------------------------------------------------------------- submit
  function showError(msg) {
    errorEl.textContent = msg;
    errorEl.hidden = false;
  }

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    errorEl.hidden = true;
    if (!photo.files.length) return showError("Add a photo of the road damage.");
    submitBtn.disabled = true;
    submitBtn.textContent = "Checking photo…";
    const data = await PW.api("/api/reports", { method: "POST", body: new FormData(form) });
    submitBtn.disabled = false;
    submitBtn.textContent = "Send report";
    if (!data.ok) return showError(data.error);
    if (data.status === "no_damage") return showError(data.message);
    showResult(data);
  });

  function showResult(d) {
    const sev = d.severity.toLowerCase();
    const top = d.detections[0];
    document.getElementById("result-img").src = d.annotated_url;
    const badge = document.getElementById("result-sev");
    badge.className = `sev-badge ${sev}`;
    badge.textContent = `${d.severity.charAt(0) + sev.slice(1)} severity (score ${d.severity_score.toFixed(2)})`;
    document.getElementById("result-title").textContent = d.status === "merged"
      ? `Already reported. Yours is report number ${d.report_count}`
      : `${top.label} reported`;
    document.getElementById("result-address").textContent = d.status === "merged"
      ? `${d.address}. The road team already has this one; we added your report and will message you too.`
      : `${d.address}${d.location_source === "photo" ? " (location taken from the photo)" : ""}`;
    document.getElementById("result-dets").innerHTML = d.detections.map((x) =>
      `<li><span>${PW.esc(x.label)}</span><span class="sev sev--${x.severity.toLowerCase()}">${PW.esc(x.severity.charAt(0) + x.severity.slice(1).toLowerCase())}, ${Math.round(x.confidence * 100)}% sure</span></li>`).join("");
    document.getElementById("result-code").textContent = d.code;
    document.getElementById("result-track").href = d.track_url;
    form.hidden = true;
    result.hidden = false;
    result.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  document.getElementById("report-another").addEventListener("click", () => {
    form.reset();
    preview.hidden = true;
    dzText.hidden = false;
    result.hidden = true;
    form.hidden = false;
    errorEl.hidden = true;
    if (latEl.value) srcEl.value = "gps";
    askLocation();
    window.scrollTo({ top: 0 });
  });
})();
