(() => {
  const latEl = document.getElementById("zone_lat");
  const lngEl = document.getElementById("zone_lng");
  const radEl = document.getElementById("zone_radius_km");
  const zones = JSON.parse(document.getElementById("zone-data")?.textContent || "[]");

  const map = L.map("zone-map", { scrollWheelZoom: false }).setView(PW.DEFAULT_CENTER, 12);
  PW.tiles(map);

  // existing zones, so a new one can be placed next to them
  const existing = zones.map((z) => L.circle([z.lat, z.lng], {
    radius: z.radius_km * 1000, color: "#5a645f", weight: 1, dashArray: "4 4", fillOpacity: z.active ? 0.06 : 0.02,
  }).bindTooltip(PW.esc(z.name)).addTo(map));
  if (existing.length) map.fitBounds(L.featureGroup(existing).getBounds().pad(0.1));

  let circle = null;
  function draw() {
    const lat = parseFloat(latEl.value), lng = parseFloat(lngEl.value);
    if (Number.isNaN(lat) || Number.isNaN(lng)) return;
    if (!radEl.value) radEl.value = "3";
    const r = (parseFloat(radEl.value) || 3) * 1000;
    if (!circle) {
      circle = L.circle([lat, lng], { radius: r, color: "#1d4f9c", weight: 2, fillOpacity: 0.12 }).addTo(map);
    } else {
      circle.setLatLng([lat, lng]).setRadius(r);
    }
  }

  map.on("click", (e) => {
    latEl.value = e.latlng.lat.toFixed(6);
    lngEl.value = e.latlng.lng.toFixed(6);
    draw();
  });
  radEl.addEventListener("input", draw);
})();
