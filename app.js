(() => {
  function dms(value, axis) {
    const abs = Math.abs(Number(value) || 0);
    const deg = Math.floor(abs);
    const minsFloat = (abs - deg) * 60;
    const mins = Math.floor(minsFloat);
    const secs = (minsFloat - mins) * 60;
    const suffix = axis === 'lat' ? (value >= 0 ? 'N' : 'S') : (value >= 0 ? 'E' : 'W');
    return `${deg}° ${mins}' ${secs.toFixed(2)}\" ${suffix}`;
  }

  function coordText(lat, lon) {
    return `${Number(lat).toFixed(4)}, ${Number(lon).toFixed(4)}`;
  }

  function bearingDeg(lat1, lon1, lat2, lon2) {
    const toRad = (deg) => deg * Math.PI / 180;
    const toDeg = (rad) => rad * 180 / Math.PI;
    const y = Math.sin(toRad(lon2 - lon1)) * Math.cos(toRad(lat2));
    const x = Math.cos(toRad(lat1)) * Math.sin(toRad(lat2)) - Math.sin(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.cos(toRad(lon2 - lon1));
    return (toDeg(Math.atan2(y, x)) + 360) % 360;
  }

  function normalizeThresholdSelect() {
    const select = document.getElementById('clusterThreshold');
    if (!select) return;
    const allowed = ['0','5','10','15','20','25','30','35','40','45','50','55','60','65','70','75','80','85','90','95','100'];
    const existing = Array.from(select.options).map(o => String(o.value));
    if (existing.length !== allowed.length || existing.some((v, i) => v !== allowed[i])) {
      select.innerHTML = allowed.map(v => `<option value="${v}">≥ ${v}%</option>`).join('');
    }
    if (!allowed.includes(String(select.value))) select.value = '30';
    else select.value = String(select.value);
  }

  function initPhase1() {
    const mapRef = window.map || window.__map || (window.L && typeof window.L.map === 'function' ? window.L.map('map') : null);
    if (mapRef) {
      window.map = mapRef;
      window.__map = mapRef;
    }

    const markerLayer = window.__phase1MarkerLayer || (mapRef && window.L && window.L.layerGroup ? window.L.layerGroup().addTo(mapRef) : null);
    if (mapRef && markerLayer && !window.__phase1MarkerLayer) window.__phase1MarkerLayer = markerLayer;

    const phase1State = window.__phase1State || { selectedMarker: null, customMarkers: [] };
    window.__phase1State = phase1State;

    function updateReadout(lat, lon) {
      const valueEl = document.getElementById('coordValue');
      const dmsEl = document.getElementById('coordDms');
      const bearingEl = document.getElementById('coordBearing');
      const localTimeEl = document.getElementById('coordLocalTime');
      if (valueEl) valueEl.textContent = coordText(lat, lon);
      if (dmsEl) dmsEl.textContent = `${dms(lat, 'lat')} / ${dms(lon, 'lon')}`;
      if (bearingEl) {
        const center = mapRef ? mapRef.getCenter() : { lat, lng: lon };
        const baseLat = center.lat ?? lat;
        const baseLon = center.lng ?? lon;
        bearingEl.textContent = `${Math.round(bearingDeg(baseLat, baseLon, lat, lon))}°`;
      }
      if (localTimeEl) {
        const now = new Date();
        const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';
        localTimeEl.textContent = `${now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} (${tz})`;
      }
    }

    function addMarker(lat, lon, label = 'Map pin') {
      if (!mapRef || !markerLayer || !window.L) return null;
      const marker = window.L.marker([lat, lon]).addTo(markerLayer);
      marker.bindPopup(`<strong>${label}</strong><br>${coordText(lat, lon)}`);
      marker.on('click', () => {
        phase1State.selectedMarker = marker;
        marker.openPopup();
        updateReadout(lat, lon);
      });
      phase1State.customMarkers.push(marker);
      return marker;
    }

    function clearMarkers() {
      if (markerLayer) markerLayer.clearLayers();
      phase1State.customMarkers = [];
      phase1State.selectedMarker = null;
    }

    function removeSelectedMarker() {
      if (!phase1State.selectedMarker || !markerLayer) return;
      markerLayer.removeLayer(phase1State.selectedMarker);
      phase1State.customMarkers = phase1State.customMarkers.filter(item => item !== phase1State.selectedMarker);
      phase1State.selectedMarker = null;
    }

    function focusLocation(lat, lon, zoom = 8) {
      if (!mapRef) return;
      mapRef.flyTo([lat, lon], zoom, { duration: 1.2 });
      updateReadout(lat, lon);
    }

    const gotoBtn = document.getElementById('gotoCoordBtn');
    const gotoLatInput = document.getElementById('gotoLatInput');
    const gotoLonInput = document.getElementById('gotoLonInput');
    const clearPinsBtn = document.getElementById('clearPinsBtn');
    const deleteSelectedBtn = document.getElementById('deleteSelectedPinBtn');
    const centerSelectedBtn = document.getElementById('centerSelectedPinBtn');

    if (gotoBtn && gotoLatInput && gotoLonInput) {
      gotoBtn.addEventListener('click', () => {
        const lat = Number(gotoLatInput.value);
        const lon = Number(gotoLonInput.value);
        if (Number.isFinite(lat) && Number.isFinite(lon)) {
          focusLocation(lat, lon, 8);
          addMarker(lat, lon, 'Manual location');
        }
      });
    }

    if (clearPinsBtn) clearPinsBtn.addEventListener('click', clearMarkers);
    if (deleteSelectedBtn) deleteSelectedBtn.addEventListener('click', removeSelectedMarker);
    if (centerSelectedBtn && phase1State.selectedMarker && mapRef) {
      centerSelectedBtn.addEventListener('click', () => {
        const latLng = phase1State.selectedMarker.getLatLng();
        if (latLng) focusLocation(latLng.lat, latLng.lng, 9);
      });
    }

    if (mapRef) {
      mapRef.on('click', (event) => {
        const { lat, lng } = event.latlng;
        updateReadout(lat, lng);
        const pinToggle = document.getElementById('pinToggle');
        if (pinToggle && pinToggle.checked) {
          const marker = addMarker(lat, lng, 'Map pin');
          if (marker) phase1State.selectedMarker = marker;
        }
      });

      mapRef.on('move', () => {
        const c = mapRef.getCenter();
        updateReadout(c.lat, c.lng);
      });
    }

    if (centerSelectedBtn) {
      centerSelectedBtn.addEventListener('click', () => {
        if (!phase1State.selectedMarker || !mapRef) return;
        const latLng = phase1State.selectedMarker.getLatLng();
        if (latLng) focusLocation(latLng.lat, latLng.lng, 9);
      });
    }

    normalizeThresholdSelect();
    if (mapRef) {
      const center = mapRef.getCenter();
      updateReadout(center.lat, center.lng);
    }

    window.__phase1 = {
      addMarker,
      clearMarkers,
      removeSelectedMarker,
      focusLocation,
      updateReadout,
      getMap: () => mapRef,
      getMarkerLayer: () => markerLayer,
    };
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initPhase1, { once: true });
  else initPhase1();
})();
