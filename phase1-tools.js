(() => {
  function initPhase1() {
    const map = window.map || window.__map || (window.L ? L.map('map') : null);
    if (map) {
      window.map = map;
      window.__map = map;
    }

    const markerLayer = window.__phase1MarkerLayer || (window.L ? L.layerGroup().addTo(map) : null);
    if (map && markerLayer && !window.__phase1MarkerLayer) {
      window.__phase1MarkerLayer = markerLayer;
    }

    const state = {
      selectedMarker: null,
      customMarkers: [],
      lastCenter: map ? map.getCenter() : null,
    };

    function parseCoordString(raw) {
      const text = String(raw || '').trim();
      if (!text) return null;
      const compact = text.replace(/[^0-9.,\-\s]/g, ' ');
      const parts = compact.split(/\s+/).map(v => v.trim()).filter(Boolean);
      if (parts.length >= 2) {
        const lat = Number(parts[0]);
        const lon = Number(parts[1]);
        if (Number.isFinite(lat) && Number.isFinite(lon)) return { lat, lon };
      }
      const asCsv = text.split(',').map(v => Number(v.trim())).filter(v => Number.isFinite(v));
      if (asCsv.length >= 2) return { lat: asCsv[0], lon: asCsv[1] };
      return null;
    }

    function toDMS(value, type) {
      const abs = Math.abs(value);
      const deg = Math.floor(abs);
      const minFloat = (abs - deg) * 60;
      const min = Math.floor(minFloat);
      const sec = (minFloat - min) * 60;
      const suffix = type === 'lat' ? (value >= 0 ? 'N' : 'S') : (value >= 0 ? 'E' : 'W');
      return `${deg}° ${min}' ${sec.toFixed(2)}\" ${suffix}`;
    }

    function formatCoord(lat, lon) {
      return `${Number(lat).toFixed(4)}, ${Number(lon).toFixed(4)}`;
    }

    function updateReadout(lat, lon) {
      const latEl = document.getElementById('coordValue');
      const dmsEl = document.getElementById('coordDms');
      const bearingEl = document.getElementById('coordBearing');
      const localTimeEl = document.getElementById('coordLocalTime');
      if (latEl) latEl.textContent = formatCoord(lat, lon);
      if (dmsEl) dmsEl.textContent = `${toDMS(lat, 'lat')} / ${toDMS(lon, 'lon')}`;
      if (bearingEl) {
        const center = map ? map.getCenter() : { lat, lng: lon };
        const fromLat = center.lat || lat;
        const fromLon = center.lng || lon;
        const bearing = calculateBearing(fromLat, fromLon, lat, lon);
        bearingEl.textContent = `${bearing}°`;
      }
      if (localTimeEl) {
        const date = new Date();
        const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';
        localTimeEl.textContent = `${date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} (${tz})`;
      }
    }

    function calculateBearing(lat1, lon1, lat2, lon2) {
      const toRad = (deg) => (deg * Math.PI) / 180;
      const toDeg = (rad) => (rad * 180) / Math.PI;
      const y = Math.sin(toRad(lon2 - lon1)) * Math.cos(toRad(lat2));
      const x = Math.cos(toRad(lat1)) * Math.sin(toRad(lat2)) - Math.sin(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.cos(toRad(lon2 - lon1));
      return (toDeg(Math.atan2(y, x)) + 360) % 360;
    }

    function addMarker(lat, lon, label = 'Custom pin') {
      if (!map || !markerLayer || !window.L) return null;
      const marker = L.marker([lat, lon]).addTo(markerLayer);
      marker.bindPopup(`<strong>${label}</strong><br>${formatCoord(lat, lon)}`);
      marker.on('click', () => {
        state.selectedMarker = marker;
        marker.openPopup();
        updateReadout(lat, lon);
      });
      state.customMarkers.push(marker);
      return marker;
    }

    function clearMarkers() {
      if (markerLayer) markerLayer.clearLayers();
      state.customMarkers = [];
      state.selectedMarker = null;
    }

    function removeSelectedMarker() {
      if (!state.selectedMarker || !markerLayer) return;
      markerLayer.removeLayer(state.selectedMarker);
      state.customMarkers = state.customMarkers.filter(item => item !== state.selectedMarker);
      state.selectedMarker = null;
    }

    function focusLocation(lat, lon, zoom = 8) {
      if (!map) return;
      map.flyTo([lat, lon], zoom, { duration: 1.2 });
      updateReadout(lat, lon);
    }

    function installThresholdControls() {
      const thresholdSelect = document.getElementById('clusterThreshold');
      if (!thresholdSelect || document.getElementById('phase1ClusterSlider')) return;

      const wrap = document.createElement('div');
      wrap.className = 'phase1-cluster-wrap';

      const inline = document.createElement('div');
      inline.className = 'phase1-cluster-inline';

      const slider = document.createElement('input');
      slider.type = 'range';
      slider.id = 'phase1ClusterSlider';
      slider.min = '0';
      slider.max = '100';
      slider.step = '5';
      slider.value = String(thresholdSelect.value || '30');
      inline.appendChild(slider);

      const number = document.createElement('input');
      number.type = 'number';
      number.id = 'phase1ClusterNumber';
      number.min = '0';
      number.max = '100';
      number.step = '5';
      number.value = String(thresholdSelect.value || '30');
      inline.appendChild(number);

      const label = document.createElement('span');
      label.className = 'muted small';
      label.textContent = 'Cutoff mode';
      wrap.appendChild(inline);
      wrap.appendChild(label);

      thresholdSelect.parentNode.insertBefore(wrap, thresholdSelect.nextSibling);

      const syncFromSelect = () => {
        const value = thresholdSelect.value || '30';
        slider.value = value;
        number.value = value;
      };

      const syncFromControls = (value) => {
        const normalized = Math.min(100, Math.max(0, Number(value) || 30));
        const rounded = Math.round(normalized / 5) * 5;
        const finalValue = String(Math.min(100, Math.max(0, rounded)));
        thresholdSelect.value = finalValue;
        slider.value = finalValue;
        number.value = finalValue;
      };

      thresholdSelect.addEventListener('change', syncFromSelect);
      slider.addEventListener('input', (e) => syncFromControls(e.target.value));
      number.addEventListener('input', (e) => syncFromControls(e.target.value));
      syncFromSelect();
    }

    function bindUi() {
      const gotoBtn = document.getElementById('gotoCoordBtn');
      const gotoLatInput = document.getElementById('gotoLatInput');
      const gotoLonInput = document.getElementById('gotoLonInput');
      const clearPinsBtn = document.getElementById('clearPinsBtn');
      const selectedDeleteBtn = document.getElementById('deleteSelectedPinBtn');
      const gpsBtn = document.getElementById('gpsBtn');

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

      if (clearPinsBtn) {
        clearPinsBtn.addEventListener('click', clearMarkers);
      }

      if (selectedDeleteBtn) {
        selectedDeleteBtn.addEventListener('click', removeSelectedMarker);
      }

      if (gpsBtn) {
        gpsBtn.addEventListener('click', () => {
          if (!navigator.geolocation) return;
          navigator.geolocation.getCurrentPosition((pos) => {
            const lat = pos.coords.latitude;
            const lon = pos.coords.longitude;
            focusLocation(lat, lon, 10);
            addMarker(lat, lon, 'Current location');
          }, () => {
            alert('Location access was denied.');
          }, { enableHighAccuracy: true, timeout: 10000 });
        });
      }

      if (map) {
        map.on('click', (event) => {
          const { lat, lng } = event.latlng;
          updateReadout(lat, lng);
          if (document.getElementById('pinToggle') && document.getElementById('pinToggle').checked) {
            const marker = addMarker(lat, lng, 'Map pin');
            if (marker) {
              state.selectedMarker = marker;
            }
          }
        });

        map.on('move', () => {
          const c = map.getCenter();
          state.lastCenter = c;
          updateReadout(c.lat, c.lng);
        });
      }
    }

    installThresholdControls();
    bindUi();

    if (map) {
      const fallbackCenter = map.getCenter();
      updateReadout(fallbackCenter.lat, fallbackCenter.lng);
    }

    window.__phase1 = {
      parseCoordString,
      addMarker,
      clearMarkers,
      removeSelectedMarker,
      focusLocation,
      updateReadout,
    };
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPhase1);
  } else {
    initPhase1();
  }
})();
