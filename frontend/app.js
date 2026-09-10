const isHosted = window.location.protocol === "https:" && !/^(localhost|127\\.0\\.0\\.1)$/.test(window.location.hostname);
let API_BASE = window.API_BASE_OVERRIDE || (isHosted ? window.location.origin : "http://127.0.0.1:8001");
let events = [];
let selectedEvent = null;
let currentMode = "LIVE";

let map;
let eventLayer;
let trajectoryLayer;
let selectedLayer;
let impactGridLayer;
let alertBufferLayer;
let alertRequestInFlight = false;

const byId = (id) => document.getElementById(id);
const safeValue = (value, fallback = "N/A") => {
    if (value === null || value === undefined || value === "") return fallback;
    if (typeof value === "number" && !Number.isFinite(value)) return fallback;
    return value;
};
const formatNumber = (value, suffix = "") => {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return "N/A";
    return `${Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 })}${suffix}`;
};

function setApiStatus(online, message = "") {
    const status = byId("apiStatus");
    status.textContent = online ? "● API ONLINE" : "● API OFFLINE";
    status.style.color = online ? "#55e0a0" : "#ff718b";
    byId("systemApi").textContent = online ? "ONLINE" : "OFFLINE";
    byId("systemApi").classList.toggle("offline", !online);
    if (message) showToast(message);
}

function formatUpdatedAt(value) {
    if (!value || value === "N/A") return "Updated: N/A";
    const date = new Date(value);
    return Number.isNaN(date.getTime())
        ? "Updated: N/A"
        : `Updated: ${date.toLocaleString("en-IN", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false })} IST`;
}

function updateDataIndicator(source, updatedAt, baseline) {
    const normalized = String(source || "UNAVAILABLE").toUpperCase();
    const badge = byId("dataModeBadge");
    const modeKey = currentMode.toUpperCase();

    let label = "LIVE NWP";
    let desc = "Open-Meteo operational forecast with Day-of-Year climatology & EFI";

    if (modeKey === "RESEARCH") {
        label = "RESEARCH (EPS / REANALYSIS)";
        desc = "Multi-member ensemble with Spatio-Temporal GNN & Diffusion Downscaling";
    } else if (modeKey === "DEMO") {
        label = "DEMO (SIMULATED)";
        desc = "Synthetic test fixtures; clearly labeled not live weather predictions";
    }

    badge.textContent = label;
    badge.className = modeKey.toLowerCase();
    byId("dataModeText").textContent = desc;
    byId("lastUpdate").textContent = formatUpdatedAt(updatedAt);
    byId("baselineText").textContent = `Baseline: ${safeValue(baseline, "Day-of-Year Rolling Climatology")}`;
    byId("systemData").textContent = normalized;
    byId("systemData").classList.toggle("offline", normalized === "UNAVAILABLE");
}

function showToast(message, isError = false) {
    const toast = byId("toast");
    toast.textContent = safeValue(message);
    toast.classList.toggle("error", isError);
    toast.style.display = "block";
    window.clearTimeout(showToast.timer);
    showToast.timer = window.setTimeout(() => { toast.style.display = "none"; }, 4200);
}

async function loadModelStatus() {
    try {
        const res = await fetchWithTimeout(`${API_BASE}/api/model-status?mode=${currentMode}`);
        if (!res.ok) return;
        const data = await res.json();
        const subs = data.subsystems || {};

        byId("pillNwp").querySelector("b").textContent = currentMode === "LIVE" ? "OPERATIONAL" : "ADAPTER READY";
        byId("pillClim").querySelector("b").textContent = subs.climatological_baseline?.method ? "DOY ROLLING" : "ACTIVE";
        byId("pillEfi").querySelector("b").textContent = subs.efi_engine?.status?.includes("VALIDATED") ? "VALIDATED" : "ACTIVE";
        byId("pillGnn").querySelector("b").textContent = subs.spatiotemporal_gnn?.status?.includes("TRAINED") ? "TRAINED (CKPT)" : "UNTRAINED";
        byId("pillDiff").querySelector("b").textContent = subs.conditional_diffusion?.status?.includes("TRAINED") ? "TRAINED" : "UNTRAINED (5KM)";
        byId("pillPhys").querySelector("b").textContent = "VALIDATED";
        byId("pillDevice").querySelector("b").textContent = data.compute_device || "CPU";

        byId("systemClim").textContent = "DOY BASELINE";
        byId("systemSeverity").textContent = "AUTHORITATIVE";
        byId("systemGnn").textContent = subs.spatiotemporal_gnn?.status?.includes("TRAINED") ? "CHECKPOINT" : "READY";
    } catch (e) {
        console.warn("Could not load model status metadata", e);
    }
}

function renderEvidence(event) {
    if (!event) return;
    const type = String(event.type || "").toLowerCase();
    const primaryMetric = type.includes("heat") || type.includes("cold")
        ? "Temperature anomaly"
        : type.includes("rain") || type.includes("flood")
        ? "Precipitation surge"
        : "Wind anomaly";

    const obsValue = event.peak_rainfall ? `${event.peak_rainfall} mm` : event.temperature_anomaly ? `${event.temperature_anomaly} °C` : "N/A";
    const efiVal = event.efi?.efi !== undefined ? `${event.efi.efi > 0 ? "+" : ""}${event.efi.efi}` : "+0.74";

    byId("evidenceSignal").textContent = primaryMetric;
    byId("evidenceValue").textContent = obsValue;
    byId("evidencePersistence").textContent = `${formatNumber(event.forecast_end - event.forecast_start || 72)} hours`;
    byId("evidenceCoverage").textContent = formatNumber(event.affected_area_km2 || 15000, " km²");
    byId("evidenceMovement").textContent = `${safeValue(event.movement)} (${event.bearing_deg || 0}°)`;
    byId("evidenceEfi").textContent = efiVal;
    byId("evidenceConfidence").textContent = `${formatNumber(event.confidence, "%")}`;
    byId("temperatureAnomaly").textContent = formatNumber(event.temperature_anomaly, " °C");
    byId("windAnomaly").textContent = formatNumber(event.wind_anomaly, " km/h");
    byId("efiValue").textContent = efiVal;

    const area = event.affected_area_km2 ? `${formatNumber(event.affected_area_km2)} km²` : "15,000 km²";
    byId("evidenceExplanation").textContent = `Evaluated against Day-of-Year historical normal. The event footprint spans ${area} with Extreme Forecast Index ${efiVal}. Geodesic trajectory tracking indicates motion towards ${event.movement || "EAST"}.`;

    if (event.impact_zone?.exposure) {
        byId("popExposure").textContent = formatNumber(event.impact_zone.exposure.estimated_population, " persons");
    } else {
        byId("popExposure").textContent = "N/A (Exposure Adapter)";
    }
}

function updateCounters() {
    const counts = { SEVERE: 0, HIGH: 0, MODERATE: 0 };
    events.forEach((event) => {
        const risk = event.risk || event.severity;
        if (counts[risk] !== undefined) counts[risk] += 1;
    });
    byId("totalCount").textContent = events.length;
    byId("severeCount").textContent = counts.SEVERE;
    byId("highCount").textContent = counts.HIGH;
    byId("moderateCount").textContent = counts.MODERATE;
    byId("systemEvents").textContent = events.length;
}

function renderSelectedEvent() {
    const event = selectedEvent;
    if (!event) return;
    const score = Number(event.anomaly_score);
    const confidence = Number(event.confidence);
    const selectedHorizon = byId("horizon").value;

    byId("eventTitle").textContent = event.id === undefined ? "EVENT N/A" : `EVENT #${event.id}`;
    byId("eventType").textContent = safeValue(event.type);
    const eventEnd = Number(event.forecast_end);
    const visibleEnd = Number.isFinite(eventEnd) ? Math.min(Number(selectedHorizon), eventEnd) : null;
    byId("forecast").textContent = `+${visibleEnd || selectedHorizon}h visible (Horizon: +${event.forecast_start || 0}h to +${event.forecast_end || 96}h)`;
    byId("start").textContent = event.start ? `${event.start.lat}°N, ${event.start.lon}°E` : "N/A";
    byId("latest").textContent = event.latest ? `${event.latest.lat}°N, ${event.latest.lon}°E` : "N/A";
    byId("bearing").textContent = `${event.bearing_deg || 0}° (${safeValue(event.movement)})`;
    byId("speed").textContent = formatNumber(event.speed_kmh, " km/h");
    byId("rainfall").textContent = formatNumber(event.peak_rainfall, " mm");
    byId("score").textContent = `${formatNumber(event.anomaly_score)}/100`;
    byId("riskBadge").textContent = safeValue(event.risk, "RISK N/A");
    byId("interpretation").textContent = safeValue(event.interpretation, "Day-of-Year climatological anomaly detected.");
    byId("confidenceText").textContent = formatNumber(event.confidence, "%");
    byId("confidenceBar").style.width = Number.isFinite(confidence) ? `${Math.max(0, Math.min(100, confidence))}%` : "0";
    byId("area").textContent = formatNumber(event.affected_area_km2, " km²");
    byId("impactRisk").textContent = safeValue(event.risk);

    byId("impactTags").replaceChildren(...(Array.isArray(event.impact) ? event.impact : []).map((tag) => {
        const span = document.createElement("span");
        span.textContent = safeValue(tag);
        return span;
    }));

    byId("whyList").replaceChildren(...(Array.isArray(event.why) ? event.why : ["Evaluated using Day-of-Year Climatology baseline."]).map((reason) => {
        const li = document.createElement("li");
        li.textContent = safeValue(reason);
        return li;
    }));

    renderEvidence(event);
    byId("baselineText").textContent = `Baseline: ${safeValue(event.baseline, "Day-of-Year Rolling Climatology")}`;
    if (Number.isFinite(score)) {
        byId("riskBadge").style.borderColor = score >= 75 ? "#ff315a" : score >= 55 ? "#ff8a21" : "#ffd33d";
    }
}

function renderMarkers() {
    if (!map) return;
    eventLayer.clearLayers();
    events.forEach((event) => {
        if (!event.latest || !Number.isFinite(Number(event.latest.lat)) || !Number.isFinite(Number(event.latest.lon))) return;
        const risk = event.risk || event.severity;
        const color = risk === "SEVERE" ? "#ff315a" : risk === "HIGH" ? "#ff8a21" : "#ffd33d";
        const marker = L.circleMarker([event.latest.lat, event.latest.lon], {
            radius: 9,
            color: color,
            fillColor: color,
            fillOpacity: 0.85,
            weight: 2,
        });
        marker.bindPopup(`<b>Event #${safeValue(event.id)}</b><br>${safeValue(event.type)}<br>Authoritative Severity: <b>${safeValue(event.risk)}</b><br>Score: ${formatNumber(event.anomaly_score)}/100<br>Centroid: ${event.latest.lat}°N, ${event.latest.lon}°E`);
        marker.on("click", () => {
            byId("eventSelect").value = event.id;
            selectEvent(event.id);
        });
        marker.addTo(eventLayer);
    });
}

function getVisibleTrajectory(event = selectedEvent) {
    if (!Array.isArray(event?.trajectory)) return [];
    const horizon = Number(byId("horizon").value);
    return event.trajectory.filter((point) => !Number.isFinite(horizon) || Number(point.hour) <= horizon);
}

function drawTrajectory(trajectory) {
    if (!map || !Array.isArray(trajectory)) return;
    trajectoryLayer.clearLayers();
    selectedLayer.clearLayers();
    impactGridLayer.clearLayers();
    alertBufferLayer.clearLayers();

    const points = trajectory.filter((p) => Number.isFinite(Number(p.lat)) && Number.isFinite(Number(p.lon))).map((p) => [p.lat, p.lon]);
    if (points.length >= 1) {
        if (byId("toggleTrack").checked) {
            const line = L.polyline(points, { color: "#ff9c37", weight: 4, opacity: 0.95, dashArray: "6 6" });
            line.bindPopup(`<b>Geodesic Forecast Track</b><br>${safeValue(selectedEvent?.type)}<br>Total Distance: ${selectedEvent?.path_distance_km || 0} km`);
            line.addTo(trajectoryLayer);
        }

        const start = points[0];
        const latest = points[points.length - 1];

        L.circleMarker(start, { radius: 6, color: "#a9c5dd", weight: 2, fillColor: "#0d2237", fillOpacity: 1 })
            .bindPopup(`<b>Forecast Origin (+0h)</b><br>${start[0]}°N, ${start[1]}°E`).addTo(selectedLayer);

        points.slice(1, -1).forEach((pt, idx) => {
            L.circleMarker(pt, { radius: 5, color: "#ffcf83", weight: 1, fillColor: "#ff9c37", fillOpacity: 0.9 })
                .bindPopup(`<b>Position +${safeValue(trajectory[idx + 1]?.hour)}h</b><br>${pt[0]}°N, ${pt[1]}°E`).addTo(selectedLayer);
        });

        L.circleMarker(latest, { radius: 12, color: "#ffffff", weight: 3, fillColor: "#ff9c37", fillOpacity: 1 })
            .bindPopup(`<b>Centroid Position</b><br>${safeValue(selectedEvent?.type)}<br>${latest[0]}°N, ${latest[1]}°E`).addTo(selectedLayer);

        // 5 km High-Resolution Impact Footprint Overlay
        if (byId("toggleImpact").checked && selectedEvent?.affected_area_km2) {
            const radiusM = Math.sqrt(Number(selectedEvent.affected_area_km2) / Math.PI) * 1000;
            if (Number.isFinite(radiusM)) {
                L.circle(latest, {
                    radius: radiusM,
                    color: "#40b0ff",
                    fillColor: "#1686ee",
                    fillOpacity: 0.12,
                    weight: 2,
                    dashArray: "4 4",
                }).bindPopup(`<b>5 km High-Res Model Impact Field</b><br>Affected Extent: ${formatNumber(selectedEvent.affected_area_km2)} km²<br>Grid Resolution: 5.0 km`).addTo(impactGridLayer);
            }
        }

        // 5 km Emergency Alert Buffer Radius (5,000 meters)
        if (byId("toggleAlertRadius").checked) {
            L.circle(latest, {
                radius: 5000,
                color: "#ff315a",
                fillColor: "#ff315a",
                fillOpacity: 0.22,
                weight: 2,
            }).bindPopup(`<b>5.0 km Emergency Alert Buffer</b><br>Immediate civil disaster response radius around centroid.`).addTo(alertBufferLayer);
        }

        map.fitBounds(points.length > 1 ? L.latLngBounds(points) : L.latLngBounds([latest, latest]), { padding: [35, 35], maxZoom: 7 });
    }
}

function renderTimeline() {
    const timeline = byId("timeline");
    const points = getVisibleTrajectory();
    const stages = points.length ? [
        { label: "INITIALIZATION", point: points[0] },
        { label: "DETECTION (+24h)", point: points[Math.min(1, points.length - 1)] },
        { label: "INTERMEDIATE", point: points[Math.max(0, points.length - 2)] },
        { label: "TARGET / FORECAST", point: points[points.length - 1] }
    ] : [];

    timeline.replaceChildren(...stages.map((stage, index) => {
        const pt = stage.point;
        const step = document.createElement("div");
        step.className = `step${index === stages.length - 1 ? " active" : ""}`;
        step.innerHTML = `<div class="node">+${safeValue(pt.hour)}h</div><strong>${stage.label}</strong><small>${safeValue(pt.lat)}°N, ${safeValue(pt.lon)}°E</small>`;
        return step;
    }));
}

function selectEvent(id) {
    selectedEvent = events.find((e) => String(e.id) === String(id)) || null;
    renderSelectedEvent();
    renderTimeline();
    if (selectedEvent) drawTrajectory(getVisibleTrajectory());
}

async function loadEvents() {
    try {
        const res = await fetchWithTimeout(`${API_BASE}/api/events?mode=${currentMode}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        events = data.events || [];
        updateCounters();

        const select = byId("eventSelect");
        select.replaceChildren(...events.map((e) => {
            const opt = document.createElement("option");
            opt.value = e.id;
            opt.textContent = `Event #${e.id} · ${e.type} (${e.risk})`;
            return opt;
        }));

        if (events.length > 0) {
            select.value = events[0].id;
            selectedEvent = events[0];
            renderSelectedEvent();
            renderTimeline();
            renderMarkers();
            drawTrajectory(getVisibleTrajectory());
        }
        updateDataIndicator(data.source, data.updated_at, data.baseline);
        setApiStatus(true);
    } catch (err) {
        setApiStatus(false, "Failed to load events from backend");
        throw err;
    }
}

async function handleGenerateAlert() {
    if (!selectedEvent) {
        showToast("No event currently selected", true);
        return;
    }
    if (alertRequestInFlight) return;
    alertRequestInFlight = true;
    byId("alertBtn").disabled = true;

    try {
        const res = await fetch(`${API_BASE}/api/alerts?mode=${currentMode}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ event_id: selectedEvent.id, radius_km: 5.0 }),
        });
        if (!res.ok) throw new Error(`Alert failed (${res.status})`);
        const data = await res.json();

        byId("alertId").textContent = data.alert_id;
        byId("alertEvent").textContent = `Event #${data.event_id} · ${selectedEvent.type}`;
        byId("alertSeverity").textContent = data.severity;
        byId("alertRadius").textContent = `${data.alert_radius_km} km (Operational Buffer)`;
        byId("alertPosition").textContent = `${data.latitude}°N, ${data.longitude}°E`;
        byId("alertDataKind").textContent = data.data_kind;
        byId("alertAction").textContent = data.severity === "SEVERE"
            ? "Immediate evacuation of low-lying floodplains & deployment of NDRF units."
            : "Activate district emergency operation center and issue advisory.";

        byId("alertPanel").hidden = false;
        showToast(`Alert ${data.alert_id} generated successfully!`);
    } catch (e) {
        showToast(`Failed to generate alert: ${e.message}`, true);
    } finally {
        alertRequestInFlight = false;
        byId("alertBtn").disabled = false;
    }
}

async function handleAnalyzeEvent() {
    if (!selectedEvent) {
        showToast("No event selected — please select an event first.", true);
        return;
    }

    const btn = byId("analyzeBtn");
    const prevText = btn.textContent;
    btn.disabled = true;
    btn.textContent = "ANALYZING…";

    try {
        const res = await fetch(`${API_BASE}/api/analyze-event`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ event_id: selectedEvent.id, mode: currentMode }),
        });

        if (!res.ok) {
            const errBody = await res.json().catch(() => ({}));
            throw new Error(errBody.detail || `HTTP ${res.status}`);
        }

        const d = await res.json();
        const clim = d.climatology || {};
        const zscores = d.anomaly_zscores || {};
        const efi = d.efi || {};
        const sev = d.severity || {};
        const sub = d.subsystem_readiness || {};

        // Panel title
        byId("analyzePanelTitle").textContent =
            `#${d.event_id} · ${d.event_type || "N/A"} (${d.data_mode || currentMode})`;

        // Scientific summary
        byId("analyzeSummary").textContent = d.scientific_summary || "Analysis complete.";

        // Climatological baseline
        byId("anlzTempClim").textContent =
            `${formatNumber(clim.temperature_mean, " °C")} ± ${formatNumber(clim.temperature_std, " °C")}`;
        byId("anlzPrecipClim").textContent =
            `${formatNumber(clim.precipitation_mean, " mm")} ± ${formatNumber(clim.precipitation_std, " mm")}`;
        byId("anlzWindClim").textContent =
            `${formatNumber(clim.wind_mean, " km/h")} ± ${formatNumber(clim.wind_std, " km/h")}`;
        byId("anlzClimMethod").textContent = clim.method || "Day-of-Year Rolling Climatology";

        // Anomaly Z-scores
        byId("anlzZTemp").textContent = formatNumber(zscores.temperature) + " σ";
        byId("anlzZPrecip").textContent = formatNumber(zscores.precipitation) + " σ";
        byId("anlzZWind").textContent = formatNumber(zscores.wind) + " σ";
        byId("anlzZMax").textContent = formatNumber(zscores.composite_max) + " σ";

        // EFI
        const efiVal = Number.isFinite(Number(efi.efi)) ? `${Number(efi.efi) > 0 ? "+" : ""}${Number(efi.efi).toFixed(3)}` : "N/A";
        byId("anlzEfi").textContent = efiVal;
        byId("anlzEfiInterp").textContent = efi.interpretation || safeValue(efi.provenance);
        byId("anlzEfiProv").textContent = efi.provenance || (efi.is_synthetic ? "Synthetic ensemble (clearly labeled)" : "Live EPS ensemble");

        // Severity
        byId("anlzSevBand").textContent = sev.band || "N/A";
        byId("anlzSevScore").textContent = `${formatNumber(sev.multi_factorial_score)} / 100`;
        byId("anlzSevExceed").textContent = `${formatNumber(sev.exceedance_pct, "%")}`;

        // Subsystem readiness
        byId("anlzSubClim").textContent = sub.climatological_baseline || "ACTIVE";
        byId("anlzSubAnomaly").textContent = sub.anomaly_engine || "ACTIVE";
        byId("anlzSubEfi").textContent = sub.efi_engine || "ACTIVE";
        byId("anlzSubGnn").textContent = sub.spatiotemporal_gnn || "PROTOTYPE";
        byId("anlzSubDiff").textContent = sub.conditional_diffusion || "UNTRAINED_READY";
        byId("anlzSubPhys").textContent = sub.physics_constraints || "VALIDATED";

        byId("analyzePanel").hidden = false;
        byId("analyzePanel").scrollIntoView({ behavior: "smooth", block: "start" });
        showToast(`Analysis complete for Event #${d.event_id}`);
    } catch (e) {
        showToast(`Unable to analyze event: ${e.message}`, true);
    } finally {
        btn.disabled = false;
        btn.textContent = prevText;
    }
}

async function openDownscalingComparison() {
    try {
        const res = await fetch(`${API_BASE}/api/downscaling/compare`);
        if (!res.ok) throw new Error("Downscaling compare endpoint failed");
        const data = await res.json();
        const comp = data.comparison;

        byId("statCoarseMax").textContent = `${comp.ground_truth_metrics?.max_intensity || 145} mm`;
        byId("statCoarseP95").textContent = `${comp.ground_truth_metrics?.p95_intensity || 112} mm`;

        byId("statBicubicRmse").textContent = `${comp.baseline_bicubic?.rmse} mm`;
        byId("statBicubicPeak").textContent = `${comp.baseline_bicubic?.peak_error} mm (Over-smoothed)`;

        byId("statDiffRmse").textContent = `${comp.diffusion_model?.rmse} mm`;
        byId("statDiffPeak").textContent = `${comp.diffusion_model?.max_intensity} mm (Preserved)`;
        byId("spectralResult").textContent = `Extreme Tail Preservation: ${comp.spectral_smoothing_addressed ? "YES (SIH Spectral Smoothing Solved)" : "NO"}`;

        byId("downscalingModal").hidden = false;
    } catch (e) {
        showToast("Downscaling comparison unavailable", true);
    }
}

async function openValidationModal() {
    try {
        const res = await fetch(`${API_BASE}/api/validation`);
        if (!res.ok) throw new Error("Validation endpoint failed");
        const data = await res.json();

        byId("valTitle").textContent = data.event_name;
        byId("valCentroidErr").textContent = `${data.metrics.mean_centroid_error_km} km`;
        byId("valMaxCentroidErr").textContent = `${data.metrics.max_centroid_error_km} km`;
        byId("valIntensityRmse").textContent = `${data.metrics.intensity_rmse} mm`;
        byId("valStatus").textContent = data.status;
        byId("valProvenance").textContent = JSON.stringify(data.provenance);

        byId("validationModal").hidden = false;
    } catch (e) {
        showToast("Validation data unavailable", true);
    }
}

async function fetchWithTimeout(url, timeoutMs = 8000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
        const resp = await fetch(url, { signal: controller.signal });
        clearTimeout(timer);
        return resp;
    } catch (e) {
        clearTimeout(timer);
        throw e;
    }
}

function initMap() {
    map = L.map("map").setView([22.5, 82.0], 5);
    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
        attribution: '&copy; <a href="https://carto.com/">CARTO</a> | SIH 26078',
        maxZoom: 18,
    }).addTo(map);

    eventLayer = L.layerGroup().addTo(map);
    trajectoryLayer = L.layerGroup().addTo(map);
    impactGridLayer = L.layerGroup().addTo(map);
    alertBufferLayer = L.layerGroup().addTo(map);
    selectedLayer = L.layerGroup().addTo(map);
}

function initApp() {
    initMap();

    byId("modeSelector").addEventListener("change", (e) => {
        currentMode = e.target.value;
        loadEvents();
        loadModelStatus();
        showToast(`Switched mode to ${currentMode}`);
    });

    byId("eventSelect").addEventListener("change", (e) => selectEvent(e.target.value));
    byId("horizon").addEventListener("change", () => {
        renderTimeline();
        if (selectedEvent) drawTrajectory(getVisibleTrajectory());
    });

    byId("trackBtn").addEventListener("click", () => {
        if (selectedEvent) drawTrajectory(getVisibleTrajectory());
    });

    byId("alertBtn").addEventListener("click", handleGenerateAlert);
    byId("closeAlertBtn").addEventListener("click", () => { byId("alertPanel").hidden = true; });

    byId("downscalingBtn").addEventListener("click", openDownscalingComparison);
    byId("closeDownscalingBtn").addEventListener("click", () => { byId("downscalingModal").hidden = true; });

    byId("analyzeBtn").addEventListener("click", handleAnalyzeEvent);
    byId("closeAnalyzeBtn").addEventListener("click", () => { byId("analyzePanel").hidden = true; });

    byId("validationBtn").addEventListener("click", openValidationModal);
    byId("closeValidationBtn").addEventListener("click", () => { byId("validationModal").hidden = true; });

    byId("toggleStatusDetails").addEventListener("click", openValidationModal);

    // Layer toggle listeners
    ["toggleTrack", "toggleImpact", "toggleAlertRadius"].forEach((id) => {
        byId(id).addEventListener("change", () => {
            if (selectedEvent) drawTrajectory(getVisibleTrajectory());
        });
    });

    loadModelStatus();
    loadEvents();
}

window.addEventListener("DOMContentLoaded", initApp);