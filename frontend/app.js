const API_BASE = "http://127.0.0.1:8000";
let events = [];
let selectedEvent = null;
let map;
let eventLayer;
let trajectoryLayer;
let selectedLayer;
let alertLayer;
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
    return Number.isNaN(date.getTime()) ? "Updated: N/A" : `Updated: ${date.toLocaleString("en-IN", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false })} IST`;
}

function updateDataIndicator(source, updatedAt, baseline) {
    const normalized = String(source || "UNAVAILABLE").toUpperCase();
    const badge = byId("dataModeBadge");
    const label = normalized === "LIVE" ? "LIVE DATA" : normalized === "CACHED" ? "CACHED DATA" : normalized === "DEMO" ? "DEMO DATA" : "LIVE DATA UNAVAILABLE";
    badge.textContent = label; badge.className = normalized.toLowerCase();
    byId("dataModeText").textContent = normalized === "LIVE" ? "Open-Meteo forecast with statistical anomaly detection" : normalized === "CACHED" ? "Cached Open-Meteo forecast with statistical anomaly detection" : normalized === "DEMO" ? "Explicit demo records; not live weather" : "Connect to the backend to retrieve forecast data";
    byId("lastUpdate").textContent = formatUpdatedAt(updatedAt);
    byId("baselineText").textContent = `Baseline: ${safeValue(baseline, "statistical reference thresholds")}`;
    byId("systemData").textContent = normalized === "UNAVAILABLE" ? "UNAVAILABLE" : normalized;
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

function getPersistenceHours(event) {
    if (Number.isFinite(Number(event.persistence_hours))) return Number(event.persistence_hours);
    if (Number.isFinite(Number(event.forecast_start)) && Number.isFinite(Number(event.forecast_end))) {
        return Math.max(0, Number(event.forecast_end) - Number(event.forecast_start));
    }
    return null;
}

function generateAnomalyEvidence(event) {
    if (!event) return null;
    const type = String(event.type || "").toLowerCase();
    const latestTrajectory = Array.isArray(event.trajectory) && event.trajectory.length
        ? event.trajectory[event.trajectory.length - 1] : null;
    let primaryMetric = "Weather anomaly";
    let primaryValue = null;

    if (type.includes("heat") || type.includes("cold") || type.includes("temperature")) {
        primaryMetric = "Temperature anomaly";
        primaryValue = event.temperature_anomaly;
    } else if (type.includes("rain") || type.includes("flood")) {
        primaryMetric = "Rainfall anomaly";
        primaryValue = event.rainfall_anomaly;
        if (primaryValue === undefined) primaryValue = event.peak_rainfall ?? latestTrajectory?.rainfall;
    } else if (type.includes("wind") || type.includes("cyclone")) {
        primaryMetric = "Wind anomaly";
        primaryValue = event.wind_anomaly ?? event.peak_wind ?? event.wind_speed;
    }

    const valueSuffix = primaryMetric === "Temperature anomaly" ? " °C" : primaryMetric === "Rainfall anomaly" ? " mm" : primaryMetric === "Wind anomaly" ? " km/h" : "";
    return {
        primaryMetric,
        primaryValue: primaryValue === null || primaryValue === undefined ? "N/A" : formatNumber(primaryValue, valueSuffix),
        persistence: getPersistenceHours(event),
        spatialCoverage: event.affected_area_km2,
        movement: event.movement,
        score: event.anomaly_score,
        confidence: event.confidence
    };
}

function renderEvidence(event) {
    const evidence = generateAnomalyEvidence(event);
    const values = evidence || {};
    byId("evidenceSignal").textContent = safeValue(values.primaryMetric);
    byId("evidenceValue").textContent = safeValue(values.primaryValue);
    byId("evidencePersistence").textContent = values.persistence === null || values.persistence === undefined ? "N/A" : `${formatNumber(values.persistence)} hours`;
    byId("evidenceCoverage").textContent = formatNumber(values.spatialCoverage, " km²");
    byId("evidenceMovement").textContent = safeValue(values.movement);
    byId("evidenceScore").textContent = formatNumber(values.score, "/100");
    byId("evidenceConfidence").textContent = formatNumber(values.confidence, "%");
    byId("temperatureAnomaly").textContent = formatNumber(event?.temperature_anomaly, " °C");
    byId("windAnomaly").textContent = formatNumber(event?.wind_anomaly, " km/h");

    if (!event) {
        byId("evidenceExplanation").textContent = "No event data is currently available.";
        return;
    }
    const type = safeValue(event.type, "Weather").toString().toLowerCase();
    const signal = type.includes("heat") ? "high-temperature" : type.includes("cold") ? "below-baseline temperature" : type.includes("rain") || type.includes("flood") ? "high-rainfall" : type.includes("wind") || type.includes("cyclone") ? "high-wind" : "extreme-weather";
    const score = Number.isFinite(Number(event.anomaly_score)) ? `with an anomaly score of ${formatNumber(event.anomaly_score)}/100` : "with an elevated anomaly score";
    const confidence = Number.isFinite(Number(event.confidence)) ? ` Detection confidence is ${formatNumber(event.confidence)}%.` : " Detection confidence is unavailable.";
    const persistence = values.persistence === null || values.persistence === undefined ? "an unknown duration" : `approximately ${formatNumber(values.persistence)} hours`;
    const area = event.affected_area_km2 === null || event.affected_area_km2 === undefined ? "an unreported area" : `${formatNumber(event.affected_area_km2)} km²`;
    const movement = safeValue(event.movement, "an unknown direction").toString().toLowerCase();
    byId("evidenceExplanation").textContent = `Persistent ${signal} conditions were detected across ${area}, ${score}. The anomaly persists for ${persistence} and shows a coherent ${movement} movement pattern.${confidence}`;
}

function updateCounters() {
    const counts = { SEVERE: 0, HIGH: 0, MODERATE: 0 };
    events.forEach((event) => { const risk = event.risk || event.severity; if (counts[risk] !== undefined) counts[risk] += 1; });
    byId("totalCount").textContent = events.length;
    byId("severeCount").textContent = counts.SEVERE;
    byId("highCount").textContent = counts.HIGH;
    byId("moderateCount").textContent = counts.MODERATE;
    byId("systemEvents").textContent = events.length;
}

function renderSelectedEvent() {
    const event = selectedEvent;
    const score = Number(event?.anomaly_score);
    const confidence = Number(event?.confidence);
    const selectedHorizon = byId("horizon").value;
    byId("eventTitle").textContent = event?.id === undefined ? "EVENT N/A" : `EVENT #${event.id}`;
    byId("eventType").textContent = safeValue(event?.type);
    byId("forecast").textContent = event?.forecast_start === undefined || event?.forecast_end === undefined ? "N/A" : `+${selectedHorizon}h of +${event.forecast_start}h to +${event.forecast_end}h`;
    byId("start").textContent = event?.start ? `${safeValue(event.start.lat)}, ${safeValue(event.start.lon)}` : "N/A";
    byId("latest").textContent = event?.latest ? `${safeValue(event.latest.lat)}, ${safeValue(event.latest.lon)}` : "N/A";
    byId("movement").textContent = safeValue(event?.movement);
    byId("rainfall").textContent = formatNumber(event?.peak_rainfall, " mm");
    byId("score").textContent = formatNumber(event?.anomaly_score, "/100");
    byId("riskBadge").textContent = safeValue(event?.risk, "RISK N/A");
    byId("interpretation").textContent = safeValue(event?.interpretation, "No interpretation available.");
    byId("confidenceText").textContent = formatNumber(event?.confidence, "%");
    byId("confidenceBar").style.width = Number.isFinite(confidence) ? `${Math.max(0, Math.min(100, confidence))}%` : "0";
    byId("area").textContent = formatNumber(event?.affected_area_km2, " km²");
    byId("impactRisk").textContent = safeValue(event?.risk);
    byId("impactTags").replaceChildren(...(Array.isArray(event?.impact) ? event.impact : []).map((tag) => { const span = document.createElement("span"); span.textContent = safeValue(tag); return span; }));
    byId("whyList").replaceChildren(...(Array.isArray(event?.why) ? event.why : ["No supporting evidence available."]).map((reason) => { const li = document.createElement("li"); li.textContent = safeValue(reason); return li; }));
    renderEvidence(event);
        byId("baselineText").textContent = `Baseline: ${safeValue(event?.baseline, "statistical reference thresholds")}`;
    if (Number.isFinite(score)) byId("riskBadge").style.borderColor = score >= 75 ? "#ff315a" : score >= 55 ? "#ff8a21" : "#ffd33d";
}

function renderMarkers() {
    if (!map) return;
    eventLayer.clearLayers();
    events.forEach((event) => {
        if (!event.latest || !Number.isFinite(Number(event.latest.lat)) || !Number.isFinite(Number(event.latest.lon))) return;
        const risk = event.risk || event.severity;
        const marker = L.circleMarker([event.latest.lat, event.latest.lon], { radius: 8, color: risk === "SEVERE" ? "#ff315a" : risk === "HIGH" ? "#ff8a21" : "#ffd33d", fillOpacity: .85, weight: 2 });
        marker.bindPopup(`<b>Event #${safeValue(event.id)}</b><br>${safeValue(event.type)}<br>Severity: ${safeValue(event.risk)}<br>Anomaly score: ${formatNumber(event.anomaly_score, "/100")}<br>Forecast: +${safeValue(event.forecast_end)}h`);
        marker.on("click", () => { byId("eventSelect").value = event.id; selectEvent(event.id); });
        marker.addTo(eventLayer);
    });
}

function renderTimeline() {
    const timeline = byId("timeline");
    const points = getVisibleTrajectory();
    const stages = points.length ? [
        { label: "FORECAST START", point: points[0] },
        { label: "DETECTION", point: points[Math.min(1, points.length - 1)] },
        { label: "CURRENT POSITION", point: points[Math.max(0, points.length - 2)] },
        { label: "LATEST / FORECAST", point: points[points.length - 1] }
    ] : [];
    timeline.replaceChildren(...stages.map((stage, index) => {
        const point = stage.point; const step = document.createElement("div"); step.className = `step${index === stages.length - 1 ? " active" : ""}`;
        step.innerHTML = `<div class="node">+${safeValue(point.hour)}h</div><strong>${stage.label}</strong><small>${safeValue(point.lat)}, ${safeValue(point.lon)}</small>`;
        return step;
    }));
}

function selectEvent(id) {
    selectedEvent = events.find((event) => String(event.id) === String(id)) || null;
    renderSelectedEvent();
    renderTimeline();
    if (selectedEvent) drawTrajectory(getVisibleTrajectory());
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
    const points = trajectory.filter((point) => Number.isFinite(Number(point.lat)) && Number.isFinite(Number(point.lon))).map((point) => [point.lat, point.lon]);
    if (points.length >= 1) {
        const line = L.polyline(points, { color: "#ff9c37", weight: 4, opacity: .95, dashArray: "8 8" });
        if (points.length > 1) { line.bindPopup(`${safeValue(selectedEvent?.type)} forecast trajectory`); line.addTo(trajectoryLayer); }
        const start = points[0]; const latest = points[points.length - 1];
        L.circleMarker(start, { radius: 6, color: "#a9c5dd", weight: 2, fillColor: "#0d2237", fillOpacity: 1 }).bindPopup(`<b>Forecast start</b><br>${start[0]}, ${start[1]}`).addTo(selectedLayer);
        points.slice(1, -1).forEach((point, index) => L.circleMarker(point, { radius: 4, color: "#ffcf83", weight: 1, fillColor: "#ff9c37", fillOpacity: .9 }).bindPopup(`<b>Forecast position +${safeValue(trajectory[index + 1]?.hour)}h</b><br>${point[0]}, ${point[1]}`).addTo(selectedLayer));
        const marker = L.circleMarker(latest, { radius: 11, color: "#ffffff", weight: 3, fillColor: "#ff9c37", fillOpacity: 1 });
        marker.bindPopup(`<b>Current / latest position</b><br>${safeValue(selectedEvent?.type)}<br>${latest[0]}, ${latest[1]}`).addTo(selectedLayer);
        if (selectedEvent?.affected_area_km2) {
            const radius = Math.sqrt(Number(selectedEvent.affected_area_km2) / Math.PI) * 1000;
            if (Number.isFinite(radius)) L.circle(latest, { radius, color: "#ff9c37", fillColor: "#ff9c37", fillOpacity: .08, weight: 1, dashArray: "4 6" }).addTo(selectedLayer);
        }
        map.fitBounds(points.length > 1 ? line.getBounds() : L.latLngBounds([latest, latest]), { padding: [30, 30], maxZoom: 7 });
    }
}

async function loadSystemHealth() {
    try {
        const response = await fetchWithTimeout(`${API_BASE}/api/system/status`);
        if (!response.ok) throw new Error(`Health request failed (${response.status})`);
        const health = await response.json();
        byId("systemApi").textContent = safeValue(health.api, "OFFLINE");
        byId("systemModel").textContent = safeValue(health.anomaly_engine, "OFFLINE");
        byId("systemMap").textContent = safeValue(health.map, "OFFLINE");
        byId("systemApi").classList.toggle("offline", health.api !== "ONLINE");
        byId("systemModel").classList.toggle("offline", health.anomaly_engine !== "ONLINE");
        byId("systemMap").classList.toggle("offline", health.map !== "ONLINE");
        updateDataIndicator(health.data, health.last_data_update, "statistical reference thresholds");
        return health;
    } catch (error) {
        byId("systemApi").textContent = "OFFLINE"; byId("systemData").textContent = "UNAVAILABLE"; byId("systemModel").textContent = "OFFLINE";
        updateDataIndicator("UNAVAILABLE", null, "statistical reference thresholds");
        ["systemApi", "systemData", "systemModel"].forEach((id) => byId(id).classList.add("offline"));
        throw error;
    }
}

async function fetchWithTimeout(url, options = {}) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 8000);
    try { return await fetch(url, { ...options, signal: controller.signal }); }
    finally { window.clearTimeout(timeout); }
}

async function trackEvent() {
    if (!selectedEvent) return showToast("Select an event before tracking.");
    const button = byId("trackBtn"); button.disabled = true; button.textContent = "TRACKING...";
    try {
        const response = await fetchWithTimeout(`${API_BASE}/api/events/${selectedEvent.id}/track`);
        if (!response.ok) throw new Error(`Tracking request failed (${response.status})`);
        const tracked = await response.json();
        if (!Array.isArray(tracked.trajectory)) throw new Error("Tracking response has no trajectory");
        selectedEvent = { ...selectedEvent, ...tracked };
        renderSelectedEvent(); renderTimeline(); drawTrajectory(getVisibleTrajectory());
        setApiStatus(true);
        showToast(`Trajectory loaded for Event #${selectedEvent.id}.`);
    } catch (error) { showToast(`Unable to track event: ${error.name === "AbortError" ? "request timed out" : error.message}`, true); setApiStatus(false); }
    finally { button.disabled = false; button.textContent = "TRACK WEATHER EVENT"; }
}

async function createAlert() {
    if (!selectedEvent) return showToast("Select an event before generating an alert.");
    if (alertRequestInFlight) return;
    const button = byId("alertBtn"); alertRequestInFlight = true; button.disabled = true; button.textContent = "GENERATING...";
    try {
        const response = await fetchWithTimeout(`${API_BASE}/api/alerts`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ event_id: selectedEvent.id, radius_km: 5 }) });
        if (!response.ok) throw new Error(`Alert request failed (${response.status})`);
        const result = await response.json();
        const latest = selectedEvent.latest; const latitude = Number(latest?.lat); const longitude = Number(latest?.lon);
        if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) throw new Error("Current event position is unavailable");
        byId("alertId").textContent = safeValue(result.alert_id); byId("alertEvent").textContent = `#${safeValue(selectedEvent.id)} · ${safeValue(selectedEvent.type)}`;
        byId("alertSeverity").textContent = safeValue(result.severity, selectedEvent.risk); byId("alertPosition").textContent = `${latitude}, ${longitude}`;
        byId("alertTimestamp").textContent = new Date().toLocaleString(); byId("alertAction").textContent = selectedEvent.type.toLowerCase().includes("heat") ? "Activate heat-health monitoring and hydration support." : "Notify local responders and monitor the 5 km impact zone.";
        byId("alertPanel").hidden = false; alertLayer.clearLayers(); const zone = L.circle([latitude, longitude], { radius: 5000, color: "#55e0a0", fillColor: "#55e0a0", fillOpacity: .18, weight: 2 }); zone.bindPopup(`<b>5 km alert zone</b><br>${safeValue(result.alert_id)}<br>${safeValue(selectedEvent.type)}`).addTo(alertLayer); map.fitBounds(zone.getBounds(), { padding: [30, 30] });
        setApiStatus(true); showToast("Alert generated successfully.");
    } catch (error) { showToast(`Unable to generate alert: ${error.name === "AbortError" ? "request timed out" : error.message}`, true); setApiStatus(false); }
    finally { alertRequestInFlight = false; button.disabled = false; button.textContent = "GENERATE 5 KM ALERT"; }
}

async function loadEvents() {
    try {
        const response = await fetchWithTimeout(`${API_BASE}/api/events`);
        if (!response.ok) throw new Error(`Event request failed (${response.status})`);
        const payload = await response.json(); events = Array.isArray(payload.events) ? payload.events : [];
        setApiStatus(true); updateDataIndicator(payload.source, payload.updated_at, payload.baseline); updateCounters();
        const selector = byId("eventSelect"); selector.replaceChildren(...events.map((event) => { const option = document.createElement("option"); option.value = event.id; option.textContent = `Event #${safeValue(event.id)} · ${safeValue(event.type)}`; return option; }));
        if (events.length) { selector.value = events[0].id; selectEvent(events[0].id); renderMarkers(); }
        else { renderSelectedEvent(); renderTimeline(); showToast("API returned no event data."); }
    } catch (error) { setApiStatus(false, `API OFFLINE: ${error.name === "AbortError" ? "request timed out" : error.message}`); byId("systemData").textContent = "OFFLINE"; byId("systemData").classList.add("offline"); updateCounters(); renderSelectedEvent(); }
}

function initialize() {
    if (window.L) { map = L.map("map").setView([23.5, 82.5], 5); L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { attribution: "&copy; OpenStreetMap contributors" }).addTo(map); eventLayer = L.layerGroup().addTo(map); trajectoryLayer = L.layerGroup().addTo(map); selectedLayer = L.layerGroup().addTo(map); alertLayer = L.layerGroup().addTo(map); byId("systemMap").textContent = "READY"; }
    else { byId("systemMap").textContent = "OFFLINE"; byId("systemMap").classList.add("offline"); }
    byId("eventSelect").addEventListener("change", (event) => selectEvent(event.target.value));
    byId("closeAlertBtn").addEventListener("click", () => { byId("alertPanel").hidden = true; });
    byId("trackBtn").addEventListener("click", trackEvent); byId("alertBtn").addEventListener("click", createAlert);
    byId("horizon").addEventListener("change", () => { renderSelectedEvent(); renderTimeline(); if (selectedEvent) drawTrajectory(getVisibleTrajectory()); });
    loadSystemHealth().catch((error) => showToast(`Health check unavailable: ${error.message}`, true));
    loadEvents();
}

window.addEventListener("DOMContentLoaded", initialize);