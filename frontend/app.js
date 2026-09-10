const API_BASE = window.location.protocol === "file:" || ["localhost", "127.0.0.1"].includes(window.location.hostname)
    ? "http://127.0.0.1:8000"
    : "";
let events = [];
let selectedEvent = null;
let map;
let eventLayer;
let trajectoryLayer;
let selectedLayer;
let alertLayer;
let locationLayer;
let fieldLayers = {};
let spatialField = [];
let locationIntelligence = null;
let alertRequestInFlight = false;
let playbackTimer;
let judgeStep = 0;
let chartMode = "composite";
let locationLoadingTimer;

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
const severityColor = (score) => Number(score) >= 80 ? "#ff315a" : Number(score) >= 60 ? "#ff8a21" : Number(score) >= 40 ? "#ffd33d" : "#55e0a0";

function setApiStatus(online, message = "") {
    const status = byId("apiStatus");
    status.textContent = online ? "● SYSTEM OPERATIONAL" : "● SYSTEM DEGRADED";
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
    byId("stripData").textContent = normalized;
    byId("stripData").classList.toggle("offline", normalized === "UNAVAILABLE");
}

function showToast(message, isError = false) {
    const toast = byId("toast");
    toast.textContent = safeValue(message);
    toast.classList.toggle("error", isError);
    toast.style.display = "block";
    window.clearTimeout(showToast.timer);
    showToast.timer = window.setTimeout(() => { toast.style.display = "none"; }, 4200);
}

function locationNumber(value, suffix = "") { return value === null || value === undefined || !Number.isFinite(Number(value)) ? "N/A" : `${Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 })}${suffix}`; }
function renderTrendChart(values, labels) {
    const container = byId("eventChart");
    if (!container) return;
    container.replaceChildren();
    if (!values.length) { const empty = document.createElement("span"); empty.className = "chartEmpty"; empty.textContent = "Select an event or location to view anomaly evolution."; container.append(empty); return; }
    const width = 720; const height = 150; const padding = 12; const min = Math.min(0, ...values); const max = Math.max(1, ...values); const range = max - min || 1;
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg"); svg.setAttribute("viewBox", `0 0 ${width} ${height}`); svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "Anomaly evolution chart");
    [0.25, 0.5, 0.75].forEach((fraction) => { const line = document.createElementNS("http://www.w3.org/2000/svg", "line"); const y = padding + fraction * (height - padding * 2); line.setAttribute("x1", padding); line.setAttribute("x2", width - padding); line.setAttribute("y1", y); line.setAttribute("y2", y); line.classList.add("chartGrid"); svg.append(line); });
    const points = values.map((value, index) => { const x = padding + (values.length === 1 ? 0 : index / (values.length - 1)) * (width - padding * 2); const y = height - padding - ((value - min) / range) * (height - padding * 2); return [x, y]; });
    const area = document.createElementNS("http://www.w3.org/2000/svg", "path"); area.setAttribute("d", `M ${points[0][0]} ${height - padding} L ${points.map((point) => point.join(" ")).join(" L ")} L ${points[points.length - 1][0]} ${height - padding} Z`); area.classList.add("chartArea"); svg.append(area);
    const path = document.createElementNS("http://www.w3.org/2000/svg", "polyline"); path.setAttribute("points", points.map((point) => point.join(",")).join(" ")); path.classList.add("chartLine"); svg.append(path);
    points.forEach((point, index) => { const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle"); circle.setAttribute("cx", point[0]); circle.setAttribute("cy", point[1]); circle.setAttribute("r", "4"); circle.classList.add("chartPoint"); circle.appendChild(document.createElementNS("http://www.w3.org/2000/svg", "title")).textContent = `${labels[index] || `+${index * 24}H`}: ${Number(values[index]).toFixed(2)}σ`; svg.append(circle); });
    container.append(svg);
}

function renderEventChart(event) {
    const points = event?.trajectory || [];
    const values = points.map((point) => { if (chartMode === "composite") return Number(point.anomaly_score || point.score || 0) / 100 * 3; return Number(point.z_scores?.[chartMode] || 0); });
    renderTrendChart(values, points.map((point) => `+${point.hour || 0}H`));
}

function locationMetricRow(label, value, severity = "") {
    const row = document.createElement("div"); row.className = "locationForecastRow";
    const name = document.createElement("span"); name.textContent = label;
    const result = document.createElement("b"); result.textContent = value; if (severity) result.dataset.severity = severity; row.append(name, result); return row;
}

function renderLocationIntelligence(result) {
    locationIntelligence = result;
    const tomorrow = result.tomorrow || {};
    const anomalies = result.anomalies || {};
    const impact = result.local_impact || {};
    const baseline = result.baseline || {};
    byId("locationCard").hidden = false;
    byId("locationName").textContent = safeValue(result.location?.name, "Selected location");
    byId("locationMeta").textContent = safeValue(result.location?.display_name, `${result.location?.latitude}, ${result.location?.longitude}`);
    byId("locationSource").textContent = safeValue(result.source, "UNAVAILABLE");
    byId("locationSource").className = `sourcePill ${String(result.source || "").toLowerCase()}`;
    byId("locationTemperature").textContent = locationNumber(tomorrow.temperature_c, " °C");
    byId("locationCondition").textContent = safeValue(tomorrow.condition, "Forecast condition unavailable");
    byId("locationRainfall").textContent = locationNumber(tomorrow.rainfall_mm, " mm");
    byId("locationWind").textContent = locationNumber(tomorrow.wind_kmh, " km/h");
    byId("locationTemperatureZ").textContent = locationNumber(anomalies.temperature_z, "σ"); byId("locationRainfallZ").textContent = locationNumber(anomalies.rainfall_z, "σ"); byId("locationWindZ").textContent = locationNumber(anomalies.wind_z, "σ");
    byId("locationComposite").textContent = locationNumber(anomalies.composite_sigma, "σ"); byId("locationRisk").textContent = safeValue(result.severity); byId("locationConfidence").textContent = locationNumber(result.confidence, "%");
    byId("locationWeatherStatus").textContent = result.severity === "NORMAL" ? "● NORMAL · Conditions within expected range" : `● ${safeValue(result.severity)} ANOMALY`;
    byId("locationWeatherStatus").dataset.severity = safeValue(result.severity);
    byId("locationExplanationList").replaceChildren(...(result.explanation || []).map((text) => { const li = document.createElement("li"); li.textContent = text; return li; }));
    byId("locationPrimary").textContent = safeValue(result.dominant_signal); byId("locationSecondary").textContent = safeValue(result.secondary_signal);
    byId("locationForecastRows").replaceChildren(...(result.forecast || []).slice(0, 5).map((item) => locationMetricRow(item.hour === 0 ? "TODAY" : item.hour === 24 ? "TOMORROW" : `+${item.hour}H`, `${locationNumber(item.temperature_c, " °C")} · ${safeValue(item.severity)}`, item.severity)));
    byId("locationBaselineText").replaceChildren(locationMetricRow("TEMPERATURE EXPECTED", locationNumber(baseline.temperature_mean, " °C")), locationMetricRow("RAINFALL EXPECTED", locationNumber(baseline.rainfall_mean, " mm")), locationMetricRow("WIND EXPECTED", locationNumber(baseline.wind_mean, " km/h")), locationMetricRow("BASELINE TYPE", safeValue(baseline.label)));
    byId("locationImpactArea").textContent = locationNumber(impact.affected_area_km2, " km²"); byId("locationImpactTemp").textContent = locationNumber(impact.temperature_c, " °C"); byId("locationImpactRain").textContent = locationNumber(impact.rainfall_mm, " mm"); byId("locationImpactWind").textContent = locationNumber(impact.wind_kmh, " km/h"); byId("locationImpactSigma").textContent = locationNumber(impact.local_anomaly_sigma, "σ"); byId("locationImpactRisk").textContent = safeValue(impact.risk);
    byId("locationEventBtn").hidden = Number(anomalies.composite_score) < 20;
    renderTrendChart((result.forecast || []).slice(0, 5).map((item) => Number(item.composite_sigma || 0)), (result.forecast || []).slice(0, 5).map((item) => `+${item.hour || 0}H`));
    if (map && locationLayer) { locationLayer.clearLayers(); const marker = L.circleMarker([result.location.latitude, result.location.longitude], { radius: 10, color: severityColor(anomalies.composite_score), fillColor: severityColor(anomalies.composite_score), fillOpacity: .85, weight: 3 }); marker.bindPopup(`<b>${safeValue(result.location.name)}</b><br>Tomorrow: ${locationNumber(tomorrow.temperature_c, " °C")}<br>Anomaly: ${locationNumber(anomalies.composite_sigma, "σ")}`).addTo(locationLayer); L.circle([result.location.latitude, result.location.longitude], { radius: 5000, color: severityColor(anomalies.composite_score), fillOpacity: .05, weight: 1, dashArray: "4 6" }).addTo(locationLayer); map.setView([result.location.latitude, result.location.longitude], 8); }
}

async function loadLocationIntelligence() {
    const query = byId("locationQuery").value.trim();
    if (!query) { byId("locationStatus").textContent = "ENTER A CITY OR COORDINATES"; return; }
    const button = byId("locationSearchBtn"); const steps = ["RESOLVING LOCATION...", "FETCHING FORECAST...", "COMPUTING BASELINE...", "CALCULATING ANOMALY...", "GENERATING INTELLIGENCE..."]; let stepIndex = 0; button.disabled = true; button.textContent = "ANALYZING..."; byId("locationStatus").textContent = steps[0]; window.clearInterval(locationLoadingTimer); locationLoadingTimer = window.setInterval(() => { stepIndex = Math.min(stepIndex + 1, steps.length - 1); byId("locationStatus").textContent = steps[stepIndex]; }, 420);
    try { const response = await fetchWithTimeout(`${API_BASE}/api/location-intelligence?location=${encodeURIComponent(query)}`); if (!response.ok) { const detail = await response.json().catch(() => ({})); throw new Error(detail.detail || `Location request failed (${response.status})`); } const result = await response.json(); renderLocationIntelligence(result); byId("locationStatus").textContent = `${safeValue(result.source)} FORECAST · ${result.fallback_reason ? "PROVIDER UNAVAILABLE · DEMO VALUES" : "ANALYSIS COMPLETE"}`; } catch (error) { byId("locationStatus").textContent = error.message.toLowerCase().includes("not found") ? "LOCATION NOT FOUND" : "FORECAST UNAVAILABLE"; showToast(byId("locationStatus").textContent, true); } finally { window.clearInterval(locationLoadingTimer); button.disabled = false; button.textContent = "ANALYZE LOCATION"; }
}

async function createLocationEvent() {
    if (!locationIntelligence?.location) return;
    const button = byId("locationEventBtn"); button.disabled = true; button.textContent = "CREATING...";
    try { const response = await fetchWithTimeout(`${API_BASE}/api/location-events`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query: locationIntelligence.location.display_name }) }); if (!response.ok) { const detail = await response.json().catch(() => ({})); throw new Error(detail.detail || `Event creation failed (${response.status})`); } const payload = await response.json(); const event = payload.event; events.push(event); const option = document.createElement("option"); option.value = event.id; option.textContent = `Event #${event.id} · ${event.type}`; byId("eventSelect").append(option); byId("eventSelect").value = event.id; updateCounters(); selectEvent(event.id); showToast(`Weather event #${event.id} created for ${locationIntelligence.location.name}.`); } catch (error) { showToast(error.message, true); } finally { button.disabled = false; button.textContent = "CREATE WEATHER EVENT"; }
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
    byId("threatList").replaceChildren(...events.slice(0, 6).map((event) => {
        const item = document.createElement("button"); item.className = "threatItem"; item.type = "button";
        const dot = document.createElement("i"); dot.style.background = severityColor(event.anomaly_score); const title = document.createElement("span"); title.textContent = safeValue(event.type, "Weather event");
        const severity = document.createElement("b"); severity.textContent = safeValue(event.risk || event.severity); const location = document.createElement("small"); location.textContent = event.latest ? `${event.latest.lat}, ${event.latest.lon}` : "Location unavailable";
        item.append(dot, title, severity, location); item.addEventListener("click", () => { byId("eventSelect").value = event.id; selectEvent(event.id); }); return item;
    }));
}

function metricRow(label, value, width) {
    const row = document.createElement("div"); row.className = "metricRow";
    const name = document.createElement("span"); name.textContent = label;
    const text = document.createElement("b"); text.textContent = value; row.append(name, text);
    const bar = document.createElement("i"); bar.style.width = `${Math.max(0, Math.min(100, Number(width) || 0))}%`; row.append(bar); return row;
}

function renderIntelligence(event) {
    const evidence = event?.evidence || {};
    const baseline = Number(evidence.baseline_value); const forecast = Number(evidence.forecast_value);
    const suffix = evidence.dominant_variable === "temperature" ? " °C" : evidence.dominant_variable === "rainfall" ? " mm" : " km/h";
    byId("scienceForecast").textContent = formatNumber(forecast, suffix); byId("scienceBaseline").textContent = formatNumber(baseline);
    byId("scienceDeviation").textContent = Number.isFinite(forecast) && Number.isFinite(baseline) ? formatNumber(forecast - baseline) : "N/A";
    byId("scienceStd").textContent = formatNumber(evidence.standard_deviation); byId("scienceZ").textContent = formatNumber(evidence.z_score, "σ"); byId("scienceTrend").textContent = formatNumber(evidence.intensity_trend_percent, "%");
    byId("lifecycleStatus").textContent = safeValue(event?.lifecycle); byId("peakForecast").textContent = event?.peak?.hour === null || event?.peak?.hour === undefined ? "N/A" : `+${event.peak.hour}H · ${formatNumber(event.peak.score, "/100")}`;
    byId("compoundStatus").textContent = safeValue(event?.compound_hazard?.label); byId("priorityStatus").textContent = safeValue(event?.priority?.level);
    byId("severityForecast").replaceChildren(...(event?.severity_forecast || []).map((item) => metricRow(`+${item.hour}H`, `${item.severity} · ${formatNumber(item.score, "/100")}`, item.score)));
    byId("confidenceTimeline").replaceChildren(...(event?.confidence_timeline || []).map((item) => metricRow(`+${item.hour}H`, formatNumber(item.confidence, "%"), item.confidence)));
    const drivers = event?.risk_drivers || {};
    byId("riskDrivers").replaceChildren(...["rainfall", "temperature", "wind", "persistence", "spatial_expansion"].map((key) => metricRow(key.replace("_", " "), formatNumber(drivers[key], "%"), drivers[key])));
    byId("evolutionGraph").replaceChildren(...(event?.trajectory || []).map((point) => { const wrap = document.createElement("div"); wrap.className = "evolutionPoint"; const bar = document.createElement("i"); bar.style.height = `${Math.max(6, Math.min(100, Number(point.anomaly_score || point.score || 0)))}%`; bar.style.background = severityColor(point.anomaly_score || point.score); wrap.append(bar); const label = document.createElement("span"); label.textContent = `+${safeValue(point.hour)}H`; wrap.append(label); return wrap; }));
    renderEventChart(event);
    const lifecycleStates = ["DETECTED", "CONFIRMED", "INTENSIFYING", "PEAK", "WEAKENING", "DISSIPATED"];
    document.querySelectorAll(".lifecycle span").forEach((node, index) => node.classList.toggle("active", lifecycleStates[index] === event?.lifecycle));
}

function renderSelectedEvent() {
    const event = selectedEvent;
    const score = Number(event?.anomaly_score);
    const confidence = Number(event?.confidence);
    const selectedHorizon = byId("horizon").value;
    byId("eventTitle").textContent = event?.id === undefined ? "EVENT N/A" : `EVENT #${event.id}`;
    byId("mapForecast").textContent = `+${selectedHorizon}H`;
    byId("mapEvent").textContent = safeValue(event?.type, "N/A").toUpperCase();
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
    const tracking = event?.tracking || {};
    byId("trackingDirection").textContent = safeValue(tracking.direction || event?.movement);
    byId("trackingSpeed").textContent = formatNumber(tracking.speed_kmh, " km/h");
    byId("trackingDistance").textContent = formatNumber(tracking.displacement_km, " km");
    byId("trackingConfidence").textContent = formatNumber(tracking.confidence || event?.confidence, "%");
    byId("analysisText").textContent = safeValue(event?.analysis, "Select ANALYZE EVENT to generate an evidence-based summary.");
    byId("responseRisk").textContent = safeValue(event?.risk || event?.severity);
    byId("responseText").textContent = event?.type ? (String(event.type).toLowerCase().includes("rain") ? "Increase local monitoring and prepare localized flood-response resources after official validation." : "Increase local monitoring and prepare targeted response resources after official validation.") : "Select an event to view decision-support guidance.";
    renderIntelligence(event);
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
        marker.bindPopup(`<strong>${safeValue(event.type)}</strong><br><span>Severity</span> ${safeValue(event.risk)}<br><span>Anomaly</span> ${formatNumber(Number(event.anomaly_score || 0) / 100 * 3, "σ")}<br><span>Peak</span> +${safeValue(event.peak?.hour || event.forecast_end)}H<br><span>Confidence</span> ${formatNumber(event.confidence, "%")}`, { className: "intelPopup" });
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
        trajectory.forEach((point, index) => { const radius = Number(point.uncertainty?.radius_km); if (Number.isFinite(radius)) L.circle([point.lat, point.lon], { radius: radius * 1000, color: index >= 3 ? "#ff718b" : index >= 2 ? "#ffcf83" : "#70bcff", fillOpacity: .025, weight: 1, dashArray: "3 6" }).bindPopup(`Prototype uncertainty corridor · +${safeValue(point.hour)}h`).addTo(selectedLayer); });
        const marker = L.circleMarker(latest, { radius: 11, color: "#ffffff", weight: 3, fillColor: "#ff9c37", fillOpacity: 1 });
        marker.bindPopup(`<b>Current / latest position</b><br>${safeValue(selectedEvent?.type)}<br>${latest[0]}, ${latest[1]}`).addTo(selectedLayer);
        if (selectedEvent?.affected_area_km2) {
            const radius = Math.sqrt(Number(selectedEvent.affected_area_km2) / Math.PI) * 1000;
            if (Number.isFinite(radius)) L.circle(latest, { radius, color: "#ff9c37", fillColor: "#ff9c37", fillOpacity: .08, weight: 1, dashArray: "4 6" }).addTo(selectedLayer);
        }
        map.fitBounds(points.length > 1 ? line.getBounds() : L.latLngBounds([latest, latest]), { padding: [30, 30], maxZoom: 7 });
    }
}

function renderSpatialField() {
    if (!map || !spatialField.length) return;
    Object.values(fieldLayers).forEach((layer) => layer.clearLayers());
    spatialField.forEach((cell) => {
        if (!Number.isFinite(Number(cell.lat)) || !Number.isFinite(Number(cell.lon))) return;
        const score = Number(cell.anomaly_score || 0); const z = cell.z_scores || {};
        const values = { composite: score, temperature: Math.abs(Number(z.temperature || 0)) * 30, rainfall: Math.max(0, Number(z.rainfall || 0)) * 30, wind: Math.max(0, Number(z.wind || 0)) * 30, persistence: Math.min(100, Number(cell.persistence || 0)) };
        Object.entries(values).forEach(([name, value]) => { const marker = L.circleMarker([cell.lat, cell.lon], { radius: Math.max(7, Math.min(22, 7 + Number(value) / 8)), color: severityColor(value), fillColor: severityColor(value), fillOpacity: name === "composite" ? .58 : .3, weight: name === "composite" ? 2 : 1 }); marker.bindPopup(`<b>${name.toUpperCase()} FIELD</b><br>+${safeValue(cell.hour)}h<br>Score: ${formatNumber(value)}`); marker.addTo(fieldLayers[name]); });
    });
}

async function loadSpatialField() {
    try { const response = await fetchWithTimeout(`${API_BASE}/api/spatial-field`); if (!response.ok) throw new Error(`Spatial field request failed (${response.status})`); const payload = await response.json(); spatialField = Array.isArray(payload.cells) ? payload.cells : []; renderSpatialField(); byId("anomalyMapStatus").textContent = `${spatialField.length} ANOMALY SIGNALS DISPLAYED`; } catch (error) { byId("anomalyMapStatus").textContent = "ANOMALY FIELD UNAVAILABLE"; showToast(`Spatial field unavailable: ${error.message}`, true); }
}

function updateJudgeStep() {
    const steps = [["STEP 1 · DETECT", "Select a monitored event and inspect the anomaly state.", "threatPanel"], ["STEP 2 · EXPLAIN", "Review the calculated evidence and risk drivers.", "whyPanel"], ["STEP 3 · TRACK", "Inspect movement, speed, and trajectory confidence.", "eventPanel"], ["STEP 4 · PREDICT", "Read the forecast evolution and confidence curve.", "chartPanel"], ["STEP 5 · LOCALIZE", "Open the 5 km impact analysis and map zone.", "locationImpact"], ["STEP 6 · ALERT", "Generate a structured decision-support alert.", "alertBtn"]];
    const current = steps[judgeStep] || steps[0]; byId("judgeStepTitle").textContent = current[0]; byId("judgeStepText").textContent = current[1]; document.querySelectorAll(".judgeStepNode").forEach((node, index) => node.classList.toggle("active", index === judgeStep)); document.querySelectorAll(".judgeFocus").forEach((node) => node.classList.remove("judgeFocus")); const target = document.querySelector(`.${current[2]}`) || byId(current[2]); if (target) { target.classList.add("judgeFocus"); if (judgeStep > 0) target.scrollIntoView({ behavior: "smooth", block: "nearest" }); }
}

async function loadSystemHealth() {
    try {
        const response = await fetchWithTimeout(`${API_BASE}/api/system/status`);
        if (!response.ok) throw new Error(`Health request failed (${response.status})`);
        const health = await response.json();
        byId("systemApi").textContent = safeValue(health.api, "OFFLINE");
        byId("systemModel").textContent = safeValue(health.anomaly_engine, "OFFLINE");
        byId("systemTracker").textContent = safeValue(health.tracker, "OFFLINE");
        byId("systemMap").textContent = safeValue(health.map, "OFFLINE");
        byId("stripModel").textContent = safeValue(health.anomaly_engine, "OFFLINE");
        byId("stripTracker").textContent = safeValue(health.tracker, "OFFLINE");
        byId("stripMap").textContent = safeValue(health.map, "OFFLINE");
        byId("systemApi").classList.toggle("offline", health.api !== "ONLINE");
        byId("systemModel").classList.toggle("offline", health.anomaly_engine !== "ONLINE");
        byId("systemMap").classList.toggle("offline", health.map !== "ONLINE");
        updateDataIndicator(health.data, health.last_data_update, "statistical reference thresholds");
        return health;
    } catch (error) {
        byId("systemApi").textContent = "OFFLINE"; byId("systemData").textContent = "UNAVAILABLE"; byId("systemModel").textContent = "OFFLINE"; byId("systemTracker").textContent = "OFFLINE";
        ["stripData", "stripModel", "stripTracker", "stripMap"].forEach((id) => byId(id).classList.add("offline"));
        updateDataIndicator("UNAVAILABLE", null, "statistical reference thresholds");
        ["systemApi", "systemData", "systemModel"].forEach((id) => byId(id).classList.add("offline"));
        throw error;
    }
}

async function fetchWithTimeout(url, options = {}) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 15000);
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

async function analyzeEvent() {
    if (!selectedEvent) return showToast("Select an event before analysis.");
    const button = byId("analyzeBtn"); button.disabled = true; button.textContent = "ANALYZING...";
    try {
        const response = await fetchWithTimeout(`${API_BASE}/api/events/${selectedEvent.id}/analysis`);
        if (!response.ok) throw new Error(`Analysis request failed (${response.status})`);
        const result = await response.json();
        selectedEvent = { ...selectedEvent, analysis: result.analysis, why: result.why, baseline: result.baseline };
        renderSelectedEvent();
        showToast("Analysis complete.");
    } catch (error) { showToast(`Unable to analyze event: ${error.message}`, true); setApiStatus(false); }
    finally { button.disabled = false; button.textContent = "ANALYZE EVENT"; }
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
        byId("alertArea").textContent = formatNumber(result.impact?.affected_area_km2, " km²"); byId("alertValidWindow").textContent = `${safeValue(result.valid_hours)} hours`;
        byId("alertTimestamp").textContent = new Date().toLocaleString(); byId("alertAction").textContent = safeValue(result.recommended_action);
        byId("alertPanel").hidden = false; alertLayer.clearLayers(); const zone = L.circle([latitude, longitude], { radius: 5000, color: "#55e0a0", fillColor: "#55e0a0", fillOpacity: .18, weight: 2 }); zone.bindPopup(`<b>5 km alert zone</b><br>${safeValue(result.alert_id)}<br>${safeValue(selectedEvent.type)}`).addTo(alertLayer); map.fitBounds(zone.getBounds(), { padding: [30, 30] });
        setApiStatus(true); showToast("Alert generated successfully.");
    } catch (error) { showToast(`Unable to generate alert: ${error.name === "AbortError" ? "request timed out" : error.message}`, true); setApiStatus(false); }
    finally { alertRequestInFlight = false; button.disabled = false; button.textContent = "GENERATE 5 KM ALERT"; }
}

function setPlaybackHour(hour) {
    const value = Number(hour);
    byId("forecastSlider").value = value;
    byId("playbackHour").textContent = `+${value}H`;
    byId("horizon").value = String(value || 24);
    if (selectedEvent) { renderSelectedEvent(); renderTimeline(); drawTrajectory(getVisibleTrajectory()); }
}

function togglePlayback() {
    const button = byId("playBtn");
    if (playbackTimer) {
        window.clearInterval(playbackTimer); playbackTimer = null; button.textContent = "PLAY FORECAST"; byId("playbackStatus").textContent = "PAUSED"; return;
    }
    byId("playbackStatus").textContent = "PLAYING"; button.textContent = "PAUSE";
    playbackTimer = window.setInterval(() => {
        const next = Number(byId("forecastSlider").value) + 24;
        if (next > 96) { setPlaybackHour(0); } else { setPlaybackHour(next); }
    }, 1200);
}

async function loadEvents() {
    try {
        const response = await fetchWithTimeout(`${API_BASE}/api/events`);
        if (!response.ok) throw new Error(`Event request failed (${response.status})`);
        const payload = await response.json(); events = Array.isArray(payload.events) ? payload.events : [];
        setApiStatus(true); updateDataIndicator(payload.source, payload.updated_at, payload.baseline); updateCounters();
        const selector = byId("eventSelect"); selector.replaceChildren(...events.map((event) => { const option = document.createElement("option"); option.value = event.id; option.textContent = `Event #${safeValue(event.id)} · ${safeValue(event.type)}`; return option; }));
        if (events.length) { selector.value = events[0].id; selectEvent(events[0].id); renderMarkers(); loadSpatialField(); }
        else { renderSelectedEvent(); renderTimeline(); showToast("API returned no event data."); }
    } catch (error) { setApiStatus(false, `API OFFLINE: ${error.name === "AbortError" ? "request timed out" : error.message}`); byId("systemData").textContent = "OFFLINE"; byId("systemData").classList.add("offline"); updateCounters(); renderSelectedEvent(); }
}

function initialize() {
    if (window.L) { map = L.map("map").setView([23.5, 82.5], 5); L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { attribution: "&copy; OpenStreetMap contributors" }).addTo(map); eventLayer = L.layerGroup().addTo(map); trajectoryLayer = L.layerGroup().addTo(map); selectedLayer = L.layerGroup().addTo(map); alertLayer = L.layerGroup().addTo(map); locationLayer = L.layerGroup().addTo(map); fieldLayers = { composite: L.layerGroup().addTo(map), temperature: L.layerGroup(), rainfall: L.layerGroup(), wind: L.layerGroup(), persistence: L.layerGroup() }; L.control.layers({}, { "Composite Risk": fieldLayers.composite, "Temperature Anomaly": fieldLayers.temperature, "Rainfall Anomaly": fieldLayers.rainfall, "Wind Anomaly": fieldLayers.wind, "Persistence": fieldLayers.persistence, "Uncertainty Corridor": selectedLayer, "Event Trajectory": trajectoryLayer, "5 km Impact Zone": alertLayer, "Location Intelligence": locationLayer }, { collapsed: false }).addTo(map); byId("systemMap").textContent = "READY"; }
    else { byId("systemMap").textContent = "OFFLINE"; byId("systemMap").classList.add("offline"); }
    byId("eventSelect").addEventListener("change", (event) => selectEvent(event.target.value));
    byId("closeAlertBtn").addEventListener("click", () => { byId("alertPanel").hidden = true; });
    byId("trackBtn").addEventListener("click", trackEvent); byId("alertBtn").addEventListener("click", createAlert);
    byId("analyzeBtn").addEventListener("click", analyzeEvent);
    byId("evidenceBtn").addEventListener("click", () => showToast("Scientific evidence is calculated by the backend."));
    byId("judgeModeBtn").addEventListener("click", () => { const panel = byId("judgePanel"); panel.hidden = !panel.hidden; if (!panel.hidden) updateJudgeStep(); });
    byId("judgeNextBtn").addEventListener("click", () => { judgeStep = (judgeStep + 1) % 6; updateJudgeStep(); });
    byId("judgeResetBtn").addEventListener("click", () => { judgeStep = 0; setPlaybackHour(0); byId("alertPanel").hidden = true; updateJudgeStep(); });
    byId("responseAlertBtn").addEventListener("click", createAlert);
    byId("locationSearchBtn").addEventListener("click", loadLocationIntelligence);
    byId("locationEventBtn").addEventListener("click", createLocationEvent);
    byId("locationImpactBtn").addEventListener("click", () => { byId("locationImpact").scrollIntoView({ behavior: "smooth", block: "center" }); });
    [["chartCompositeBtn", "composite"], ["chartTemperatureBtn", "temperature"], ["chartRainfallBtn", "rainfall"], ["chartWindBtn", "wind"]].forEach(([id, mode]) => byId(id).addEventListener("click", () => { chartMode = mode; document.querySelectorAll(".chartSwitch button").forEach((button) => button.classList.remove("active")); byId(id).classList.add("active"); if (selectedEvent) renderEventChart(selectedEvent); else if (locationIntelligence) renderTrendChart((locationIntelligence.forecast || []).slice(0, 5).map((item) => mode === "composite" ? Number(item.composite_sigma || 0) : Number(item[`${mode}_z`] || 0)), (locationIntelligence.forecast || []).slice(0, 5).map((item) => `+${item.hour || 0}H`)); }));
    byId("forecastSlider").addEventListener("input", (event) => setPlaybackHour(event.target.value));
    byId("playBtn").addEventListener("click", togglePlayback);
    byId("resetBtn").addEventListener("click", () => { if (playbackTimer) togglePlayback(); setPlaybackHour(0); });
    byId("horizon").addEventListener("change", () => { renderSelectedEvent(); renderTimeline(); if (selectedEvent) drawTrajectory(getVisibleTrajectory()); });
    loadSystemHealth().catch((error) => showToast(`Health check unavailable: ${error.message}`, true));
    loadEvents();
}

window.addEventListener("DOMContentLoaded", initialize);