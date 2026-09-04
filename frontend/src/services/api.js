/**
 * API Service — Centralized HTTP client for the FastAPI backend.
 * Base URL defaults to http://localhost:8000.
 */
import axios from 'axios';

const isBrowser = typeof window !== 'undefined';
const host = isBrowser && window.location.hostname ? window.location.hostname : 'localhost';
const API_BASE = `http://${host}:8000`;

const api = axios.create({
  baseURL: API_BASE,
  timeout: 15000,
  headers: { 'Content-Type': 'application/json' },
});

// ── Network ──────────────────────────────────────────────────────────────────

export async function fetchNetwork(windowId, source = 'lan') {
  const params = { source };
  if (windowId != null) params.window_id = windowId;
  const { data } = await api.get('/api/network', { params });
  return data;
}

// ── Risk ─────────────────────────────────────────────────────────────────────

export async function fetchRisk(windowId, source = 'lan') {
  const params = { source };
  if (windowId != null) params.window_id = windowId;
  const { data } = await api.get('/api/risk', { params });
  return data;
}

// ── Predictions ──────────────────────────────────────────────────────────────

export async function fetchPredictions(windowId, topK = 5, model = 'xgboost', source = 'lan') {
  const params = { top_k: topK, model, source };
  if (windowId != null) params.window_id = windowId;
  const { data } = await api.get('/api/predictions', { params });
  return data;
}

// ── Timeline ─────────────────────────────────────────────────────────────────

export async function fetchTimeline() {
  const { data } = await api.get('/api/timeline');
  return data;
}

// ── Evaluation ───────────────────────────────────────────────────────────────

export async function fetchEvaluation() {
  const { data } = await api.get('/api/evaluation');
  return data;
}

// ── Explanation ──────────────────────────────────────────────────────────────

export async function fetchExplanation(deviceId, windowId) {
  const params = {};
  if (windowId != null) params.window_id = windowId;
  const { data } = await api.get(`/api/explanation/${encodeURIComponent(deviceId)}`, { params });
  return data;
}

// ── Recommendations ──────────────────────────────────────────────────────────

export async function fetchRecommendations(deviceId, windowId) {
  const params = {};
  if (windowId != null) params.window_id = windowId;
  const { data } = await api.get(`/api/recommendations/${encodeURIComponent(deviceId)}`, { params });
  return data;
}

// ── Attack Path ──────────────────────────────────────────────────────────────

export async function fetchAttackPath(deviceId, windowId) {
  const params = {};
  if (windowId != null) params.window_id = windowId;
  const { data } = await api.get(`/api/attack-path/${encodeURIComponent(deviceId)}`, { params });
  return data;
}

// ── Analyze ──────────────────────────────────────────────────────────────────

export async function triggerAnalysis(windowId, model = 'xgboost', topK = 5, weights = null) {
  const payload = {
    window_id: windowId,
    model,
    top_k: topK,
  };
  if (weights) {
    if (weights.w_prob != null) payload.w_prob = weights.w_prob;
    if (weights.w_anom != null) payload.w_anom = weights.w_anom;
    if (weights.w_crit != null) payload.w_crit = weights.w_crit;
    if (weights.w_expo != null) payload.w_expo = weights.w_expo;
    if (weights.w_vuln != null) payload.w_vuln = weights.w_vuln;
  }
  const { data } = await api.post('/api/analyze', payload);
  return data;
}

// ── LAN Network Devices ──────────────────────────────────────────────────────

export async function fetchLanDevices() {
  const { data } = await api.get('/api/network/lan-devices');
  return data;
}

// ── Health ────────────────────────────────────────────────────────────────────

export async function fetchHealth() {
  const { data } = await api.get('/health');
  return data;
}

export default api;
