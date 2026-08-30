/**
 * API Service — Centralized HTTP client for the FastAPI backend.
 * Base URL defaults to http://localhost:8000.
 */
import axios from 'axios';

const API_BASE = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 15000,
  headers: { 'Content-Type': 'application/json' },
});

// ── Network ──────────────────────────────────────────────────────────────────

export async function fetchNetwork(windowId) {
  const params = {};
  if (windowId != null) params.window_id = windowId;
  const { data } = await api.get('/api/network', { params });
  return data;
}

// ── Risk ─────────────────────────────────────────────────────────────────────

export async function fetchRisk(windowId) {
  const params = {};
  if (windowId != null) params.window_id = windowId;
  const { data } = await api.get('/api/risk', { params });
  return data;
}

// ── Predictions ──────────────────────────────────────────────────────────────

export async function fetchPredictions(windowId, topK = 5, model = 'xgboost') {
  const params = { top_k: topK, model };
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

export async function triggerAnalysis(windowId, model = 'xgboost', topK = 5) {
  const { data } = await api.post('/api/analyze', {
    window_id: windowId,
    model,
    top_k: topK,
  });
  return data;
}

// ── Health ────────────────────────────────────────────────────────────────────

export async function fetchHealth() {
  const { data } = await api.get('/health');
  return data;
}

export default api;
