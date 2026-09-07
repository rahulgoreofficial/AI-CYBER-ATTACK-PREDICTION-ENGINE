/**
 * API Service — Centralized HTTP client & WebSocket manager for the FastAPI backend.
 * Base URL defaults to http://localhost:8000.
 * WebSocket URL defaults to ws://localhost:8000/ws/network.
 */
import axios from 'axios';

const isBrowser = typeof window !== 'undefined';
const host = isBrowser && window.location.hostname ? window.location.hostname : 'localhost';
const API_BASE = `http://${host}:8000`;
const WS_BASE = `ws://${host}:8000`;

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


// ══════════════════════════════════════════════════════════════════════════════
// WEBSOCKET MANAGER — Real-time Network Event Stream
// ══════════════════════════════════════════════════════════════════════════════

/**
 * Creates a persistent WebSocket connection to the backend network scanner.
 * Auto-reconnects on disconnect with exponential backoff.
 *
 * @param {function} onMessage - Callback when a network event is received.
 *        Receives parsed JSON: { type, device, total_devices, all_devices, timestamp, scan_cycle }
 * @param {function} onStatusChange - Callback for connection status changes.
 *        Receives status string: 'connecting', 'connected', 'disconnected', 'error'
 * @returns {{ close: function, send: function, getStatus: function }}
 */
export function createNetworkWebSocket(onMessage, onStatusChange) {
  let ws = null;
  let reconnectTimer = null;
  let reconnectDelay = 1000; // Start at 1s, exponential backoff
  const MAX_RECONNECT_DELAY = 15000;
  let intentionallyClosed = false;
  let status = 'disconnected';

  function updateStatus(newStatus) {
    status = newStatus;
    if (onStatusChange) onStatusChange(newStatus);
  }

  function connect() {
    if (intentionallyClosed) return;

    try {
      updateStatus('connecting');
      ws = new WebSocket(`${WS_BASE}/ws/network`);

      ws.onopen = () => {
        reconnectDelay = 1000; // Reset backoff on success
        updateStatus('connected');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type !== 'ack' && onMessage) {
            onMessage(data);
          }
        } catch (err) {
          console.warn('Failed to parse WebSocket message:', err);
        }
      };

      ws.onclose = (event) => {
        updateStatus('disconnected');
        if (!intentionallyClosed) {
          // Auto-reconnect with exponential backoff
          reconnectTimer = setTimeout(() => {
            reconnectDelay = Math.min(reconnectDelay * 1.5, MAX_RECONNECT_DELAY);
            connect();
          }, reconnectDelay);
        }
      };

      ws.onerror = (error) => {
        console.debug('WebSocket error:', error);
        updateStatus('error');
      };
    } catch (err) {
      console.warn('WebSocket connection failed:', err);
      updateStatus('error');
      if (!intentionallyClosed) {
        reconnectTimer = setTimeout(() => {
          reconnectDelay = Math.min(reconnectDelay * 1.5, MAX_RECONNECT_DELAY);
          connect();
        }, reconnectDelay);
      }
    }
  }

  // Start connection
  connect();

  return {
    close() {
      intentionallyClosed = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (ws) {
        ws.close();
        ws = null;
      }
      updateStatus('disconnected');
    },
    send(message) {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(typeof message === 'string' ? message : JSON.stringify(message));
      }
    },
    getStatus() {
      return status;
    },
  };
}


export default api;
