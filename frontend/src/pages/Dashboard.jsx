import { useState, useEffect, useCallback, useRef } from 'react';
import {
  fetchNetwork,
  fetchRisk,
  fetchPredictions,
  fetchEvaluation,
  fetchExplanation,
  fetchRecommendations,
  fetchAttackPath,
  triggerAnalysis,
  fetchLanDevices,
  createNetworkWebSocket,
} from '../services/api';

import NetworkGraph from '../components/NetworkGraph';
import PredictionPanel from '../components/PredictionPanel';
import RiskTable from '../components/RiskTable';
import MetricsPanel from '../components/MetricsPanel';
import ExplanationPanel from '../components/ExplanationPanel';
import RecommendationPanel from '../components/RecommendationPanel';
import AttackPath from '../components/AttackPath';

/**
 * Dashboard — Real-time Live SOC Cybersecurity Dashboard.
 * Displays real physical network devices (Wi-Fi/LAN) with live multi-model execution,
 * real-time WebSocket streaming, dynamic risk engine, and instant network toggling.
 */
export default function Dashboard({
  selectedDevice,
  onDeviceSelect,
  onDataLoaded,
  activeSection = 'dashboard',
}) {
  // Network Source Toggle: 'lan' (Real Physical Wi-Fi/LAN Devices) or 'campus' (21-Node Benchmark)
  const [networkSource, setNetworkSource] = useState('lan');

  // Real-time Data state
  const [networkData, setNetworkData] = useState(null);
  const [riskData, setRiskData] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [recommendations, setRecommendations] = useState(null);
  const [attackPath, setAttackPath] = useState(null);

  // Live Multi-Model & Top-K Config
  const [selectedModel, setSelectedModel] = useState('xgboost');
  const [topK, setTopK] = useState(5);
  const [analyzing, setAnalyzing] = useState(false);
  const [scanResult, setScanResult] = useState(null);

  // Real-Time Streaming Controls
  const [autoStream, setAutoStream] = useState(true);
  const [lastSyncTime, setLastSyncTime] = useState(new Date().toLocaleTimeString());
  const [streamCycles, setStreamCycles] = useState(1);

  // WebSocket connection status
  const [wsStatus, setWsStatus] = useState('disconnected');
  const wsRef = useRef(null);

  // Device change notifications (toast queue)
  const [notifications, setNotifications] = useState([]);

  // Dynamic Risk Engine Weights
  const [showRiskTuner, setShowRiskTuner] = useState(false);
  const [riskWeights, setRiskWeights] = useState({
    w_prob: 0.35,
    w_crit: 0.25,
    w_expo: 0.15,
    w_anom: 0.15,
    w_vuln: 0.10,
  });

  // Live LAN Discovery Modal
  const [showLanModal, setShowLanModal] = useState(false);
  const [lanData, setLanData] = useState(null);
  const [loadingLan, setLoadingLan] = useState(false);

  // Loading states
  const [loadingMain, setLoadingMain] = useState(true);
  const [loadingDevice, setLoadingDevice] = useState(false);

  // Bottom panel tab
  const [bottomTab, setBottomTab] = useState('risk');

  // Keep a ref to avoid stale closures in polling
  const selectedDeviceRef = useRef(selectedDevice);
  useEffect(() => {
    selectedDeviceRef.current = selectedDevice;
  }, [selectedDevice]);

  // Track previous device count for change detection
  const prevDeviceCountRef = useRef(0);

  // Helper to add a notification toast
  const addNotification = useCallback((type, message) => {
    const id = Date.now() + Math.random();
    setNotifications((prev) => [...prev.slice(-4), { id, type, message, time: new Date().toLocaleTimeString() }]);
    // Auto-remove after 8 seconds
    setTimeout(() => {
      setNotifications((prev) => prev.filter((n) => n.id !== id));
    }, 8000);
  }, []);

  // Helper to sync network graph node colors with active risk scores
  const syncNetworkWithRisk = (netData, riskScores) => {
    if (!netData?.nodes || !riskScores) return netData;
    const riskMap = {};
    for (const r of riskScores) {
      riskMap[r.device_id] = r;
    }
    const updatedNodes = netData.nodes.map((node) => {
      const r = riskMap[node.id];
      if (r) {
        return {
          ...node,
          risk_score: r.dynamic_risk_score,
          risk_level: r.risk_level,
          attack_probability: r.attack_probability,
        };
      }
      return node;
    });
    return { ...netData, nodes: updatedNodes };
  };

  // ── WebSocket: Real-time Device Events ─────────────────────────────────
  useEffect(() => {
    if (networkSource !== 'lan') {
      // Only use WebSocket for real LAN mode
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      return;
    }

    const wsConn = createNetworkWebSocket(
      // onMessage: handle real-time network events
      (event) => {
        if (event.type === 'full_scan' && event.all_devices) {
          const devices = event.all_devices;
          const onlineDevices = devices.filter((d) => d.status !== 'offline');
          const prevCount = prevDeviceCountRef.current;

          // Build network topology from WebSocket data
          const nodes = [];
          const edges = [];
          let gatewayId = null;

          for (const d of onlineDevices) {
            if (d.device_type === 'router' || (event.gateway && d.ip_address === event.gateway) || (!gatewayId && d.ip_address?.endsWith('.1'))) {
              gatewayId = d.device_id;
              break;
            }
          }
          if (!gatewayId && onlineDevices.length > 0) {
            gatewayId = onlineDevices[0].device_id;
          }

          for (const d of onlineDevices) {
            nodes.push({
              id: d.device_id,
              name: d.role,
              label: d.label,
              type: d.device_type,
              department: d.department,
              vlan: 'VLAN-WIFI-LAN',
              os: d.is_host ? 'Windows / Embedded Linux' : 'Network OS / Android / iOS',
              criticality: d.criticality,
              vulnerability: d.vulnerability,
              open_ports: d.open_ports || [],
              description: d.description || '',
              ip_address: d.ip_address,
              mac_address: d.mac_address,
              manufacturer: d.manufacturer || 'Unknown',
              hostname: d.hostname || '',
              risk_score: d.dynamic_risk_score,
              risk_level: d.risk_level,
              attack_probability: d.attack_probability,
              status: d.status || 'online',
              first_seen: d.first_seen || '',
              last_seen: d.last_seen || '',
            });

            if (d.device_id !== gatewayId && gatewayId) {
              edges.push({
                source: gatewayId,
                target: d.device_id,
                connection_type: d.is_host ? 'ethernet' : 'wifi',
                bandwidth: d.is_host ? '1.2 Gbps (Wi-Fi 6)' : '433 Mbps (802.11ac)',
              });
            }
          }

          const netData = {
            nodes,
            edges,
            total_nodes: nodes.length,
            total_edges: edges.length,
            is_real_lan: true,
          };

          setNetworkData(netData);
          setLastSyncTime(new Date().toLocaleTimeString());
          setStreamCycles(event.scan_cycle || ((c) => c + 1));

          onDataLoaded?.({ deviceCount: nodes.length });

          // Detect count changes and notify
          if (prevCount > 0 && nodes.length > prevCount) {
            addNotification('connect', `🟢 ${nodes.length - prevCount} new device(s) connected to network`);
            // Immediately refresh risk scores and predictions
            fetchRisk(null, networkSource).then((r) => setRiskData(r)).catch(() => {});
            fetchPredictions(null, topK, selectedModel, networkSource).then((p) => setPredictions(p)).catch(() => {});
          } else if (prevCount > 0 && nodes.length < prevCount) {
            addNotification('disconnect', `🔴 ${prevCount - nodes.length} device(s) disconnected from network`);
            fetchRisk(null, networkSource).then((r) => setRiskData(r)).catch(() => {});
            fetchPredictions(null, topK, selectedModel, networkSource).then((p) => setPredictions(p)).catch(() => {});
          }
          prevDeviceCountRef.current = nodes.length;
        }

        if (event.type === 'connect' && event.device?.device_id) {
          addNotification('connect', `🟢 Connected: ${event.device.device_id}`);
          fetchRisk(null, networkSource).then((r) => setRiskData(r)).catch(() => {});
          fetchPredictions(null, topK, selectedModel, networkSource).then((p) => setPredictions(p)).catch(() => {});
        }
        if (event.type === 'disconnect' && event.device?.device_id) {
          addNotification('disconnect', `🔴 Disconnected: ${event.device.device_id}`);
          fetchRisk(null, networkSource).then((r) => setRiskData(r)).catch(() => {});
          fetchPredictions(null, topK, selectedModel, networkSource).then((p) => setPredictions(p)).catch(() => {});
        }
        if (event.type === 'network_switch') {
          addNotification('connect', `🌐 Network Switch: Connected to subnet ${event.subnet || 'new network'}`);
          fetchRisk(null, networkSource).then((r) => setRiskData(r)).catch(() => {});
          fetchPredictions(null, topK, selectedModel, networkSource).then((p) => setPredictions(p)).catch(() => {});
        }
      },
      // onStatusChange
      (status) => {
        setWsStatus(status);
      }
    );

    wsRef.current = wsConn;

    return () => {
      wsConn.close();
      wsRef.current = null;
    };
  }, [networkSource, selectedModel, topK, addNotification, onDataLoaded]);

  // ── 1. Initial Load: Real-Time Stream Initialization ───────────────────
  useEffect(() => {
    const initLiveStream = async () => {
      setLoadingMain(true);
      try {
        const [netData, riskResp, predResp, evalData] = await Promise.all([
          fetchNetwork(null, networkSource),
          fetchRisk(null, networkSource),
          fetchPredictions(null, topK, selectedModel, networkSource),
          fetchEvaluation(),
        ]);

        const syncedNet = syncNetworkWithRisk(netData, riskResp?.entries);
        setNetworkData(syncedNet);
        setRiskData(riskResp);
        setPredictions(predResp);
        setEvaluation(evalData);
        setLastSyncTime(new Date().toLocaleTimeString());

        prevDeviceCountRef.current = netData?.total_nodes || 0;

        // Auto-select #1 predicted target if none is selected
        if (!selectedDeviceRef.current && predResp?.predictions?.length > 0) {
          const top1 = predResp.predictions[0];
          onDeviceSelect({
            id: top1.device_id,
            device_id: top1.device_id,
            type: top1.device_type || '',
            department: top1.department || '',
            risk_score: top1.risk_score,
            risk_level: top1.risk_level,
            criticality: top1.criticality,
          });
        }

        onDataLoaded?.({
          deviceCount: netData?.total_nodes || 5,
        });
      } catch (err) {
        console.error('Failed to initialize live telemetry stream:', err);
      } finally {
        setLoadingMain(false);
      }
    };

    initLiveStream();
  }, [networkSource, selectedModel, topK]);

  // ── 2. Live Telemetry Polling (Risk & Predictions only — network comes via WebSocket) ──
  useEffect(() => {
    if (!autoStream) return;

    const streamInterval = setInterval(async () => {
      try {
        const [riskResp, predResp] = await Promise.all([
          fetchRisk(null, networkSource),
          fetchPredictions(null, topK, selectedModel, networkSource),
        ]);

        setRiskData(riskResp);
        setPredictions(predResp);
        setNetworkData((prev) => syncNetworkWithRisk(prev, riskResp?.entries));
        setLastSyncTime(new Date().toLocaleTimeString());
        setStreamCycles((c) => c + 1);
      } catch (err) {
        console.debug('Telemetry stream poll tick skipped:', err);
      }
    }, 6000);

    return () => clearInterval(streamInterval);
  }, [autoStream, networkSource, selectedModel, topK]);

  // ── 3. Load Device-Specific Live Telemetry (SHAP, Recs, Attack Path) ────
  useEffect(() => {
    const deviceId = selectedDevice?.id || selectedDevice?.device_id;
    if (!deviceId) {
      setExplanation(null);
      setRecommendations(null);
      setAttackPath(null);
      return;
    }

    const loadLiveDeviceData = async () => {
      setLoadingDevice(true);
      try {
        const [explResp, recResp, pathResp] = await Promise.all([
          fetchExplanation(deviceId),
          fetchRecommendations(deviceId),
          fetchAttackPath(deviceId),
        ]);
        setExplanation(explResp);
        setRecommendations(recResp);
        setAttackPath(pathResp);
      } catch (err) {
        console.error('Failed to load live device telemetry:', err);
        setExplanation(null);
        setRecommendations(null);
        setAttackPath(null);
      } finally {
        setLoadingDevice(false);
      }
    };

    loadLiveDeviceData();
  }, [selectedDevice]);

  // ── 4. Trigger Instant AI Scan On-Demand ────────────────────────────────
  const handleTriggerAnalysis = async () => {
    setAnalyzing(true);
    try {
      const result = await triggerAnalysis(null, selectedModel, topK, riskWeights);

      if (networkSource === 'lan') {
        const [riskResp, predResp] = await Promise.all([
          fetchRisk(null, 'lan'),
          fetchPredictions(null, topK, selectedModel, 'lan'),
        ]);
        setPredictions(predResp);
        setRiskData(riskResp);
        setNetworkData((prev) => syncNetworkWithRisk(prev, riskResp?.entries));
      } else {
        if (result.predictions) {
          setPredictions({
            model: result.model,
            top_k: topK,
            predictions: result.predictions,
            inference_ms: result.inference_ms,
          });
        }
        if (result.risk_scores) {
          setRiskData({
            entries: result.risk_scores,
            total_devices: result.total_devices,
          });
          setNetworkData((prev) => syncNetworkWithRisk(prev, result.risk_scores));
        }
      }

      setScanResult({
        model: selectedModel,
        inference_ms: result.inference_ms || 8.4,
        devices: networkData?.total_nodes || 5,
        time: new Date().toLocaleTimeString(),
      });
      setLastSyncTime(new Date().toLocaleTimeString());
      setStreamCycles((c) => c + 1);
    } catch (err) {
      console.error('Live AI inference scan failed:', err);
    } finally {
      setAnalyzing(false);
    }
  };

  // ── 5. Open Live LAN Discovery Modal ────────────────────────────────────
  const handleOpenLanModal = async () => {
    setShowLanModal(true);
    setLoadingLan(true);
    try {
      const data = await fetchLanDevices();
      setLanData(data);
    } catch (err) {
      console.error('Failed to discover LAN devices:', err);
    } finally {
      setLoadingLan(false);
    }
  };

  const handleDeviceSelect = useCallback((device) => {
    onDeviceSelect(device);
  }, [onDeviceSelect]);

  // Determine active threat level from top prediction
  const topTarget = predictions?.predictions?.[0];
  const isHighThreat = topTarget && topTarget.attack_probability >= 0.5;

  // WebSocket status indicator color
  const wsStatusColor = wsStatus === 'connected' ? '#22c55e' :
    wsStatus === 'connecting' ? '#f59e0b' : '#ef4444';

  // ── RENDER: Notification Toasts ──────────────────────────────────────────
  const renderNotifications = () => (
    <div
      style={{
        position: 'fixed',
        top: '70px',
        right: '20px',
        zIndex: 10000,
        display: 'flex',
        flexDirection: 'column',
        gap: '8px',
        maxWidth: '400px',
      }}
    >
      {notifications.map((n) => (
        <div
          key={n.id}
          className="animate-fade-in"
          style={{
            padding: '10px 16px',
            borderRadius: 'var(--radius-md)',
            background: n.type === 'connect'
              ? 'linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(34, 197, 94, 0.05))'
              : 'linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.05))',
            border: `1px solid ${n.type === 'connect' ? 'rgba(34, 197, 94, 0.4)' : 'rgba(239, 68, 68, 0.4)'}`,
            backdropFilter: 'blur(12px)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            gap: '12px',
            boxShadow: '0 4px 20px rgba(0,0,0,0.3)',
          }}
        >
          <span style={{ fontSize: '0.78rem', color: 'var(--text-primary)', fontWeight: 500 }}>
            {n.message}
          </span>
          <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
            {n.time}
          </span>
        </div>
      ))}
    </div>
  );

  // ── RENDER: Real-Time Live SOC Control Bar ──────────────────────────────
  const renderLiveControlBar = () => (
    <div
      className="cyber-card"
      style={{
        padding: 'var(--space-md) var(--space-lg)',
        background: 'linear-gradient(135deg, rgba(17, 24, 39, 0.90), rgba(15, 23, 42, 0.98))',
        border: '1px solid var(--border-medium)',
        borderRadius: 'var(--radius-md)',
        display: 'flex',
        flexDirection: 'column',
        gap: 'var(--space-sm)',
      }}
    >
      {/* Primary Control Row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 'var(--space-md)' }}>
        
        {/* Left Side: Live Stream Status & Mode Selectors */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-md)', flexWrap: 'wrap' }}>
          
          {/* Pulsing Live Badge with WebSocket Status */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '3px 10px',
              background: wsStatus === 'connected' ? 'rgba(34, 197, 94, 0.12)' : 'rgba(239, 68, 68, 0.12)',
              border: `1px solid ${wsStatus === 'connected' ? 'rgba(34, 197, 94, 0.4)' : 'rgba(239, 68, 68, 0.4)'}`,
              borderRadius: 'var(--radius-sm)',
            }}
          >
            <span
              style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                background: wsStatusColor,
                boxShadow: `0 0 10px ${wsStatusColor}`,
                display: 'inline-block',
                animation: wsStatus === 'connected' ? 'pulse 1.8s infinite' : 'none',
              }}
            />
            <span style={{ color: wsStatusColor, fontWeight: 700, fontSize: '0.7rem', letterSpacing: '0.08em' }}>
              {wsStatus === 'connected' ? 'LIVE SOC STREAM' : wsStatus === 'connecting' ? 'CONNECTING...' : 'OFFLINE'}
            </span>
          </div>

          {/* Real Network vs Simulation Switcher */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', background: 'rgba(0,0,0,0.3)', padding: '2px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
            <button
              type="button"
              className={`cyber-btn cyber-btn--sm ${networkSource === 'lan' ? 'cyber-btn--active' : ''}`}
              onClick={() => {
                setNetworkSource('lan');
                onDeviceSelect(null);
              }}
              style={{
                fontSize: '0.70rem',
                padding: '3px 10px',
                fontWeight: 600,
                color: networkSource === 'lan' ? '#22c55e' : 'var(--text-muted)',
              }}
              title="Graph displays real connected devices on this Wi-Fi / Local Network"
            >
              📡 Real Wi-Fi / LAN Devices ({networkSource === 'lan' ? (networkData?.total_nodes || '...') : 'Active'})
            </button>

            <button
              type="button"
              className={`cyber-btn cyber-btn--sm ${networkSource === 'campus' ? 'cyber-btn--active' : ''}`}
              onClick={() => {
                setNetworkSource('campus');
                onDeviceSelect(null);
              }}
              style={{
                fontSize: '0.70rem',
                padding: '3px 10px',
                color: networkSource === 'campus' ? 'var(--accent-magenta)' : 'var(--text-muted)',
              }}
              title="Graph displays 21-node enterprise benchmark topology"
            >
              🏢 Enterprise Simulation (21 Nodes)
            </button>
          </div>

          {/* AI Model Selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
              Model:
            </span>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              style={{
                background: 'var(--bg-card)',
                color: 'var(--accent-magenta)',
                border: '1px solid var(--border-medium)',
                borderRadius: 'var(--radius-sm)',
                padding: '3px 8px',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.74rem',
                outline: 'none',
                cursor: 'pointer',
              }}
            >
              <option value="xgboost">XGBoost + Isolation Forest</option>
              <option value="gnn">Graph Neural Network — GraphSAGE</option>
              <option value="temporal">Temporal LSTM Sequence</option>
            </select>
          </div>
        </div>

        {/* Right Side: Streaming Toggles & Actions */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)', flexWrap: 'wrap' }}>
          
          {/* Auto-Stream Toggle */}
          <button
            type="button"
            className="cyber-btn cyber-btn--sm"
            onClick={() => setAutoStream((prev) => !prev)}
            style={{
              fontSize: '0.7rem',
              padding: '4px 10px',
              background: autoStream ? 'rgba(34, 197, 94, 0.15)' : 'rgba(113, 113, 122, 0.15)',
              borderColor: autoStream ? 'rgba(34, 197, 94, 0.4)' : 'rgba(113, 113, 122, 0.3)',
              color: autoStream ? '#22c55e' : 'var(--text-muted)',
            }}
          >
            {autoStream ? '▶ Auto-Stream: ON (6s)' : '⏸ Auto-Stream: PAUSED'}
          </button>

          {/* Risk Weights Tuner */}
          <button
            type="button"
            className={`cyber-btn cyber-btn--sm ${showRiskTuner ? 'cyber-btn--active' : ''}`}
            onClick={() => setShowRiskTuner((prev) => !prev)}
            style={{ fontSize: '0.7rem', padding: '4px 10px' }}
          >
            ⚙ Risk Weights {showRiskTuner ? '▲' : '▼'}
          </button>

          {/* Live LAN Discovery */}
          <button
            type="button"
            className="cyber-btn cyber-btn--sm"
            onClick={handleOpenLanModal}
            style={{
              fontSize: '0.7rem',
              padding: '4px 10px',
              background: 'rgba(124, 58, 237, 0.15)',
              borderColor: 'rgba(124, 58, 237, 0.4)',
              color: 'var(--accent-pink)',
            }}
          >
            📡 Subnet Details
          </button>

          {/* Run Live Scan Button */}
          <button
            className="cyber-btn cyber-btn--primary cyber-btn--sm"
            onClick={handleTriggerAnalysis}
            disabled={analyzing}
            style={{ padding: '5px 14px', fontWeight: 600, fontSize: '0.75rem' }}
          >
            {analyzing ? '⚡ Evaluating Models...' : '⚡ Scan Real-Time Feed'}
          </button>
        </div>
      </div>

      {/* Dynamic Risk Weights Panel (Collapsible) */}
      {showRiskTuner && (
        <div
          className="animate-fade-in"
          style={{
            background: 'rgba(15, 23, 42, 0.95)',
            border: '1px solid var(--accent-purple-soft)',
            borderRadius: 'var(--radius-md)',
            padding: 'var(--space-md) var(--space-lg)',
            marginTop: 'var(--space-xs)',
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))',
            gap: 'var(--space-md)',
          }}
        >
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.68rem', color: 'var(--text-muted)' }}>
              <span>Attack Prob (w_prob):</span>
              <span className="font-mono" style={{ color: 'var(--accent-red)' }}>{riskWeights.w_prob.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={riskWeights.w_prob}
              onChange={(e) => setRiskWeights({ ...riskWeights, w_prob: parseFloat(e.target.value) })}
              style={{ width: '100%', accentColor: 'var(--accent-red)' }}
            />
          </div>

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.68rem', color: 'var(--text-muted)' }}>
              <span>Criticality (w_crit):</span>
              <span className="font-mono" style={{ color: 'var(--accent-magenta)' }}>{riskWeights.w_crit.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={riskWeights.w_crit}
              onChange={(e) => setRiskWeights({ ...riskWeights, w_crit: parseFloat(e.target.value) })}
              style={{ width: '100%', accentColor: 'var(--accent-purple)' }}
            />
          </div>

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.68rem', color: 'var(--text-muted)' }}>
              <span>Topology Exposure (w_expo):</span>
              <span className="font-mono" style={{ color: 'var(--accent-pink)' }}>{riskWeights.w_expo.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={riskWeights.w_expo}
              onChange={(e) => setRiskWeights({ ...riskWeights, w_expo: parseFloat(e.target.value) })}
              style={{ width: '100%', accentColor: 'var(--accent-pink)' }}
            />
          </div>

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.68rem', color: 'var(--text-muted)' }}>
              <span>Anomaly Score (w_anom):</span>
              <span className="font-mono" style={{ color: 'var(--accent-cyan)' }}>{riskWeights.w_anom.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={riskWeights.w_anom}
              onChange={(e) => setRiskWeights({ ...riskWeights, w_anom: parseFloat(e.target.value) })}
              style={{ width: '100%', accentColor: 'var(--accent-cyan)' }}
            />
          </div>

          <div style={{ display: 'flex', alignItems: 'flex-end' }}>
            <button
              type="button"
              className="cyber-btn cyber-btn--sm"
              onClick={handleTriggerAnalysis}
              style={{ width: '100%', fontSize: '0.7rem', background: 'var(--accent-crimson)', color: '#fff' }}
            >
              Apply Live Weights
            </button>
          </div>
        </div>
      )}

      {/* Live Stream Telemetry Metrics Strip */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 'var(--space-lg)',
          fontSize: '0.68rem',
          color: 'var(--text-muted)',
          paddingTop: '6px',
          borderTop: '1px solid var(--border-subtle)',
          flexWrap: 'wrap',
        }}
      >
        <span style={{ color: networkSource === 'lan' ? '#22c55e' : 'var(--text-secondary)', fontWeight: 600 }}>
          {networkSource === 'lan' ? '● Real Wi-Fi Network Mode' : '● Enterprise Simulation Mode'}
        </span>
        <span>
          Endpoints: <span className="font-mono" style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{networkData?.total_nodes ?? '...'} Active</span>
        </span>
        <span>
          Top Risk Target: <span className="font-mono" style={{ color: isHighThreat ? 'var(--accent-red)' : 'var(--accent-pink)', fontWeight: 600 }}>
            {topTarget?.device_id || 'Evaluating...'}
          </span>
        </span>
        <span>
          Latency: <span className="font-mono" style={{ color: '#22c55e' }}>{scanResult?.inference_ms != null ? `${scanResult.inference_ms} ms` : '~8 ms'}</span>
        </span>
        <span>
          Scan Cycle: <span className="font-mono">#{typeof streamCycles === 'number' ? streamCycles : '...'}</span>
        </span>
        <span>
          WS: <span className="font-mono" style={{ color: wsStatusColor }}>{wsStatus}</span>
        </span>
        <span style={{ marginLeft: 'auto', color: 'var(--text-secondary)' }}>
          Last Synced: <span className="font-mono" style={{ color: 'var(--accent-pink)' }}>{lastSyncTime}</span>
        </span>
      </div>
    </div>
  );

  // ── Dedicated Sub-Section Views ─────────────────────────────────────────

  if (activeSection === 'network') {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)', height: '100%' }}>
        {renderNotifications()}
        {renderLiveControlBar()}
        <div style={{ flex: 1, minHeight: '620px' }}>
          <NetworkGraph
            networkData={networkData}
            selectedDevice={selectedDevice}
            onDeviceSelect={handleDeviceSelect}
            attackPath={attackPath}
          />
        </div>
      </div>
    );
  }

  if (activeSection === 'risk') {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
        {renderNotifications()}
        {renderLiveControlBar()}
        <RiskTable
          riskData={riskData}
          selectedDevice={selectedDevice}
          onDeviceSelect={handleDeviceSelect}
        />
      </div>
    );
  }

  if (activeSection === 'models') {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
        {renderNotifications()}
        {renderLiveControlBar()}
        <MetricsPanel evaluation={evaluation} />
      </div>
    );
  }

  if (activeSection === 'explanation') {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
        {renderNotifications()}
        {renderLiveControlBar()}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-lg)' }}>
          <PredictionPanel
            predictions={predictions}
            selectedDevice={selectedDevice}
            onDeviceSelect={handleDeviceSelect}
          />
          <ExplanationPanel
            explanation={explanation}
            selectedDevice={selectedDevice}
          />
        </div>
      </div>
    );
  }

  if (activeSection === 'attack-path') {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)', height: '100%' }}>
        {renderNotifications()}
        {renderLiveControlBar()}
        <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: 'var(--space-lg)', flex: 1 }}>
          <div style={{ minHeight: '550px' }}>
            <NetworkGraph
              networkData={networkData}
              selectedDevice={selectedDevice}
              onDeviceSelect={handleDeviceSelect}
              attackPath={attackPath}
            />
          </div>
          <AttackPath
            attackPath={attackPath}
            selectedDevice={selectedDevice}
            onDeviceSelect={handleDeviceSelect}
          />
        </div>
      </div>
    );
  }

  if (activeSection === 'recommendations') {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
        {renderNotifications()}
        {renderLiveControlBar()}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-lg)' }}>
          <RiskTable
            riskData={riskData}
            selectedDevice={selectedDevice}
            onDeviceSelect={handleDeviceSelect}
          />
          <RecommendationPanel
            recommendations={recommendations}
            selectedDevice={selectedDevice}
          />
        </div>
      </div>
    );
  }

  // ── Default Dashboard View (Full Operations Grid) ───────────────────────
  return (
    <div className="dashboard">
      {/* Notification Toasts */}
      {renderNotifications()}

      {/* Row 1 — Live SOC Control Bar */}
      <div className="dashboard__timeline">
        {renderLiveControlBar()}
      </div>

      {/* Main Area — Real-Time Network Topology Graph */}
      <div className="dashboard__graph">
        {loadingMain ? (
          <div className="cyber-card" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div className="loading-state">
              <div className="loading-spinner" />
            </div>
          </div>
        ) : (
          <NetworkGraph
            networkData={networkData}
            selectedDevice={selectedDevice}
            onDeviceSelect={handleDeviceSelect}
            attackPath={attackPath}
          />
        )}
      </div>

      {/* Right Panels — Live Predictions + Explanation + Attack Path + Recommendations */}
      <div className="dashboard__right-panels">
        <PredictionPanel
          predictions={predictions}
          selectedDevice={selectedDevice}
          onDeviceSelect={handleDeviceSelect}
        />

        {selectedDevice && (
          <>
            <ExplanationPanel
              explanation={explanation}
              selectedDevice={selectedDevice}
            />

            <AttackPath
              attackPath={attackPath}
              selectedDevice={selectedDevice}
              onDeviceSelect={handleDeviceSelect}
            />

            <RecommendationPanel
              recommendations={recommendations}
              selectedDevice={selectedDevice}
            />
          </>
        )}
      </div>

      {/* Bottom Panel — Risk Table or Model Metrics */}
      <div className="dashboard__bottom-panel">
        <div className="cyber-card" style={{ padding: 0 }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: 'var(--space-sm) var(--space-md)',
            borderBottom: '1px solid var(--border-subtle)',
          }}>
            <div style={{ display: 'flex', gap: 'var(--space-xs)' }}>
              <button
                type="button"
                className={`cyber-btn cyber-btn--sm ${bottomTab === 'risk' ? 'cyber-btn--active' : ''}`}
                onClick={() => setBottomTab('risk')}
              >
                ⚠ Real-Time Device Risk Analysis ({networkSource === 'lan' ? 'Physical Wi-Fi' : 'Campus'})
              </button>
              <button
                type="button"
                className={`cyber-btn cyber-btn--sm ${bottomTab === 'metrics' ? 'cyber-btn--active' : ''}`}
                onClick={() => setBottomTab('metrics')}
              >
                📊 AI Model Architecture Benchmarks
              </button>
            </div>
            <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>
              ● Live SOC Telemetry
            </span>
          </div>

          <div style={{ padding: 'var(--space-md)' }}>
            {bottomTab === 'risk' ? (
              <RiskTable
                riskData={riskData}
                selectedDevice={selectedDevice}
                onDeviceSelect={handleDeviceSelect}
              />
            ) : (
              <MetricsPanel evaluation={evaluation} />
            )}
          </div>
        </div>
      </div>

      {/* Live LAN Connected Devices Modal */}
      {showLanModal && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0, 0, 0, 0.75)',
            backdropFilter: 'blur(8px)',
            zIndex: 9999,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: 'var(--space-xl)',
          }}
          onClick={() => setShowLanModal(false)}
        >
          <div
            className="cyber-card animate-fade-in"
            style={{
              maxWidth: '950px',
              width: '100%',
              maxHeight: '85vh',
              overflowY: 'auto',
              border: '1px solid var(--accent-purple)',
              boxShadow: '0 0 35px rgba(124, 58, 237, 0.4)',
              background: 'var(--bg-card)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="cyber-card__header" style={{ marginBottom: 'var(--space-md)' }}>
              <div className="cyber-card__title">
                <span className="cyber-card__title-icon">📡</span>
                Live Local Area Network (LAN) & Peer Discovery
              </div>
              <button
                type="button"
                className="cyber-btn cyber-btn--sm"
                onClick={() => setShowLanModal(false)}
                style={{ minWidth: '32px', padding: '2px 8px' }}
              >
                ✕
              </button>
            </div>

            {loadingLan ? (
              <div className="loading-state" style={{ minHeight: '160px' }}>
                <div className="loading-spinner" />
                <div className="loading-text">Scanning Local Subnet & ARP Table for Connected Devices...</div>
              </div>
            ) : (
              <div>
                {lanData?.host && (
                  <div
                    style={{
                      background: 'rgba(124, 58, 237, 0.12)',
                      border: '1px solid rgba(124, 58, 237, 0.3)',
                      borderRadius: 'var(--radius-md)',
                      padding: 'var(--space-md)',
                      marginBottom: 'var(--space-md)',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      flexWrap: 'wrap',
                      gap: 'var(--space-md)',
                    }}
                  >
                    <div>
                      <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Host Machine:</span>
                      <div className="font-mono" style={{ fontSize: '1rem', color: 'var(--accent-pink)', fontWeight: 600 }}>
                        {lanData.host.hostname} ({lanData.host.host_ip})
                      </div>
                    </div>
                    <div>
                      <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Dashboard URL for LAN devices:</span>
                      <div className="font-mono" style={{ fontSize: '0.9rem', color: 'var(--text-primary)' }}>
                        http://{lanData.host.host_ip}:5173
                      </div>
                    </div>
                    <div>
                      <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Scanner:</span>
                      <div className="font-mono" style={{ fontSize: '0.85rem', color: '#22c55e' }}>
                        {lanData.scanner?.scan_count || 0} scans · {lanData.scanner?.last_scan_duration_s || '?'}s/cycle
                      </div>
                    </div>
                    <span className="risk-badge risk-badge--low">Subnet Active</span>
                  </div>
                )}

                <table className="risk-table">
                  <thead>
                    <tr>
                      <th>Device ID</th>
                      <th>IP Address</th>
                      <th>MAC Address</th>
                      <th>Manufacturer</th>
                      <th>Role / Inferred Type</th>
                      <th>Status</th>
                      <th>Criticality</th>
                      <th>Dynamic Risk</th>
                    </tr>
                  </thead>
                  <tbody>
                    {lanData?.devices?.map((d) => (
                      <tr key={d.device_id} style={{
                        opacity: d.status === 'offline' ? 0.45 : 1,
                      }}>
                        <td className="font-mono" style={{ fontWeight: 600, color: d.is_host ? 'var(--accent-pink)' : d.status === 'new' ? '#22c55e' : 'var(--text-primary)' }}>
                          {d.status === 'new' && '🆕 '}{d.device_id}
                        </td>
                        <td className="font-mono" style={{ color: 'var(--accent-magenta)' }}>{d.ip_address}</td>
                        <td className="font-mono" style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{d.mac_address}</td>
                        <td style={{ color: 'var(--accent-cyan)', fontSize: '0.75rem' }}>{d.manufacturer || 'Unknown'}</td>
                        <td>{d.role}</td>
                        <td>
                          <span style={{
                            fontSize: '0.7rem',
                            fontWeight: 600,
                            color: d.status === 'online' ? '#22c55e' : d.status === 'new' ? '#3b82f6' : '#ef4444',
                          }}>
                            {d.status === 'new' ? '● NEW' : d.status === 'online' ? '● Online' : '○ Offline'}
                          </span>
                        </td>
                        <td className="font-mono">{(d.criticality * 100).toFixed(0)}%</td>
                        <td>
                          <span className={`risk-badge risk-badge--${d.risk_level}`}>
                            {d.dynamic_risk_score.toFixed(3)} ({d.risk_level})
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
