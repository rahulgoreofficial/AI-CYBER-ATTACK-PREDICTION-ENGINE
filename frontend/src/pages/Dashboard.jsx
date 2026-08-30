import { useState, useEffect, useCallback } from 'react';
import {
  fetchNetwork,
  fetchRisk,
  fetchPredictions,
  fetchTimeline,
  fetchEvaluation,
  fetchExplanation,
  fetchRecommendations,
  fetchAttackPath,
  triggerAnalysis,
} from '../services/api';

import TimelineSelector from '../components/TimelineSelector';
import NetworkGraph from '../components/NetworkGraph';
import PredictionPanel from '../components/PredictionPanel';
import RiskTable from '../components/RiskTable';
import MetricsPanel from '../components/MetricsPanel';
import ExplanationPanel from '../components/ExplanationPanel';
import RecommendationPanel from '../components/RecommendationPanel';
import AttackPath from '../components/AttackPath';

/**
 * Dashboard — Main page assembling all dashboard panels in a grid layout.
 * Supports multi-view switching based on activeSection.
 */
export default function Dashboard({
  selectedDevice,
  onDeviceSelect,
  onDataLoaded,
  currentWindowId,
  onWindowChange,
  activeSection = 'dashboard',
}) {
  // Data state
  const [timeline, setTimeline] = useState(null);
  const [networkData, setNetworkData] = useState(null);
  const [riskData, setRiskData] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [recommendations, setRecommendations] = useState(null);
  const [attackPath, setAttackPath] = useState(null);

  // Model & Analysis config
  const [selectedModel, setSelectedModel] = useState('xgboost');
  const [topK, setTopK] = useState(5);
  const [analyzing, setAnalyzing] = useState(false);

  // Loading states
  const [loadingMain, setLoadingMain] = useState(true);
  const [loadingDevice, setLoadingDevice] = useState(false);

  // Bottom panel tab
  const [bottomTab, setBottomTab] = useState('risk');

  // ── Load initial data (timeline, evaluation) ────────────────────────────
  useEffect(() => {
    const loadInitial = async () => {
      try {
        const [timelineData, evalData] = await Promise.all([
          fetchTimeline(),
          fetchEvaluation(),
        ]);
        setTimeline(timelineData);
        setEvaluation(evalData);

        // Set initial window to first attack window or latest
        if (timelineData?.windows?.length) {
          const attackWindows = timelineData.windows.filter((w) => w.has_attack);
          const initialWindow = attackWindows.length > 0
            ? attackWindows[0].window_id
            : timelineData.windows[0].window_id;
          onWindowChange(initialWindow);
        }
      } catch (err) {
        console.error('Failed to load initial data:', err);
      }
    };
    loadInitial();
  }, []);

  // ── Load window-specific data ───────────────────────────────────────────
  useEffect(() => {
    if (currentWindowId == null) return;

    const loadWindowData = async () => {
      setLoadingMain(true);
      try {
        const [netData, riskResp, predResp] = await Promise.all([
          fetchNetwork(currentWindowId),
          fetchRisk(currentWindowId),
          fetchPredictions(currentWindowId, topK, selectedModel),
        ]);
        setNetworkData(netData);
        setRiskData(riskResp);
        setPredictions(predResp);

        // Auto-select #1 predicted device if none selected
        if (!selectedDevice && predResp?.predictions?.length > 0) {
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

        // Notify parent of stats
        onDataLoaded?.({
          deviceCount: netData?.total_nodes,
          windowCount: timeline?.total_windows,
        });
      } catch (err) {
        console.error('Failed to load window data:', err);
      } finally {
        setLoadingMain(false);
      }
    };
    loadWindowData();
  }, [currentWindowId, selectedModel, topK]);

  // ── Load device-specific data ───────────────────────────────────────────
  useEffect(() => {
    const deviceId = selectedDevice?.id || selectedDevice?.device_id;
    if (!deviceId) {
      setExplanation(null);
      setRecommendations(null);
      setAttackPath(null);
      return;
    }

    const loadDeviceData = async () => {
      setLoadingDevice(true);
      try {
        const [explResp, recResp, pathResp] = await Promise.all([
          fetchExplanation(deviceId, currentWindowId),
          fetchRecommendations(deviceId, currentWindowId),
          fetchAttackPath(deviceId, currentWindowId),
        ]);
        setExplanation(explResp);
        setRecommendations(recResp);
        setAttackPath(pathResp);
      } catch (err) {
        console.error('Failed to load device data:', err);
        setExplanation(null);
        setRecommendations(null);
        setAttackPath(null);
      } finally {
        setLoadingDevice(false);
      }
    };
    loadDeviceData();
  }, [selectedDevice, currentWindowId]);

  // ── Trigger Live Analysis ───────────────────────────────────────────────
  const handleTriggerAnalysis = async () => {
    if (currentWindowId == null) return;
    setAnalyzing(true);
    try {
      const result = await triggerAnalysis(currentWindowId, selectedModel, topK);
      if (result.predictions) {
        setPredictions({
          window_id: result.window_id,
          model: result.model,
          top_k: topK,
          predictions: result.predictions,
        });
      }
      if (result.risk_scores) {
        setRiskData({
          window_id: result.window_id,
          entries: result.risk_scores,
          total_devices: result.total_devices,
        });
      }
    } catch (err) {
      console.error('Analysis failed:', err);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleDeviceSelect = useCallback((device) => {
    onDeviceSelect(device);
  }, [onDeviceSelect]);

  // ── Dedicated View Renderers based on Sidebar Selection ─────────────────

  if (activeSection === 'network') {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)', height: '100%' }}>
        <TimelineSelector timeline={timeline} currentWindowId={currentWindowId} onWindowChange={onWindowChange} />
        <div style={{ flex: 1, minHeight: '600px' }}>
          <NetworkGraph networkData={networkData} selectedDevice={selectedDevice} onDeviceSelect={handleDeviceSelect} attackPath={attackPath} />
        </div>
      </div>
    );
  }

  if (activeSection === 'risk') {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)' }}>
        <TimelineSelector timeline={timeline} currentWindowId={currentWindowId} onWindowChange={onWindowChange} />
        <RiskTable riskData={riskData} selectedDevice={selectedDevice} onDeviceSelect={handleDeviceSelect} />
      </div>
    );
  }

  if (activeSection === 'models') {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)' }}>
        <MetricsPanel evaluation={evaluation} />
      </div>
    );
  }

  if (activeSection === 'explanation') {
    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-lg)' }}>
        <PredictionPanel predictions={predictions} selectedDevice={selectedDevice} onDeviceSelect={handleDeviceSelect} />
        <ExplanationPanel explanation={explanation} selectedDevice={selectedDevice} />
      </div>
    );
  }

  if (activeSection === 'attack-path') {
    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: 'var(--space-lg)', height: '100%' }}>
        <div style={{ minHeight: '550px' }}>
          <NetworkGraph networkData={networkData} selectedDevice={selectedDevice} onDeviceSelect={handleDeviceSelect} attackPath={attackPath} />
        </div>
        <AttackPath attackPath={attackPath} selectedDevice={selectedDevice} onDeviceSelect={handleDeviceSelect} />
      </div>
    );
  }

  if (activeSection === 'recommendations') {
    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-lg)' }}>
        <RiskTable riskData={riskData} selectedDevice={selectedDevice} onDeviceSelect={handleDeviceSelect} />
        <RecommendationPanel recommendations={recommendations} selectedDevice={selectedDevice} />
      </div>
    );
  }

  // ── Default Dashboard View (Full Grid) ──────────────────────────────────
  return (
    <div className="dashboard">
      {/* Row 1 — Timeline & Action Bar */}
      <div className="dashboard__timeline">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 'var(--space-md)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-md)' }}>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Model:</span>
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
                  fontSize: '0.75rem',
                  outline: 'none',
                }}
              >
                <option value="xgboost">XGBoost Baseline</option>
                <option value="gnn">Graph Neural Network (GNN)</option>
                <option value="temporal">Temporal LSTM</option>
              </select>

              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginLeft: 8 }}>Top-K:</span>
              <select
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                style={{
                  background: 'var(--bg-card)',
                  color: 'var(--text-primary)',
                  border: '1px solid var(--border-medium)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '3px 8px',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.75rem',
                  outline: 'none',
                }}
              >
                <option value={3}>Top 3</option>
                <option value={5}>Top 5</option>
                <option value={10}>Top 10</option>
              </select>
            </div>

            <button
              className="cyber-btn cyber-btn--primary cyber-btn--sm"
              onClick={handleTriggerAnalysis}
              disabled={analyzing}
            >
              {analyzing ? 'Scanning...' : '⚡ Run AI Prediction Scan'}
            </button>
          </div>

          <TimelineSelector
            timeline={timeline}
            currentWindowId={currentWindowId}
            onWindowChange={onWindowChange}
          />
        </div>
      </div>

      {/* Main Area — Network Graph */}
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

      {/* Right Panels — Predictions + Explanation + Attack Path + Recommendations */}
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

      {/* Bottom — Risk Table / Metrics (tabbed) */}
      <div className="dashboard__bottom">
        <div className="cyber-card" style={{ padding: 0 }}>
          <div style={{ padding: 'var(--space-lg) var(--space-lg) 0' }}>
            <div className="cyber-tabs">
              <button
                className={`cyber-tab ${bottomTab === 'risk' ? 'cyber-tab--active' : ''}`}
                onClick={() => setBottomTab('risk')}
              >
                ⚠ Risk Analysis
              </button>
              <button
                className={`cyber-tab ${bottomTab === 'metrics' ? 'cyber-tab--active' : ''}`}
                onClick={() => setBottomTab('metrics')}
              >
                ◈ Model Metrics
              </button>
            </div>
          </div>

          <div style={{ padding: '0 var(--space-sm) var(--space-sm)' }}>
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
    </div>
  );
}
