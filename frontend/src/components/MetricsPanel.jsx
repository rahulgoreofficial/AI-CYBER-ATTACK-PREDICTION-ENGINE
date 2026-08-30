import { useState } from 'react';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  Legend,
} from 'recharts';

/**
 * MetricsPanel — Model performance comparison with metric cards,
 * radar chart, and bar chart visualization.
 */
export default function MetricsPanel({ evaluation }) {
  const [view, setView] = useState('cards');

  if (!evaluation || !evaluation.models?.length) {
    return (
      <div className="cyber-card">
        <div className="cyber-card__header">
          <div className="cyber-card__title">
            <span className="cyber-card__title-icon">◈</span>
            Model Performance
          </div>
        </div>
        <div className="empty-state">
          <div className="empty-state__icon">◌</div>
          <div className="empty-state__text">No evaluation data available</div>
        </div>
      </div>
    );
  }

  const { models, best_model, best_f1 } = evaluation;

  // Colors for models
  const modelColors = [
    '#dc2626', '#7c3aed', '#ec4899', '#a855f7', '#d946ef', '#6366f1',
  ];

  // Radar data
  const radarMetrics = ['top_1_hit_rate', 'top_3_hit_rate', 'top_5_hit_rate', 'mrr', 'f1', 'roc_auc'];
  const radarLabels = { top_1_hit_rate: 'Top-1', top_3_hit_rate: 'Top-3', top_5_hit_rate: 'Top-5', mrr: 'MRR', f1: 'F1', roc_auc: 'ROC-AUC' };
  const radarData = radarMetrics.map((metric) => {
    const entry = { metric: radarLabels[metric] };
    models.forEach((m) => { entry[m.model] = m[metric] || 0; });
    return entry;
  });

  // Bar data — F1 comparison
  const barData = models.map((m, i) => ({
    name: shortenName(m.model),
    f1: m.f1,
    roc_auc: m.roc_auc,
    color: modelColors[i % modelColors.length],
  }));

  return (
    <div className="cyber-card">
      <div className="cyber-card__header">
        <div className="cyber-card__title">
          <span className="cyber-card__title-icon">◈</span>
          Model Performance
        </div>
        <span className="cyber-card__badge" style={{
          background: 'rgba(234, 179, 8, 0.15)',
          color: 'var(--risk-medium)',
          border: '1px solid rgba(234, 179, 8, 0.3)',
        }}>
          ★ {best_model}
        </span>
      </div>

      {/* View Tabs */}
      <div className="cyber-tabs">
        <button
          className={`cyber-tab ${view === 'cards' ? 'cyber-tab--active' : ''}`}
          onClick={() => setView('cards')}
        >
          Metric Cards
        </button>
        <button
          className={`cyber-tab ${view === 'radar' ? 'cyber-tab--active' : ''}`}
          onClick={() => setView('radar')}
        >
          Radar Comparison
        </button>
        <button
          className={`cyber-tab ${view === 'bar' ? 'cyber-tab--active' : ''}`}
          onClick={() => setView('bar')}
        >
          Bar Chart
        </button>
      </div>

      {/* Metric Cards View */}
      {view === 'cards' && (
        <div className="metrics-grid">
          {models.map((m, i) => {
            const isBest = m.model === best_model;
            return (
              <div
                key={m.model}
                className={`metric-card animate-fade-in ${isBest ? 'metric-card--best' : ''}`}
                style={{ animationDelay: `${i * 80}ms` }}
              >
                <div className="metric-card__name">
                  {isBest && <span style={{ marginRight: 4 }}>★</span>}
                  {shortenName(m.model)}
                </div>
                <div className="metric-card__value">{(m.f1 * 100).toFixed(1)}%</div>
                <div className="metric-card__sub">F1 Score</div>
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr',
                  gap: '4px 12px',
                  marginTop: 8,
                  fontSize: '0.6rem',
                  color: 'var(--text-muted)',
                }}>
                  <span>Top-1: <span className="font-mono" style={{ color: 'var(--text-secondary)' }}>{(m.top_1_hit_rate * 100).toFixed(0)}%</span></span>
                  <span>Top-3: <span className="font-mono" style={{ color: 'var(--text-secondary)' }}>{(m.top_3_hit_rate * 100).toFixed(0)}%</span></span>
                  <span>MRR: <span className="font-mono" style={{ color: 'var(--text-secondary)' }}>{m.mrr.toFixed(3)}</span></span>
                  <span>AUC: <span className="font-mono" style={{ color: 'var(--text-secondary)' }}>{m.roc_auc.toFixed(3)}</span></span>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Radar Chart View */}
      {view === 'radar' && (
        <div style={{ width: '100%', height: 300 }}>
          <ResponsiveContainer>
            <RadarChart data={radarData}>
              <PolarGrid stroke="rgba(124, 58, 237, 0.15)" />
              <PolarAngleAxis
                dataKey="metric"
                tick={{ fill: '#a1a1aa', fontSize: 11, fontFamily: 'Inter' }}
              />
              <PolarRadiusAxis
                domain={[0, 1]}
                tick={{ fill: '#71717a', fontSize: 9 }}
                axisLine={false}
              />
              {models.map((m, i) => (
                <Radar
                  key={m.model}
                  name={shortenName(m.model)}
                  dataKey={m.model}
                  stroke={modelColors[i % modelColors.length]}
                  fill={modelColors[i % modelColors.length]}
                  fillOpacity={0.08}
                  strokeWidth={1.5}
                />
              ))}
              <Legend
                wrapperStyle={{ fontSize: 10, color: '#a1a1aa', fontFamily: 'Inter' }}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Bar Chart View */}
      {view === 'bar' && (
        <div style={{ width: '100%', height: 260 }}>
          <ResponsiveContainer>
            <BarChart data={barData} barGap={4}>
              <XAxis
                dataKey="name"
                tick={{ fill: '#a1a1aa', fontSize: 10, fontFamily: 'Inter' }}
                axisLine={{ stroke: 'rgba(124, 58, 237, 0.15)' }}
                tickLine={false}
              />
              <YAxis
                domain={[0, 1]}
                tick={{ fill: '#71717a', fontSize: 9 }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                contentStyle={{
                  background: '#101018',
                  border: '1px solid rgba(124, 58, 237, 0.3)',
                  borderRadius: 8,
                  fontSize: 12,
                  fontFamily: 'Inter',
                  color: '#e4e4e7',
                }}
              />
              <Bar dataKey="f1" name="F1 Score" radius={[4, 4, 0, 0]}>
                {barData.map((entry, idx) => (
                  <Cell key={idx} fill={entry.color} fillOpacity={0.8} />
                ))}
              </Bar>
              <Bar dataKey="roc_auc" name="ROC-AUC" radius={[4, 4, 0, 0]} fillOpacity={0.4}>
                {barData.map((entry, idx) => (
                  <Cell key={idx} fill={entry.color} fillOpacity={0.4} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

function shortenName(name) {
  if (!name) return '';
  return name.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}
