import { useState, useMemo } from 'react';

/**
 * RiskTable — Full device risk dashboard with sortable columns,
 * risk-level badges, and heatmap intensity backgrounds.
 */
export default function RiskTable({ riskData, selectedDevice, onDeviceSelect }) {
  const [sortKey, setSortKey] = useState('dynamic_risk_score');
  const [sortDir, setSortDir] = useState('desc');

  const columns = [
    { key: 'risk_rank', label: '#', width: '40px' },
    { key: 'device_id', label: 'Device', width: 'auto' },
    { key: 'dynamic_risk_score', label: 'Risk Score', width: '90px' },
    { key: 'attack_probability', label: 'Attack Prob', width: '90px' },
    { key: 'anomaly_score', label: 'Anomaly', width: '80px' },
    { key: 'asset_criticality', label: 'Criticality', width: '80px' },
    { key: 'vulnerability_score', label: 'Vuln', width: '70px' },
    { key: 'risk_level', label: 'Level', width: '90px' },
  ];

  const entries = riskData?.entries || [];

  const sortedEntries = useMemo(() => {
    if (!entries.length) return [];
    return [...entries].sort((a, b) => {
      let aVal = a[sortKey];
      let bVal = b[sortKey];
      if (typeof aVal === 'string') aVal = aVal.toLowerCase();
      if (typeof bVal === 'string') bVal = bVal.toLowerCase();
      if (aVal < bVal) return sortDir === 'asc' ? -1 : 1;
      if (aVal > bVal) return sortDir === 'asc' ? 1 : -1;
      return 0;
    });
  }, [entries, sortKey, sortDir]);

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortDir('desc');
    }
  };

  const selectedId = selectedDevice?.id || selectedDevice?.device_id;

  if (!entries.length) {
    return (
      <div className="cyber-card">
        <div className="cyber-card__header">
          <div className="cyber-card__title">
            <span className="cyber-card__title-icon">⚠</span>
            Device Risk Analysis
          </div>
        </div>
        <div className="empty-state">
          <div className="empty-state__icon">◌</div>
          <div className="empty-state__text">No risk data available</div>
        </div>
      </div>
    );
  }

  return (
    <div className="cyber-card" style={{ padding: 'var(--space-lg)' }}>
      <div className="cyber-card__header">
        <div className="cyber-card__title">
          <span className="cyber-card__title-icon">⚠</span>
          Device Risk Analysis
        </div>
        <span className="cyber-card__badge" style={{
          background: 'var(--accent-purple-soft)',
          color: 'var(--accent-magenta)',
          border: '1px solid rgba(124, 58, 237, 0.3)',
        }}>
          {entries.length} DEVICES
        </span>
      </div>

      <div className="risk-table-wrapper" style={{ maxHeight: '320px', overflowY: 'auto' }}>
        <table className="risk-table">
          <thead>
            <tr>
              {columns.map((col) => (
                <th
                  key={col.key}
                  style={{ width: col.width }}
                  className={sortKey === col.key ? 'th--sorted' : ''}
                  onClick={() => handleSort(col.key)}
                >
                  {col.label}
                  {sortKey === col.key && (
                    <span style={{ marginLeft: 4, fontSize: '0.6rem' }}>
                      {sortDir === 'asc' ? '▲' : '▼'}
                    </span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sortedEntries.map((entry) => {
              const isSelected = selectedId === entry.device_id;
              return (
                <tr
                  key={entry.device_id}
                  className={isSelected ? 'tr--selected' : ''}
                  onClick={() => onDeviceSelect?.({
                    id: entry.device_id,
                    device_id: entry.device_id,
                    risk_score: entry.dynamic_risk_score,
                    risk_level: entry.risk_level,
                    type: '',
                    department: '',
                  })}
                >
                  <td>
                    <span className="font-mono" style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                      {entry.risk_rank}
                    </span>
                  </td>
                  <td>
                    <span className="risk-table__device">{entry.device_id}</span>
                  </td>
                  <td>
                    <span
                      className="risk-table__score"
                      style={{
                        color: getRiskColor(entry.dynamic_risk_score),
                        textShadow: entry.dynamic_risk_score > 0.7
                          ? `0 0 8px ${getRiskColor(entry.dynamic_risk_score)}40`
                          : 'none',
                      }}
                    >
                      {(entry.dynamic_risk_score * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td>
                    <span className="risk-table__score" style={{ color: 'var(--text-secondary)' }}>
                      {(entry.attack_probability * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td>
                    <ScoreBar value={entry.anomaly_score} />
                  </td>
                  <td>
                    <ScoreBar value={entry.asset_criticality} color="var(--accent-purple)" />
                  </td>
                  <td>
                    <ScoreBar value={entry.vulnerability_score} color="var(--accent-pink)" />
                  </td>
                  <td>
                    <span className={`risk-badge risk-badge--${entry.risk_level || 'low'}`}>
                      {entry.risk_level || 'low'}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/** Mini inline bar for numeric scores */
function ScoreBar({ value, color = 'var(--accent-red)' }) {
  const pct = ((value || 0) * 100).toFixed(0);
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <div style={{
        width: 40,
        height: 4,
        background: 'rgba(113, 113, 122, 0.15)',
        borderRadius: 'var(--radius-full)',
        overflow: 'hidden',
      }}>
        <div style={{
          width: `${pct}%`,
          height: '100%',
          background: color,
          borderRadius: 'var(--radius-full)',
          transition: 'width var(--transition-slow)',
        }} />
      </div>
      <span className="font-mono" style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>
        {pct}
      </span>
    </div>
  );
}

function getRiskColor(score) {
  if (score >= 0.75) return 'var(--risk-critical)';
  if (score >= 0.5) return 'var(--risk-high)';
  if (score >= 0.25) return 'var(--risk-medium)';
  return 'var(--risk-low)';
}
