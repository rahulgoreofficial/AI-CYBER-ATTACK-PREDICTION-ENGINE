/**
 * PredictionPanel — Top-K predicted future attack targets.
 * Ranked list with attack probability bars, risk badges, and device metadata.
 */
export default function PredictionPanel({ predictions, selectedDevice, onDeviceSelect }) {
  if (!predictions || !predictions.predictions?.length) {
    return (
      <div className="cyber-card">
        <div className="cyber-card__header">
          <div className="cyber-card__title">
            <span className="cyber-card__title-icon">⎯⟫</span>
            Predicted Targets
          </div>
        </div>
        <div className="empty-state">
          <div className="empty-state__icon">◌</div>
          <div className="empty-state__text">No predictions available</div>
        </div>
      </div>
    );
  }

  const selectedId = selectedDevice?.id || selectedDevice?.device_id;

  return (
    <div className="cyber-card">
      <div className="cyber-card__header">
        <div className="cyber-card__title">
          <span className="cyber-card__title-icon">⎯⟫</span>
          Predicted Targets
        </div>
        <span className="cyber-card__badge" style={{
          background: 'var(--accent-red-soft)',
          color: 'var(--accent-red)',
          border: '1px solid rgba(220, 38, 38, 0.3)',
        }}>
          TOP {predictions.predictions.length}
        </span>
      </div>

      <div className="prediction-list">
        {predictions.predictions.map((pred) => {
          const isSelected = selectedId === pred.device_id;
          const rankClass = pred.rank <= 3
            ? `prediction-item__rank--${pred.rank}`
            : 'prediction-item__rank--default';
          const probPct = (pred.attack_probability * 100).toFixed(1);
          const probColor = getProbColor(pred.attack_probability);

          return (
            <div
              key={pred.device_id}
              className={`prediction-item animate-fade-in ${isSelected ? 'prediction-item--selected' : ''}`}
              style={{ animationDelay: `${(pred.rank - 1) * 60}ms` }}
              onClick={() => onDeviceSelect?.({
                id: pred.device_id,
                device_id: pred.device_id,
                type: pred.device_type || '',
                department: pred.department || '',
                risk_score: pred.risk_score,
                risk_level: pred.risk_level,
                criticality: pred.criticality,
              })}
            >
              <div className={`prediction-item__rank ${rankClass}`}>
                #{pred.rank}
              </div>

              <div className="prediction-item__info">
                <div className="prediction-item__name">{pred.device_id}</div>
                <div className="prediction-item__type">
                  {pred.device_type || 'device'}
                  {pred.department ? ` · ${pred.department}` : ''}
                </div>
              </div>

              <div className="prediction-item__prob">
                <div className="prediction-item__prob-value" style={{ color: probColor }}>
                  {probPct}%
                </div>
                <div className="prediction-item__prob-bar">
                  <div
                    className="prediction-item__prob-fill"
                    style={{ width: `${Math.max(probPct, 3)}%` }}
                  />
                </div>
              </div>

              {pred.risk_level && (
                <span className={`risk-badge risk-badge--${pred.risk_level}`}>
                  {pred.risk_level}
                </span>
              )}
            </div>
          );
        })}
      </div>

      <div style={{
        marginTop: 'var(--space-md)',
        paddingTop: 'var(--space-sm)',
        borderTop: '1px solid var(--border-subtle)',
        display: 'flex',
        justifyContent: 'space-between',
        fontSize: '0.65rem',
        color: 'var(--text-muted)',
      }}>
        <span>Model: <span className="font-mono" style={{ color: 'var(--accent-magenta)' }}>{predictions.model?.toUpperCase()}</span></span>
        <span>Feed: <span className="font-mono" style={{ color: '#22c55e' }}>● Real-Time Stream</span></span>
      </div>
    </div>
  );
}

function getProbColor(prob) {
  if (prob >= 0.8) return 'var(--risk-critical)';
  if (prob >= 0.6) return 'var(--risk-high)';
  if (prob >= 0.3) return 'var(--risk-medium)';
  return 'var(--risk-low)';
}
