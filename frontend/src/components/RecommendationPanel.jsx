/**
 * RecommendationPanel — Defensive action cards for the selected device.
 * Shows prioritized recommendations with category badges and reason text.
 */
export default function RecommendationPanel({ recommendations, selectedDevice }) {
  const deviceId = selectedDevice?.id || selectedDevice?.device_id;

  const priorityConfig = {
    critical: { color: 'var(--risk-critical)', icon: '🔴', bg: 'rgba(239, 68, 68, 0.08)', border: 'rgba(239, 68, 68, 0.25)' },
    high: { color: 'var(--risk-high)', icon: '🟠', bg: 'rgba(249, 115, 22, 0.08)', border: 'rgba(249, 115, 22, 0.25)' },
    medium: { color: 'var(--risk-medium)', icon: '🟡', bg: 'rgba(234, 179, 8, 0.08)', border: 'rgba(234, 179, 8, 0.25)' },
    low: { color: 'var(--risk-low)', icon: '🟢', bg: 'rgba(34, 197, 94, 0.08)', border: 'rgba(34, 197, 94, 0.25)' },
  };

  const categoryIcons = {
    incident_response: '🚨',
    isolation: '🔒',
    access_control: '🔑',
    monitoring: '📡',
    patching: '🩹',
    micro_segmentation: '🧱',
    data_protection: '💾',
  };

  if (!deviceId) {
    return (
      <div className="cyber-card">
        <div className="cyber-card__header">
          <div className="cyber-card__title">
            <span className="cyber-card__title-icon">🛡</span>
            Defensive Actions
          </div>
        </div>
        <div className="empty-state">
          <div className="empty-state__icon">🛡</div>
          <div className="empty-state__text">Select a device to view recommendations</div>
        </div>
      </div>
    );
  }

  const items = recommendations?.recommendations || [];

  return (
    <div className="cyber-card animate-fade-in">
      <div className="cyber-card__header">
        <div className="cyber-card__title">
          <span className="cyber-card__title-icon">🛡</span>
          Defensive Actions
        </div>
        <span className="cyber-card__badge" style={{
          background: 'var(--accent-red-soft)',
          color: 'var(--accent-red)',
          border: '1px solid rgba(220, 38, 38, 0.3)',
        }}>
          {items.length} ACTIONS
        </span>
      </div>

      {items.length > 0 ? (
        <div className="recommendation-list">
          {items.map((rec, i) => {
            const pConfig = priorityConfig[rec.priority] || priorityConfig.medium;
            const catIcon = categoryIcons[rec.category] || '📋';

            return (
              <div
                key={i}
                className="recommendation-card animate-fade-in"
                style={{
                  animationDelay: `${i * 60}ms`,
                  background: pConfig.bg,
                  borderColor: pConfig.border,
                }}
              >
                <div className="recommendation-card__header">
                  <span className={`risk-badge risk-badge--${rec.priority || 'medium'}`}>
                    {rec.priority || 'medium'}
                  </span>
                  <span className="recommendation-card__category">
                    {catIcon} {(rec.category || 'general').replace(/_/g, ' ')}
                  </span>
                </div>
                <div className="recommendation-card__action">{rec.action}</div>
                {rec.reason && (
                  <div className="recommendation-card__reason">{rec.reason}</div>
                )}
              </div>
            );
          })}
        </div>
      ) : (
        <div className="empty-state" style={{ padding: 'var(--space-xl)' }}>
          <div className="empty-state__text">No recommendations for this device</div>
        </div>
      )}
    </div>
  );
}
