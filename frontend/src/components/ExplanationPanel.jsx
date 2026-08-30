import { useMemo } from 'react';

/**
 * ExplanationPanel — SHAP-based feature explanation for the selected device.
 * Shows a horizontal waterfall chart of top contributing features with
 * direction indicators (increases risk / decreases risk).
 */
export default function ExplanationPanel({ explanation, selectedDevice }) {
  const deviceId = selectedDevice?.id || selectedDevice?.device_id;

  // Aggregate top features from the most recent explanation entry
  const topFeatures = useMemo(() => {
    if (!explanation?.explanations?.length) return [];
    // Use the first explanation (most recent window)
    const expl = explanation.explanations[0];
    if (!expl?.top_features?.length) return [];
    return expl.top_features.slice(0, 8);
  }, [explanation]);

  const globalImportance = explanation?.global_importance?.slice(0, 6) || [];

  // Max absolute SHAP value for scaling bars
  const maxAbsShap = useMemo(() => {
    if (!topFeatures.length) return 1;
    return Math.max(...topFeatures.map((f) => Math.abs(f.shap_value)), 0.001);
  }, [topFeatures]);

  if (!deviceId) {
    return (
      <div className="cyber-card">
        <div className="cyber-card__header">
          <div className="cyber-card__title">
            <span className="cyber-card__title-icon">◐</span>
            Feature Explanations
          </div>
        </div>
        <div className="empty-state">
          <div className="empty-state__icon">◐</div>
          <div className="empty-state__text">Select a device to view explanations</div>
        </div>
      </div>
    );
  }

  return (
    <div className="cyber-card animate-fade-in">
      <div className="cyber-card__header">
        <div className="cyber-card__title">
          <span className="cyber-card__title-icon">◐</span>
          SHAP Explanation
        </div>
        <span className="font-mono" style={{
          fontSize: '0.7rem',
          color: 'var(--accent-magenta)',
        }}>
          {deviceId}
        </span>
      </div>

      {/* Attack probability summary */}
      {explanation?.explanations?.[0] && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 'var(--space-lg)',
          marginBottom: 'var(--space-lg)',
          padding: 'var(--space-md)',
          borderRadius: 'var(--radius-md)',
          background: 'var(--accent-red-soft)',
          border: '1px solid rgba(220, 38, 38, 0.2)',
        }}>
          <div>
            <div style={{ fontSize: '0.6rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              Attack Probability
            </div>
            <div className="font-mono" style={{
              fontSize: '1.2rem',
              fontWeight: 700,
              color: 'var(--accent-red)',
            }}>
              {(explanation.explanations[0].attack_probability * 100).toFixed(1)}%
            </div>
          </div>
          <div>
            <div style={{ fontSize: '0.6rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              Base Value
            </div>
            <div className="font-mono" style={{
              fontSize: '1.2rem',
              fontWeight: 700,
              color: 'var(--text-secondary)',
            }}>
              {(explanation.explanations[0].base_value * 100).toFixed(1)}%
            </div>
          </div>
          <div style={{ marginLeft: 'auto', fontSize: '0.6rem', color: 'var(--text-muted)' }}>
            W{explanation.explanations[0].window_id}
          </div>
        </div>
      )}

      {/* Feature waterfall bars */}
      {topFeatures.length > 0 ? (
        <div className="explanation-features">
          {topFeatures.map((feat, i) => {
            const isPositive = feat.shap_value > 0 || feat.direction === 'increases_risk';
            const barWidth = Math.max((Math.abs(feat.shap_value) / maxAbsShap) * 100, 4);
            const contribPct = feat.contribution_pct ? feat.contribution_pct.toFixed(1) : null;

            return (
              <div
                key={feat.name}
                className="explanation-feature animate-fade-in"
                style={{ animationDelay: `${i * 50}ms` }}
              >
                <div className="explanation-feature__name" title={feat.name}>
                  {formatFeatureName(feat.name)}
                </div>
                <div className="explanation-feature__bar-container">
                  <div
                    className={`explanation-feature__bar ${
                      isPositive ? 'explanation-feature__bar--positive' : 'explanation-feature__bar--negative'
                    }`}
                    style={{ width: `${barWidth}%` }}
                  />
                  <span className="explanation-feature__value">
                    {isPositive ? '+' : ''}{feat.shap_value.toFixed(4)}
                    {contribPct ? ` (${contribPct}%)` : ''}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="empty-state" style={{ padding: 'var(--space-xl)' }}>
          <div className="empty-state__text">No feature explanations available for this device</div>
        </div>
      )}

      {/* Global importance */}
      {globalImportance.length > 0 && (
        <div style={{ marginTop: 'var(--space-lg)', paddingTop: 'var(--space-md)', borderTop: '1px solid var(--border-subtle)' }}>
          <div style={{
            fontSize: '0.65rem',
            fontWeight: 600,
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
            color: 'var(--text-muted)',
            marginBottom: 'var(--space-sm)',
          }}>
            Global Feature Importance
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 8px' }}>
            {globalImportance.map((g) => (
              <span
                key={g.feature}
                style={{
                  fontSize: '0.62rem',
                  padding: '2px 8px',
                  borderRadius: 'var(--radius-full)',
                  background: 'var(--accent-purple-soft)',
                  color: 'var(--text-secondary)',
                  border: '1px solid var(--border-subtle)',
                }}
                title={`Mean |SHAP|: ${g.mean_abs_shap.toFixed(4)}`}
              >
                #{g.rank} {formatFeatureName(g.feature)}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function formatFeatureName(name) {
  if (!name) return '';
  return name
    .replace(/_/g, ' ')
    .replace(/\b(dst|src|fwd|bwd|pkt|avg|std|min|max)\b/gi, (m) => m.toUpperCase())
    .slice(0, 24);
}
