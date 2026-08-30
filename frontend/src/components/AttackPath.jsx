/**
 * AttackPath — Attack propagation path visualization.
 * Shows a step-by-step path from source to target with risk indicators
 * and an animated connector line.
 */
export default function AttackPath({ attackPath, selectedDevice, onDeviceSelect }) {
  const deviceId = selectedDevice?.id || selectedDevice?.device_id;

  if (!deviceId) {
    return (
      <div className="cyber-card">
        <div className="cyber-card__header">
          <div className="cyber-card__title">
            <span className="cyber-card__title-icon">⟿</span>
            Attack Propagation Path
          </div>
        </div>
        <div className="empty-state">
          <div className="empty-state__icon">⟿</div>
          <div className="empty-state__text">Select a device to trace attack path</div>
        </div>
      </div>
    );
  }

  const pathNodes = attackPath?.path || [];
  const totalSteps = attackPath?.total_steps || 0;

  return (
    <div className="cyber-card animate-fade-in">
      <div className="cyber-card__header">
        <div className="cyber-card__title">
          <span className="cyber-card__title-icon">⟿</span>
          Attack Propagation Path
        </div>
        <span className="cyber-card__badge" style={{
          background: 'var(--accent-purple-soft)',
          color: 'var(--accent-magenta)',
          border: '1px solid rgba(124, 58, 237, 0.3)',
        }}>
          {totalSteps} HOPS
        </span>
      </div>

      {pathNodes.length > 1 ? (
        <>
          {/* Path visualization */}
          <div className="attack-path">
            {pathNodes.map((node, i) => {
              const isSource = i === 0;
              const isTarget = i === pathNodes.length - 1;
              const nodeClass = isTarget
                ? 'attack-path__node--target'
                : isSource
                  ? 'attack-path__node--source'
                  : 'attack-path__node--hop';

              const riskPct = ((node.risk_score || 0) * 100).toFixed(1);
              const probPct = ((node.attack_probability || 0) * 100).toFixed(1);

              return (
                <div
                  key={node.device_id}
                  className="attack-path__step animate-fade-in"
                  style={{
                    animationDelay: `${i * 120}ms`,
                    cursor: 'pointer',
                    padding: 'var(--space-sm)',
                    borderRadius: 'var(--radius-md)',
                    transition: 'background var(--transition-fast)',
                  }}
                  onClick={() => onDeviceSelect?.({
                    id: node.device_id,
                    device_id: node.device_id,
                    type: node.device_type,
                    risk_score: node.risk_score,
                  })}
                  onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--accent-purple-soft)'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
                >
                  {/* Connector line (not on last node) */}
                  {i < pathNodes.length - 1 && (
                    <div className="attack-path__connector" />
                  )}

                  {/* Node circle */}
                  <div className={`attack-path__node ${nodeClass}`}>
                    {isSource ? 'S' : isTarget ? 'T' : node.step}
                  </div>

                  {/* Node details */}
                  <div className="attack-path__details">
                    <div className="attack-path__device-name">
                      {node.device_id}
                      {isSource && (
                        <span style={{ fontSize: '0.6rem', color: 'var(--accent-purple)', marginLeft: 6 }}>
                          SOURCE
                        </span>
                      )}
                      {isTarget && (
                        <span style={{ fontSize: '0.6rem', color: 'var(--accent-red)', marginLeft: 6 }}>
                          TARGET
                        </span>
                      )}
                    </div>
                    <div className="attack-path__device-meta">
                      {node.device_type || 'device'}
                      {' · '}
                      Risk: <span className="font-mono" style={{ color: getRiskColor(node.risk_score) }}>{riskPct}%</span>
                      {' · '}
                      Prob: <span className="font-mono">{probPct}%</span>
                    </div>
                  </div>

                  {/* Arrow indicator (not on last node) */}
                  {i < pathNodes.length - 1 && (
                    <div style={{
                      fontSize: '1rem',
                      color: 'var(--accent-red)',
                      opacity: 0.6,
                      flexShrink: 0,
                    }}>
                      ↓
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Description */}
          {attackPath?.description && (
            <div style={{
              marginTop: 'var(--space-md)',
              paddingTop: 'var(--space-md)',
              borderTop: '1px solid var(--border-subtle)',
              fontSize: '0.7rem',
              color: 'var(--text-muted)',
              lineHeight: 1.5,
              fontStyle: 'italic',
            }}>
              {attackPath.description}
            </div>
          )}
        </>
      ) : (
        <div className="empty-state" style={{ padding: 'var(--space-xl)' }}>
          <div className="empty-state__icon">◯</div>
          <div className="empty-state__text">
            {attackPath?.description || `No multi-hop attack path found for ${deviceId}`}
          </div>
        </div>
      )}
    </div>
  );
}

function getRiskColor(score) {
  if (score >= 0.75) return 'var(--risk-critical)';
  if (score >= 0.5) return 'var(--risk-high)';
  if (score >= 0.25) return 'var(--risk-medium)';
  return 'var(--risk-low)';
}
