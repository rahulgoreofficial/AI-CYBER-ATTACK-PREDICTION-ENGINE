/**
 * TimelineSelector — Horizontal scrollable timeline of analysis windows.
 * Each window shows its ID, device count, and attack indicator.
 */
export default function TimelineSelector({ timeline, currentWindowId, onWindowChange }) {
  if (!timeline || !timeline.windows?.length) {
    return (
      <div className="cyber-card" style={{ padding: 'var(--space-md) var(--space-xl)' }}>
        <div className="cyber-card__title" style={{ marginBottom: 0 }}>
          <span className="cyber-card__title-icon">◷</span>
          Timeline
        </div>
      </div>
    );
  }

  return (
    <div className="cyber-card" style={{ padding: 'var(--space-md) var(--space-xl)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-lg)', marginBottom: 'var(--space-sm)' }}>
        <div className="cyber-card__title" style={{ marginBottom: 0 }}>
          <span className="cyber-card__title-icon">◷</span>
          Analysis Timeline
        </div>
        <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>
          {timeline.total_windows} windows available
        </span>
      </div>
      <div className="timeline">
        {timeline.windows.map((w) => (
          <div
            key={w.window_id}
            className={`timeline__item ${
              w.window_id === currentWindowId ? 'timeline__item--active' : ''
            } ${w.has_attack ? 'timeline__item--attack' : ''}`}
            onClick={() => onWindowChange(w.window_id)}
            title={`Window ${w.window_id} · ${w.device_count} devices${w.has_attack ? ' · ATTACKS DETECTED' : ''}`}
          >
            <span className="timeline__window-id">W{w.window_id}</span>
            <span className="timeline__device-count">{w.device_count} dev</span>
          </div>
        ))}
      </div>
    </div>
  );
}
