import { useState, useEffect } from 'react';
import { fetchHealth } from '../services/api';

/**
 * Header — Top bar with health status, stats, and current time window info.
 */
export default function Header({ deviceCount, windowCount, currentWindowId }) {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    let mounted = true;
    const checkHealth = async () => {
      try {
        const data = await fetchHealth();
        if (mounted) setHealth(data);
      } catch {
        if (mounted) setHealth({ status: 'error' });
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => { mounted = false; clearInterval(interval); };
  }, []);

  const isHealthy = health?.status === 'healthy';

  return (
    <header className="header">
      <div className="header__left">
        <h1 className="header__title">
          <span className="text-gradient">Threat Intelligence</span> Dashboard
        </h1>
        <div className="header__stats">
          <div className="header__stat">
            <span>Devices:</span>
            <span className="header__stat-value">{deviceCount ?? '—'}</span>
          </div>
          <div className="header__stat">
            <span>Windows:</span>
            <span className="header__stat-value">{windowCount ?? '—'}</span>
          </div>
          {currentWindowId != null && (
            <div className="header__stat">
              <span>Active Window:</span>
              <span className="header__stat-value" style={{ color: 'var(--accent-magenta)' }}>
                W{currentWindowId}
              </span>
            </div>
          )}
        </div>
      </div>

      <div className="header__right">
        <div className="header__health">
          <span
            className={`header__health-dot ${!isHealthy ? 'header__health-dot--error' : ''}`}
          />
          <span>{isHealthy ? 'Backend Online' : 'Backend Offline'}</span>
        </div>
      </div>
    </header>
  );
}
