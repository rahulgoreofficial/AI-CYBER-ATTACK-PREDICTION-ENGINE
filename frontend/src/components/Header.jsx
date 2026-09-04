import { useState, useEffect } from 'react';
import { fetchHealth } from '../services/api';

/**
 * Header — Live SOC top bar with health status, real-time clock,
 * monitored device count, and network accessibility badge.
 */
export default function Header({ deviceCount }) {
  const [health, setHealth] = useState(null);
  const [currentTime, setCurrentTime] = useState(new Date().toLocaleTimeString());

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

  // Update real-time clock every second
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date().toLocaleTimeString());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const isHealthy = health?.status === 'healthy';
  const currentHost = typeof window !== 'undefined' ? window.location.hostname : 'localhost';

  return (
    <header className="header">
      <div className="header__left">
        <h1 className="header__title">
          <span className="text-gradient">Threat Intelligence</span> Dashboard
        </h1>

        <div className="header__stats">
          {/* Live SOC Monitor Badge */}
          <div
            className="header__stat"
            style={{
              background: 'rgba(34, 197, 94, 0.12)',
              border: '1px solid rgba(34, 197, 94, 0.35)',
              padding: '2px 8px',
              borderRadius: 'var(--radius-sm)',
            }}
          >
            <span
              style={{
                display: 'inline-block',
                width: '7px',
                height: '7px',
                borderRadius: '50%',
                background: '#22c55e',
                boxShadow: '0 0 8px #22c55e',
                marginRight: '6px',
              }}
            />
            <span style={{ color: '#22c55e', fontWeight: 600, fontSize: '0.68rem', letterSpacing: '0.06em' }}>
              LIVE MONITORING
            </span>
          </div>

          <div className="header__stat">
            <span>Network Devices:</span>
            <span className="header__stat-value">{deviceCount ?? 21} Active</span>
          </div>

          <div className="header__stat">
            <span>SOC Time:</span>
            <span className="header__stat-value font-mono" style={{ color: 'var(--accent-pink)' }}>
              {currentTime}
            </span>
          </div>
        </div>
      </div>

      <div className="header__right">
        {/* Network & LAN Host Badge */}
        <div
          className="header__stat"
          style={{
            background: 'var(--bg-card)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-sm)',
            padding: '3px 10px',
            fontSize: '0.7rem',
          }}
          title="This dashboard and API are accessible to any device connected to the same Wi-Fi/Ethernet network"
        >
          <span style={{ color: 'var(--accent-pink)', marginRight: 4 }}>🌐 LAN:</span>
          <span className="font-mono" style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
            {currentHost}:5173
          </span>
          <span
            style={{
              marginLeft: 6,
              fontSize: '0.6rem',
              padding: '1px 5px',
              borderRadius: '4px',
              background: 'rgba(34, 197, 94, 0.15)',
              color: 'var(--risk-low)',
              border: '1px solid rgba(34, 197, 94, 0.3)',
            }}
          >
            Subnet Live
          </span>
        </div>

        <div className="header__health">
          <span
            className={`header__health-dot ${!isHealthy ? 'header__health-dot--error' : ''}`}
          />
          <span>{isHealthy ? 'AI Engine Online' : 'AI Engine Offline'}</span>
        </div>
      </div>
    </header>
  );
}
