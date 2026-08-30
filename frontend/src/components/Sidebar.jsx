import { useState, useEffect } from 'react';
import { fetchHealth } from '../services/api';

/**
 * Sidebar — Fixed left navigation panel with branding, nav links, and
 * selected device info.
 */
export default function Sidebar({ selectedDevice, activeSection, onSectionChange }) {
  const sections = [
    { id: 'dashboard', icon: '⬡', label: 'Dashboard' },
    { id: 'network',   icon: '◉', label: 'Network Graph' },
    { id: 'risk',      icon: '⚠', label: 'Risk Analysis' },
    { id: 'models',    icon: '◈', label: 'Model Metrics' },
  ];

  return (
    <aside className="sidebar">
      {/* Brand */}
      <div className="sidebar__brand">
        <div className="sidebar__logo">⬡</div>
        <div className="sidebar__brand-text">
          <span className="sidebar__brand-name">CyberShield AI</span>
          <span className="sidebar__brand-sub">Attack Prediction Engine</span>
        </div>
      </div>

      {/* Navigation */}
      <nav className="sidebar__nav">
        <div className="sidebar__section">
          <div className="sidebar__section-label">Navigation</div>
          {sections.map((s) => (
            <div
              key={s.id}
              className={`sidebar__link ${activeSection === s.id ? 'sidebar__link--active' : ''}`}
              onClick={() => onSectionChange(s.id)}
            >
              <span className="sidebar__link-icon">{s.icon}</span>
              {s.label}
            </div>
          ))}
        </div>

        <div className="sidebar__section">
          <div className="sidebar__section-label">Analysis</div>
          <div
            className={`sidebar__link ${activeSection === 'explanation' ? 'sidebar__link--active' : ''}`}
            onClick={() => onSectionChange('explanation')}
          >
            <span className="sidebar__link-icon">◐</span>
            Explanations
          </div>
          <div
            className={`sidebar__link ${activeSection === 'attack-path' ? 'sidebar__link--active' : ''}`}
            onClick={() => onSectionChange('attack-path')}
          >
            <span className="sidebar__link-icon">⟿</span>
            Attack Path
          </div>
          <div
            className={`sidebar__link ${activeSection === 'recommendations' ? 'sidebar__link--active' : ''}`}
            onClick={() => onSectionChange('recommendations')}
          >
            <span className="sidebar__link-icon">🛡</span>
            Recommendations
          </div>
        </div>
      </nav>

      {/* Selected Device Info */}
      {selectedDevice && (
        <div className="sidebar__device-info animate-fade-in">
          <div className="sidebar__device-label">Selected Device</div>
          <div className="sidebar__device-name">{selectedDevice.id || selectedDevice.device_id}</div>
          <div className="sidebar__device-meta">
            {selectedDevice.type} · {selectedDevice.department || selectedDevice.vlan}
          </div>
          {selectedDevice.risk_score != null && (
            <div className="sidebar__device-meta" style={{ marginTop: 4 }}>
              Risk: <span className="font-mono" style={{ color: getRiskColor(selectedDevice.risk_level) }}>
                {(selectedDevice.risk_score * 100).toFixed(1)}%
              </span>
            </div>
          )}
        </div>
      )}
    </aside>
  );
}

function getRiskColor(level) {
  const map = {
    critical: 'var(--risk-critical)',
    high: 'var(--risk-high)',
    medium: 'var(--risk-medium)',
    low: 'var(--risk-low)',
  };
  return map[level] || 'var(--text-muted)';
}
