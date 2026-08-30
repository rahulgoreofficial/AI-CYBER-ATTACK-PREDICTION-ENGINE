import { useState, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import Dashboard from './pages/Dashboard';

/**
 * App — Root component managing the global layout (sidebar + header + content)
 * and top-level state (selected device, active section, current window).
 */
export default function App() {
  const [selectedDevice, setSelectedDevice] = useState(null);
  const [activeSection, setActiveSection] = useState('dashboard');
  const [currentWindowId, setCurrentWindowId] = useState(null);
  const [stats, setStats] = useState({ deviceCount: null, windowCount: null });

  const handleDeviceSelect = useCallback((device) => {
    setSelectedDevice(device);
  }, []);

  const handleDataLoaded = useCallback((data) => {
    setStats((prev) => ({ ...prev, ...data }));
  }, []);

  const handleWindowChange = useCallback((windowId) => {
    setCurrentWindowId(windowId);
  }, []);

  return (
    <div className="app-layout">
      <Sidebar
        selectedDevice={selectedDevice}
        activeSection={activeSection}
        onSectionChange={setActiveSection}
      />

      <main className="app-main">
        <Header
          deviceCount={stats.deviceCount}
          windowCount={stats.windowCount}
          currentWindowId={currentWindowId}
        />

        <div className="app-content">
          <Dashboard
            selectedDevice={selectedDevice}
            onDeviceSelect={handleDeviceSelect}
            onDataLoaded={handleDataLoaded}
            currentWindowId={currentWindowId}
            onWindowChange={handleWindowChange}
            activeSection={activeSection}
          />
        </div>
      </main>
    </div>
  );
}
