import { useEffect, useRef, useCallback } from 'react';
import cytoscape from 'cytoscape';
import dagre from 'cytoscape-dagre';

// Register dagre layout once
if (!cytoscape._dagreRegistered) {
  cytoscape.use(dagre);
  cytoscape._dagreRegistered = true;
}

/**
 * NetworkGraph — Interactive campus network topology rendered with Cytoscape.js.
 * Nodes are color-coded by risk level, with pulsing animations on critical nodes.
 */
export default function NetworkGraph({
  networkData,
  selectedDevice,
  onDeviceSelect,
  attackPath,
}) {
  const containerRef = useRef(null);
  const cyRef = useRef(null);

  // Device type → shape mapping
  const typeShapeMap = {
    server: 'round-rectangle',
    workstation: 'ellipse',
    router: 'diamond',
    switch: 'hexagon',
    firewall: 'star',
    access_point: 'triangle',
    printer: 'rectangle',
  };

  // Risk level → color mapping
  const riskColorMap = {
    critical: '#ef4444',
    high: '#f97316',
    medium: '#eab308',
    low: '#22c55e',
  };

  const getNodeColor = useCallback((node) => {
    const level = node.risk_level;
    if (level && riskColorMap[level]) return riskColorMap[level];
    // Default color based on type
    const typeColors = {
      server: '#7c3aed',
      workstation: '#6366f1',
      router: '#8b5cf6',
      switch: '#a78bfa',
      firewall: '#dc2626',
      access_point: '#a855f7',
      printer: '#71717a',
    };
    return typeColors[node.type] || '#6366f1';
  }, []);

  // Build Cytoscape elements from network data
  const buildElements = useCallback(() => {
    if (!networkData) return [];

    const elements = [];

    // Nodes
    for (const node of networkData.nodes) {
      const color = getNodeColor(node);
      const isCritical = node.risk_level === 'critical' || node.risk_level === 'high';
      const shape = typeShapeMap[node.type] || 'ellipse';

      elements.push({
        group: 'nodes',
        data: {
          id: node.id,
          label: node.label || node.id.replace(/-/g, '\n'),
          type: node.type,
          department: node.department,
          vlan: node.vlan,
          criticality: node.criticality,
          risk_score: node.risk_score,
          risk_level: node.risk_level,
          attack_probability: node.attack_probability,
          color,
          shape,
          borderWidth: isCritical ? 3 : 1.5,
          borderColor: isCritical ? color : 'rgba(124, 58, 237, 0.4)',
          size: mapSize(node.criticality),
        },
      });
    }

    // Edges
    for (const edge of networkData.edges) {
      const edgeStyle = getEdgeStyle(edge.connection_type);
      elements.push({
        group: 'edges',
        data: {
          id: `${edge.source}-${edge.target}`,
          source: edge.source,
          target: edge.target,
          connection_type: edge.connection_type,
          ...edgeStyle,
        },
      });
    }

    return elements;
  }, [networkData, getNodeColor]);

  // Initialize / update Cytoscape
  useEffect(() => {
    if (!containerRef.current || !networkData) return;

    const elements = buildElements();

    if (cyRef.current) {
      cyRef.current.destroy();
    }

    const cy = cytoscape({
      container: containerRef.current,
      elements,
      layout: {
        name: 'dagre',
        rankDir: 'TB',
        nodeSep: 55,
        rankSep: 70,
        padding: 30,
      },
      style: [
        // Node style
        {
          selector: 'node',
          style: {
            'background-color': 'data(color)',
            'background-opacity': 0.85,
            label: 'data(label)',
            color: '#e4e4e7',
            'font-size': '8px',
            'font-family': "'Inter', sans-serif",
            'font-weight': 500,
            'text-valign': 'bottom',
            'text-halign': 'center',
            'text-margin-y': 6,
            'text-wrap': 'wrap',
            'text-max-width': '80px',
            width: 'data(size)',
            height: 'data(size)',
            shape: 'data(shape)',
            'border-width': 'data(borderWidth)',
            'border-color': 'data(borderColor)',
            'border-opacity': 0.8,
            'overlay-padding': '4px',
            'text-outline-width': 1,
            'text-outline-color': '#0a0a0f',
            'text-outline-opacity': 0.8,
            'transition-property': 'border-width, border-color, background-opacity, width, height',
            'transition-duration': '200ms',
          },
        },
        // Selected node
        {
          selector: 'node:selected',
          style: {
            'border-width': 4,
            'border-color': '#ec4899',
            'background-opacity': 1,
            'z-index': 999,
          },
        },
        // Highlighted (attack path)
        {
          selector: 'node.attack-path-node',
          style: {
            'border-width': 4,
            'border-color': '#dc2626',
            'background-opacity': 1,
            'z-index': 998,
          },
        },
        // Edge style
        {
          selector: 'edge',
          style: {
            width: 'data(edgeWidth)',
            'line-color': 'data(edgeColor)',
            'line-opacity': 0.35,
            'curve-style': 'bezier',
            'line-style': 'data(lineStyle)',
            'target-arrow-shape': 'none',
            'transition-property': 'line-opacity, line-color, width',
            'transition-duration': '200ms',
          },
        },
        // Highlighted edge (attack path)
        {
          selector: 'edge.attack-path-edge',
          style: {
            'line-color': '#dc2626',
            'line-opacity': 0.9,
            width: 3,
            'target-arrow-shape': 'triangle',
            'target-arrow-color': '#dc2626',
            'arrow-scale': 0.8,
            'z-index': 999,
          },
        },
        // Dimmed elements (when showing attack path)
        {
          selector: 'node.dimmed',
          style: {
            'background-opacity': 0.15,
            'border-opacity': 0.1,
            'text-opacity': 0.2,
          },
        },
        {
          selector: 'edge.dimmed',
          style: {
            'line-opacity': 0.05,
          },
        },
      ],
      minZoom: 0.3,
      maxZoom: 3,
      wheelSensitivity: 0.3,
    });

    // Click handler
    cy.on('tap', 'node', (evt) => {
      const nodeData = evt.target.data();
      onDeviceSelect?.(nodeData);
    });

    // Deselect on background click
    cy.on('tap', (evt) => {
      if (evt.target === cy) {
        onDeviceSelect?.(null);
      }
    });

    // Hover effects
    cy.on('mouseover', 'node', (evt) => {
      evt.target.style({
        'background-opacity': 1,
        'border-width': Math.max(Number(evt.target.data('borderWidth')), 3),
      });
      containerRef.current.style.cursor = 'pointer';
    });

    cy.on('mouseout', 'node', (evt) => {
      if (!evt.target.selected()) {
        evt.target.style({
          'background-opacity': 0.85,
          'border-width': evt.target.data('borderWidth'),
        });
      }
      containerRef.current.style.cursor = 'default';
    });

    cyRef.current = cy;

    return () => {
      cy.destroy();
      cyRef.current = null;
    };
  }, [networkData, buildElements, onDeviceSelect]);

  // Highlight selected device
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;

    cy.nodes().unselect();
    if (selectedDevice) {
      const id = selectedDevice.id || selectedDevice.device_id;
      const node = cy.getElementById(id);
      if (node.length) {
        node.select();
        cy.animate({ center: { eles: node }, duration: 300 });
      }
    }
  }, [selectedDevice]);

  // Highlight attack path
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;

    // Clear previous highlights
    cy.elements().removeClass('attack-path-node attack-path-edge dimmed');

    if (attackPath && attackPath.path && attackPath.path.length > 1) {
      // Dim all
      cy.elements().addClass('dimmed');

      // Highlight path nodes
      for (const step of attackPath.path) {
        const node = cy.getElementById(step.device_id);
        if (node.length) {
          node.removeClass('dimmed').addClass('attack-path-node');
        }
      }

      // Highlight path edges
      for (let i = 0; i < attackPath.path.length - 1; i++) {
        const src = attackPath.path[i].device_id;
        const tgt = attackPath.path[i + 1].device_id;
        // Try both directions
        const edge = cy.getElementById(`${src}-${tgt}`).length
          ? cy.getElementById(`${src}-${tgt}`)
          : cy.getElementById(`${tgt}-${src}`);
        if (edge.length) {
          edge.removeClass('dimmed').addClass('attack-path-edge');
        }
      }
    }
  }, [attackPath]);

  return (
    <div className="cyber-card network-graph" style={{ padding: 0, overflow: 'hidden' }}>
      <div ref={containerRef} className="network-graph__canvas" />
      <NetworkLegend />
    </div>
  );
}

/* ── Helpers ─────────────────────────────────────────────────────────────────── */

function mapSize(criticality) {
  const base = 28;
  const scale = 22;
  return base + (criticality || 0.3) * scale;
}

function getEdgeStyle(connectionType) {
  const styles = {
    trunk:    { edgeWidth: 2.5, edgeColor: '#7c3aed', lineStyle: 'solid' },
    fiber:    { edgeWidth: 2,   edgeColor: '#6366f1', lineStyle: 'solid' },
    ethernet: { edgeWidth: 1.5, edgeColor: '#6366f1', lineStyle: 'solid' },
    wifi:     { edgeWidth: 1,   edgeColor: '#a855f7', lineStyle: 'dashed' },
  };
  return styles[connectionType] || styles.ethernet;
}

/* ── Network Legend ──────────────────────────────────────────────────────────── */

function NetworkLegend() {
  const riskItems = [
    { label: 'Critical', color: '#ef4444' },
    { label: 'High', color: '#f97316' },
    { label: 'Medium', color: '#eab308' },
    { label: 'Low', color: '#22c55e' },
  ];

  const typeItems = [
    { label: 'Server', color: '#7c3aed' },
    { label: 'Workstation', color: '#6366f1' },
    { label: 'Router', color: '#8b5cf6' },
    { label: 'Firewall', color: '#dc2626' },
  ];

  return (
    <div className="network-legend">
      {riskItems.map((item) => (
        <div key={item.label} className="network-legend__item">
          <span
            className="network-legend__dot"
            style={{ background: item.color, boxShadow: `0 0 6px ${item.color}50` }}
          />
          {item.label}
        </div>
      ))}
      <span style={{ color: 'var(--border-subtle)', margin: '0 2px' }}>│</span>
      {typeItems.map((item) => (
        <div key={item.label} className="network-legend__item">
          <span className="network-legend__dot" style={{ background: item.color }} />
          {item.label}
        </div>
      ))}
    </div>
  );
}
