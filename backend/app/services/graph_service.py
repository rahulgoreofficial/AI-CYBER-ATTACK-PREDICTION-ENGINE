"""
Graph Service — Network topology formatting for API/Cytoscape.js
================================================================

Converts the campus topology into API-ready format with risk overlays.
"""

from __future__ import annotations

import logging
from typing import Optional

from backend.app.services.data_loader import get_data_store

logger = logging.getLogger("backend.graph_service")


def get_network_topology(window_id: Optional[int] = None) -> dict:
    """
    Get the full network topology with optional risk overlay.

    Args:
        window_id: If provided, overlay risk scores from this window
                   onto the topology nodes.

    Returns:
        Dict with 'nodes', 'edges', 'total_nodes', 'total_edges'.
    """
    store = get_data_store()

    # Build nodes from topology devices
    nodes = []
    for device in store.devices:
        node = {
            "id": device["id"],
            "name": device.get("name", device["id"]),
            "type": device["type"],
            "department": device.get("department", ""),
            "vlan": device.get("vlan", ""),
            "os": device.get("os", ""),
            "criticality": device.get("criticality", 0.0),
            "vulnerability": device.get("vulnerability", 0.0),
            "open_ports": device.get("open_ports", []),
            "description": device.get("description", ""),
            "risk_score": None,
            "risk_level": None,
            "attack_probability": None,
        }
        nodes.append(node)

    # Overlay risk scores if window_id is provided
    if window_id is not None and window_id in store.risk_by_window:
        risk_entries = store.risk_by_window[window_id]
        risk_map = {e["device_id"]: e for e in risk_entries}

        for node in nodes:
            risk = risk_map.get(node["id"])
            if risk:
                node["risk_score"] = risk["dynamic_risk_score"]
                node["risk_level"] = store.get_risk_level(risk["dynamic_risk_score"])
                node["attack_probability"] = risk["attack_probability"]
    else:
        # Use the latest window's risk data if no window_id specified
        latest_wid = store.get_latest_window_id()
        if latest_wid in store.risk_by_window:
            risk_entries = store.risk_by_window[latest_wid]
            risk_map = {e["device_id"]: e for e in risk_entries}
            for node in nodes:
                risk = risk_map.get(node["id"])
                if risk:
                    node["risk_score"] = risk["dynamic_risk_score"]
                    node["risk_level"] = store.get_risk_level(risk["dynamic_risk_score"])
                    node["attack_probability"] = risk["attack_probability"]

    # Build edges from topology connections
    edges = []
    for conn in store.connections:
        edge = {
            "source": conn["from"],
            "target": conn["to"],
            "connection_type": conn.get("type", ""),
            "bandwidth": conn.get("bandwidth", ""),
        }
        edges.append(edge)

    return {
        "nodes": nodes,
        "edges": edges,
        "total_nodes": len(nodes),
        "total_edges": len(edges),
    }
