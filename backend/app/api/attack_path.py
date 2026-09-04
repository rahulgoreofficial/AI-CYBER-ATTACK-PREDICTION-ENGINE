"""
Attack Path API — GET /api/attack-path/{device_id}
====================================================

Returns the attack propagation path for a device.
"""

from fastapi import APIRouter, Path, Query, HTTPException
from typing import Optional

from backend.app.models.schemas import AttackPathResponse
from backend.app.services.data_loader import get_data_store

router = APIRouter(prefix="/api", tags=["Attack Path"])


@router.get("/attack-path/{device_id}", response_model=AttackPathResponse)
async def get_attack_path(
    device_id: str = Path(..., description="Device ID, e.g. WEB-SERVER-01"),
    window_id: Optional[int] = Query(None, description="Time window ID (default: latest)"),
):
    """
    Get attack propagation path for/towards a device.

    Constructs a plausible attack propagation path based on the network
    topology and risk scores. Since CICIDS2017 Wednesday attacks are
    single-hop DoS (no multi-step chains), this reconstructs a path
    from the network topology by tracing high-risk neighbors.
    """
    store = get_data_store()

    # Check if this is a real physical LAN device
    if any(k in device_id for k in ["192.168", "HOST", "LAN", "Router", "Gateway", "Phone", "Smart", "Peer", "Client"]) or device_info is None:
        from backend.app.services.lan_service import get_lan_attack_path
        return get_lan_attack_path(device_id)

    if window_id is None:
        window_id = store.get_latest_window_id()

    # Build adjacency from topology connections
    neighbors: dict[str, list[str]] = {}
    for conn in store.connections:
        src, dst = conn["from"], conn["to"]
        neighbors.setdefault(src, []).append(dst)
        neighbors.setdefault(dst, []).append(src)

    # Get risk data for the window
    risk_entries = store.risk_by_window.get(window_id, [])
    risk_map = {e["device_id"]: e for e in risk_entries}

    # Build attack path: trace from high-risk neighbors toward the target device
    # Strategy: BFS backward through the topology, prioritizing high-risk devices
    path_nodes = []
    visited = set()

    def _build_path(target: str, max_depth: int = 4) -> list[dict]:
        """Trace backward from the target through high-risk neighbors."""
        result = []

        # Add the target device itself
        target_risk = risk_map.get(target, {})
        target_info = store.get_device_info(target) or {}
        result.append({
            "device_id": target,
            "device_type": target_info.get("type", ""),
            "attack_probability": target_risk.get("attack_probability", 0.0),
            "risk_score": target_risk.get("dynamic_risk_score", 0.0),
            "step": 0,
        })
        visited.add(target)

        # Trace backward through neighbors
        current = target
        for step in range(1, max_depth + 1):
            nbrs = neighbors.get(current, [])
            # Find the highest-risk unvisited neighbor
            best_nbr = None
            best_risk = -1.0
            for nbr in nbrs:
                if nbr in visited:
                    continue
                nbr_risk = risk_map.get(nbr, {}).get("dynamic_risk_score", 0.0)
                if nbr_risk > best_risk:
                    best_risk = nbr_risk
                    best_nbr = nbr

            if best_nbr is None:
                break

            visited.add(best_nbr)
            nbr_info = store.get_device_info(best_nbr) or {}
            nbr_risk_data = risk_map.get(best_nbr, {})
            result.append({
                "device_id": best_nbr,
                "device_type": nbr_info.get("type", ""),
                "attack_probability": nbr_risk_data.get("attack_probability", 0.0),
                "risk_score": nbr_risk_data.get("dynamic_risk_score", 0.0),
                "step": step,
            })
            current = best_nbr

        # Reverse so the path goes from source → target
        result.reverse()
        for i, node in enumerate(result):
            node["step"] = i

        return result

    path_nodes = _build_path(device_id)

    # Determine description
    if len(path_nodes) > 1:
        source = path_nodes[0]["device_id"]
        description = (
            f"Potential attack propagation path from {source} to {device_id} "
            f"({len(path_nodes)} hops), traced through highest-risk neighbors."
        )
    else:
        description = (
            f"No multi-hop attack path found for {device_id}. "
            f"CICIDS2017 Wednesday attacks are primarily single-hop DoS."
        )

    return {
        "device_id": device_id,
        "path": path_nodes,
        "total_steps": len(path_nodes),
        "description": description,
    }
