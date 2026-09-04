"""
LAN API — GET /api/network/lan-devices
=======================================

Returns live discovered connected devices on the local network (Wi-Fi / Ethernet subnet).
"""

from fastapi import APIRouter
from backend.app.services.lan_service import get_connected_lan_devices, get_host_network_info

router = APIRouter(prefix="/api/network", tags=["Network LAN"])


@router.get("/lan-devices")
async def get_lan_devices():
    """
    Get live discovered devices on the same local area network (LAN / Wi-Fi).
    Includes host machine IP, gateway router, and connected peer devices with
    baseline dynamic risk scores.
    """
    host_info = get_host_network_info()
    devices = get_connected_lan_devices()
    return {
        "host": host_info,
        "total_discovered": len(devices),
        "devices": devices,
    }
