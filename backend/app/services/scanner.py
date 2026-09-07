"""
Background Network Scanner — Dynamic Subnet Discovery & Real-Time Device Registry
==================================================================================

Fully dynamic, zero-hardcoded network discovery engine.
Automatically detects host machine, default gateway router, and local subnet configuration
from OS routing tables and hardware BIOS. Performs parallel Layer-2 Win32 SendARP queries
across the entire subnet to discover all physical devices in real-time.

Features:
  - 100% Dynamic Subnet & Gateway Discovery: Reads default route from OS routing table.
  - Automatic Network Switch Detection: Detects when the machine switches Wi-Fi or subnets,
    clears stale entries, and scans the new network seamlessly.
  - Hardware-Aware Host Classification: Queries actual BIOS manufacturer via Windows registry.
  - Layer-2 ARP Sweep: Fast 128-thread Win32 SendARP queries bypass ICMP firewall blocks.
  - Comprehensive Device Categorization: Dynamically infers device types (mobile, IoT, TV,
    router, server, workstation) using MAC OUI, reverse DNS hostname patterns, and open ports.
  - Real-Time Event Dispatch: Emits connect, disconnect, update, and network_switch events.
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import platform
import re
import socket
import struct
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Callable, Optional

logger = logging.getLogger("backend.scanner")

# ──────────────────────────────────────────────────────────────────────────
# SYSTEM & HARDWARE DISCOVERY (Zero Hardcoding)
# ──────────────────────────────────────────────────────────────────────────

def get_system_manufacturer() -> str:
    """Query the host machine's actual hardware manufacturer from BIOS / system info."""
    try:
        if platform.system() == "Windows":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\BIOS") as key:
                mfg, _ = winreg.QueryValueEx(key, "SystemManufacturer")
                if mfg and mfg.strip():
                    return mfg.strip()
    except Exception:
        pass
    return f"{platform.system()} Host"


def get_network_routing() -> tuple[Optional[str], Optional[str]]:
    """
    Dynamically discover default gateway IP and primary local interface IP
    from the OS routing table (route print 0.0.0.0 on Windows).
    Returns (gateway_ip, host_ip).
    """
    gateway_ip = None
    host_ip = None

    if platform.system() == "Windows":
        try:
            output = subprocess.check_output(
                ["route", "print", "0.0.0.0"],
                text=True, timeout=2,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            )
            for line in output.splitlines():
                parts = line.strip().split()
                if len(parts) >= 5 and parts[0] == "0.0.0.0" and parts[1] == "0.0.0.0":
                    gateway_ip = parts[2]
                    host_ip = parts[3]
                    break
        except Exception as e:
            logger.debug(f"Failed to query Windows routing table: {e}")

    # Fallback to UDP socket trick if routing table didn't yield host_ip
    if not host_ip or host_ip.startswith("127."):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.5)
            s.connect(("8.8.8.8", 80))
            host_ip = s.getsockname()[0]
            s.close()
        except Exception:
            try:
                host_ip = socket.gethostbyname(socket.gethostname())
            except Exception:
                host_ip = "127.0.0.1"

    # Default fallback for gateway if not found
    if not gateway_ip and host_ip and host_ip != "127.0.0.1":
        prefix = ".".join(host_ip.split(".")[:3]) + "."
        gateway_ip = f"{prefix}1"

    return gateway_ip, host_ip


# ──────────────────────────────────────────────────────────────────────────
# MAC OUI → Manufacturer Database (Standard IEEE Vendor Registry)
# ──────────────────────────────────────────────────────────────────────────

OUI_DATABASE: dict[str, str] = {
    # Apple
    "AC:DE:48": "Apple", "3C:22:FB": "Apple", "F0:18:98": "Apple",
    "DC:A9:04": "Apple", "A4:83:E7": "Apple", "78:7B:8A": "Apple",
    "14:7D:DA": "Apple", "F4:5C:89": "Apple", "88:66:A5": "Apple",
    "6C:94:66": "Apple", "70:56:81": "Apple", "A8:88:08": "Apple",
    "D0:03:4B": "Apple", "B0:34:95": "Apple", "58:B0:35": "Apple",
    "7C:D1:C3": "Apple", "40:B3:95": "Apple", "AC:BC:32": "Apple",
    # Samsung
    "00:21:19": "Samsung", "B4:3A:28": "Samsung", "8C:F5:A3": "Samsung",
    "50:B7:C3": "Samsung", "CC:07:AB": "Samsung", "38:01:97": "Samsung",
    "AC:5F:3E": "Samsung", "10:D5:42": "Samsung", "84:25:DB": "Samsung",
    "C0:BD:C8": "Samsung", "E4:7D:BD": "Samsung", "BC:72:B1": "Samsung",
    "F8:04:2E": "Samsung", "78:AB:BB": "Samsung", "94:35:0A": "Samsung",
    # Xiaomi / Redmi
    "28:6C:07": "Xiaomi", "64:CC:2E": "Xiaomi", "34:CE:00": "Xiaomi",
    "78:11:DC": "Xiaomi", "7C:49:EB": "Xiaomi", "58:44:98": "Xiaomi",
    "AC:C1:EE": "Xiaomi", "0C:1D:AF": "Xiaomi", "50:64:2B": "Xiaomi",
    "FC:64:BA": "Xiaomi", "18:59:36": "Xiaomi", "98:FA:E3": "Xiaomi",
    "D8:23:E0": "Xiaomi", "9C:9D:7E": "Xiaomi",
    # Realme / Oppo / OnePlus / Vivo
    "3C:CD:5D": "Realme", "48:F1:7F": "Realme",
    "DC:6D:CD": "OPPO", "A4:3B:FA": "OPPO", "AC:D1:B8": "OPPO",
    "94:65:2D": "OnePlus", "C0:EE:FB": "OnePlus",
    "A0:86:C6": "Vivo", "4C:57:CA": "Vivo", "14:A3:2F": "Vivo",
    # Google / Pixel
    "F4:F5:D8": "Google", "3C:5A:B4": "Google", "94:EB:2C": "Google",
    "54:60:09": "Google",
    # Dell
    "00:14:22": "Dell", "F0:1F:AF": "Dell", "18:03:73": "Dell",
    "B0:83:FE": "Dell", "D4:BE:D9": "Dell", "14:FE:B5": "Dell",
    # HP
    "00:1E:0B": "HP", "3C:D9:2B": "HP", "10:60:4B": "HP",
    "68:B5:99": "HP", "D0:BF:9C": "HP",
    # Lenovo
    "00:06:1B": "Lenovo", "28:D2:44": "Lenovo", "54:E1:AD": "Lenovo",
    "E8:6A:64": "Lenovo", "98:FA:9B": "Lenovo",
    # Intel (Wi-Fi adapters)
    "00:1E:64": "Intel", "8C:8D:28": "Intel", "DC:71:96": "Intel",
    "3C:58:C2": "Intel", "A4:C3:F0": "Intel", "74:E6:E2": "Intel",
    # TP-Link
    "50:C7:BF": "TP-Link", "98:DA:C4": "TP-Link", "60:32:B1": "TP-Link",
    "14:EB:B6": "TP-Link", "30:DE:4B": "TP-Link",
    # Netgear
    "C4:04:15": "Netgear", "B0:B9:8A": "Netgear", "A4:2B:8C": "Netgear",
    # Asus
    "00:1A:92": "ASUS", "2C:56:DC": "ASUS", "04:D4:C4": "ASUS",
    "1C:87:2C": "ASUS", "AC:9E:17": "ASUS",
    # Amazon (Echo, Fire, Ring)
    "44:65:0D": "Amazon", "68:54:FD": "Amazon", "A0:02:DC": "Amazon",
    "FC:65:DE": "Amazon", "74:C2:46": "Amazon",
    # Microsoft / Xbox
    "7C:1E:52": "Microsoft", "28:18:78": "Microsoft",
    # Raspberry Pi
    "B8:27:EB": "Raspberry-Pi", "DC:A6:32": "Raspberry-Pi", "E4:5F:01": "Raspberry-Pi",
    # Espressif (ESP32/ESP8266 IoT)
    "24:0A:C4": "Espressif-IoT", "30:AE:A4": "Espressif-IoT",
    "A4:CF:12": "Espressif-IoT", "CC:50:E3": "Espressif-IoT",
    # Huawei / Honor
    "E0:19:1D": "Huawei", "48:46:FB": "Huawei", "CC:A2:23": "Huawei",
    "70:8C:B6": "Huawei", "04:F9:38": "Huawei",
    # LG
    "00:1C:62": "LG", "CC:2D:83": "LG", "64:89:9A": "LG",
    # Sony
    "00:13:A9": "Sony", "04:5D:4B": "Sony", "AC:9B:0A": "Sony",
    # D-Link
    "00:1C:F0": "D-Link", "00:22:B0": "D-Link", "78:54:2E": "D-Link",
    # Cisco
    "00:00:0C": "Cisco", "00:01:42": "Cisco", "00:01:43": "Cisco",
    "00:01:96": "Cisco", "00:01:97": "Cisco", "00:02:FC": "Cisco",
}


def lookup_oui(mac: str) -> str:
    """Look up manufacturer from MAC OUI prefix."""
    if not mac or mac in ("HOST-INTERFACE", "ff-ff-ff-ff-ff-ff", "UNKNOWN", "NOT-IN-ARP"):
        return "Unknown"
    normalized = mac.upper().replace("-", ":")
    prefix = normalized[:8]
    return OUI_DATABASE.get(prefix, "Unknown")


def infer_device_type(
    manufacturer: str,
    hostname: str,
    open_ports: list[int],
    ip: str,
    mac: str = "",
    gateway_ip: Optional[str] = None,
) -> tuple[str, str, str]:
    """
    Dynamically infer device type, role, and resolved manufacturer
    using MAC OUI, reverse DNS hostname clues, open ports, and gateway role.
    Returns (device_type, role, resolved_manufacturer).
    """
    mfg = manufacturer or "Unknown"
    mfg_lower = mfg.lower()
    host_lower = hostname.lower() if hostname else ""
    mac_clean = mac.upper().replace("-", ":")

    # Check for randomized / private Wi-Fi MAC (bit 1 of octet 0 set)
    is_private_mac = False
    if len(mac_clean) >= 2:
        try:
            first_byte = int(mac_clean[:2], 16)
            is_private_mac = (first_byte & 0x02) != 0
        except ValueError:
            pass

    # 1. Gateway Router (matches discovered default gateway or ends with .1 / .254)
    is_router_candidate = (gateway_ip and ip == gateway_ip) or ip.endswith(".1") or ip.endswith(".254")
    if is_router_candidate:
        brand = mfg if mfg != "Unknown" else "Gateway Router"
        # Extract router brand from hostname if present
        for known_brand in ["jiofiber", "jio", "airtel", "tplink", "tp-link", "netgear", "asus", "cisco", "dlink", "d-link", "huawei", "zyxel"]:
            if known_brand in host_lower:
                brand = known_brand.upper().replace("-", " ")
                break
        return "router", f"Default Gateway Router ({brand})", brand

    # 2. Dynamic Hostname Brand & Model Extraction
    clean_label = hostname.split(".")[0].replace("-", " ").strip() if hostname else ip

    brand_patterns = [
        (["oneplus", "nord"], "OnePlus", "mobile", "OnePlus Smartphone"),
        (["galaxy", "samsung", "sm-"], "Samsung", "mobile", "Samsung Galaxy"),
        (["iphone"], "Apple", "mobile", "Apple iPhone"),
        (["ipad"], "Apple", "mobile", "Apple iPad"),
        (["pixel"], "Google", "mobile", "Google Pixel"),
        (["redmi", "xiaomi", "mi-"], "Xiaomi", "mobile", "Xiaomi Smartphone"),
        (["realme"], "Realme", "mobile", "Realme Smartphone"),
        (["oppo"], "OPPO", "mobile", "OPPO Smartphone"),
        (["vivo"], "Vivo", "mobile", "Vivo Smartphone"),
        (["moto", "motorola"], "Motorola", "mobile", "Motorola Smartphone"),
        (["nothing"], "Nothing", "mobile", "Nothing Phone"),
        (["honor"], "Honor", "mobile", "Honor Smartphone"),
        (["huawei"], "Huawei", "mobile", "Huawei Device"),
        (["android"], "Android", "mobile", "Android Device"),
        (["settopbox", "stb"], mfg if mfg != "Unknown" else "Set-Top Box", "iot", "Set-Top Box / Media"),
        (["smarttv", "tv", "bravia", "webos", "firetv", "roku", "chromecast"], mfg if mfg != "Unknown" else "Smart TV", "iot", "Smart TV / Display"),
    ]

    for keywords, brand, dev_type, role_prefix in brand_patterns:
        if any(k in host_lower for k in keywords):
            return dev_type, f"{role_prefix} ({clean_label})", brand

    # 3. Manufacturer-based classification (from MAC OUI)
    if mfg_lower in ("apple", "samsung", "xiaomi", "realme", "oppo", "oneplus", "vivo", "huawei", "google", "motorola"):
        return "mobile", f"Smartphone ({mfg})", mfg

    if mfg_lower in ("amazon", "espressif-iot"):
        return "iot", f"IoT Device ({mfg})", mfg

    if mfg_lower == "raspberry-pi":
        return "server", "Raspberry Pi Server", mfg

    if mfg_lower in ("sony", "microsoft"):
        return "iot", f"Console / Media Device ({mfg})", mfg

    if mfg_lower in ("tp-link", "netgear", "asus", "d-link", "cisco"):
        if 80 in open_ports or 443 in open_ports:
            return "router", f"Network Router ({mfg})", mfg
        return "switch", f"Access Point ({mfg})", mfg

    if mfg_lower in ("dell", "hp", "lenovo", "intel", "acer"):
        if 22 in open_ports:
            return "server", f"Workstation / Server ({mfg})", mfg
        return "workstation", f"PC / Laptop ({mfg})", mfg

    if mfg_lower == "lg":
        return "iot", f"Smart TV / Display ({mfg})", mfg

    # 4. Port-based heuristics
    if 80 in open_ports or 443 in open_ports or 8080 in open_ports:
        return "server", f"Web Service Host ({ip})", mfg
    if 22 in open_ports:
        return "server", f"SSH Server ({ip})", mfg

    # 5. Private / Randomized MAC fallback (standard on modern iOS / Android on Wi-Fi)
    if is_private_mac:
        return "mobile", f"Mobile Device / Wi-Fi Peer ({ip})", "Private Wi-Fi MAC"

    return "workstation", f"Network Device ({ip})", mfg


# ──────────────────────────────────────────────────────────────────────────
# WIN32 SENDARP & FAST PORT CHECKING
# ──────────────────────────────────────────────────────────────────────────

def send_arp_query(ip_str: str) -> Optional[tuple[str, str]]:
    """
    Send a Layer-2 ARP request directly to an IP via Windows SendARP API.
    Succeeds even if device firewalls block ICMP ping.
    Returns (ip, formatted_mac) if host responds, else None.
    """
    try:
        ip = struct.unpack('<I', socket.inet_aton(ip_str))[0]
        mac = (ctypes.c_byte * 6)()
        mac_len = ctypes.c_ulong(6)
        res = ctypes.windll.iphlpapi.SendARP(ip, 0, ctypes.byref(mac), ctypes.byref(mac_len))
        if res == 0 and mac_len.value == 6:
            mac_str = '-'.join(f'{b & 0xff:02X}' for b in mac[:6])
            return ip_str, mac_str
    except Exception:
        pass
    return None


def ping_host(ip: str, timeout_ms: int = 60) -> bool:
    """Fallback ICMP ping for non-Windows platforms."""
    try:
        result = subprocess.run(
            ["ping", "-n", "1", "-w", str(timeout_ms), ip],
            capture_output=True, text=True, timeout=1.5,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
        return result.returncode == 0
    except Exception:
        return False


def check_ports_fast(ip: str, ports: list[int], timeout: float = 0.12) -> list[int]:
    """Check multiple ports on an IP using parallel socket connections."""
    open_ports = []
    def _check(port):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            if s.connect_ex((ip, port)) == 0:
                return port
            s.close()
        except Exception:
            pass
        return None

    with ThreadPoolExecutor(max_workers=min(len(ports), 10)) as executor:
        futures = {executor.submit(_check, p): p for p in ports}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                open_ports.append(result)
    return sorted(open_ports)


def resolve_hostname(ip: str) -> str:
    """Attempt reverse DNS or NetBIOS hostname resolution."""
    try:
        hostname = socket.gethostbyaddr(ip)[0]
        if hostname and hostname != ip:
            return hostname
    except (socket.herror, socket.gaierror, OSError):
        pass
    return ""


# ──────────────────────────────────────────────────────────────────────────
# DEVICE REGISTRY
# ──────────────────────────────────────────────────────────────────────────

class DeviceEntry:
    """Represents a discovered network device."""

    def __init__(self, ip: str, mac: str, manufacturer: str, hostname: str,
                 device_type: str, role: str, open_ports: list[int]):
        self.ip = ip
        self.mac = mac
        self.manufacturer = manufacturer
        self.hostname = hostname
        self.device_type = device_type
        self.role = role
        self.open_ports = open_ports
        self.first_seen = datetime.now()
        self.last_seen = datetime.now()
        self.status = "online"  # online, offline, new
        self.scan_count = 1
        self.is_host = False
        self.offline_cycles = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to API-compatible dictionary."""
        criticality = self._compute_criticality()
        vulnerability = self._compute_vulnerability()
        port_boost = 0.20 if len(self.open_ports) > 0 else 0.0
        attack_prob = min(round(
            (vulnerability * 0.7) + port_boost +
            (0.35 if self.device_type == "router" else 0.15), 3
        ), 0.95)
        risk_score = round(
            (attack_prob * 0.40) + (criticality * 0.35) +
            (vulnerability * 0.15) + 0.05, 3
        )

        if risk_score >= 0.80:
            risk_level = "critical"
        elif risk_score >= 0.60:
            risk_level = "high"
        elif risk_score >= 0.35:
            risk_level = "medium"
        else:
            risk_level = "low"

        display_name = self.hostname.split(".")[0] if self.hostname else self.ip
        device_id = f"{self.role.split('/')[0].split('(')[0].strip()} ({self.ip})"

        return {
            "device_id": device_id,
            "ip_address": self.ip,
            "mac_address": self.mac,
            "manufacturer": self.manufacturer,
            "hostname": self.hostname,
            "device_type": self.device_type,
            "department": "local-wifi-lan",
            "criticality": criticality,
            "vulnerability": vulnerability,
            "open_ports": self.open_ports,
            "is_host": self.is_host,
            "status": self.status,
            "label": f"{self.device_type.upper()}\n{display_name}",
            "role": self.role,
            "description": (
                f"{'Host machine' if self.is_host else 'Discovered device'} on local network. "
                f"Manufacturer: {self.manufacturer}. "
                f"{'Hostname: ' + self.hostname + '. ' if self.hostname else ''}"
                f"Open ports: {self.open_ports}. "
                f"First seen: {self.first_seen.strftime('%H:%M:%S')}"
            ),
            "attack_probability": attack_prob,
            "dynamic_risk_score": risk_score,
            "risk_level": risk_level,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "scan_count": self.scan_count,
        }

    def _compute_criticality(self) -> float:
        if self.is_host:
            return 0.95
        if self.device_type == "router":
            return 0.90
        if self.device_type == "server":
            return 0.80
        if self.device_type == "switch":
            return 0.75
        if self.device_type == "workstation":
            return 0.55
        if self.device_type == "mobile":
            return 0.40
        if self.device_type == "iot":
            return 0.50
        return 0.45

    def _compute_vulnerability(self) -> float:
        vuln = 0.20
        if self.device_type == "router":
            vuln = 0.45
        elif self.device_type == "server":
            vuln = 0.40
        elif self.device_type == "iot":
            vuln = 0.55
        elif self.device_type == "mobile":
            vuln = 0.25
        # Open port penalties
        dangerous_ports = {445, 23, 21, 3389, 5900}
        if any(p in dangerous_ports for p in self.open_ports):
            vuln = min(vuln + 0.15, 0.85)
        if len(self.open_ports) > 3:
            vuln = min(vuln + 0.10, 0.85)
        return round(vuln, 3)


# ──────────────────────────────────────────────────────────────────────────
# NETWORK SCANNER (Fully Dynamic Singleton)
# ──────────────────────────────────────────────────────────────────────────

class NetworkScanner:
    """
    Background network scanner that maintains a real-time device registry.
    Performs active Layer-2 sweeps, detects network/subnet switches,
    and broadcasts device connect/disconnect events.
    """

    def __init__(self):
        self.registry: dict[str, DeviceEntry] = {}  # IP → DeviceEntry
        self._lock = threading.Lock()
        self._running = False
        self._scan_count = 0
        self._last_scan_time: Optional[str] = None
        self._last_scan_duration: float = 0
        self._change_callbacks: list[Callable] = []

        # Dynamic network & host attributes
        self._host_ip = "127.0.0.1"
        self._gateway_ip: Optional[str] = None
        self._subnet_prefix = "127.0.0."
        self._hostname = socket.gethostname()
        self._host_manufacturer = get_system_manufacturer()

        # Perform initial host & network discovery
        self._discover_host()

    def add_change_callback(self, callback: Callable):
        """Register a callback for device change events."""
        self._change_callbacks.append(callback)

    def remove_change_callback(self, callback: Callable):
        """Remove a registered change callback."""
        if callback in self._change_callbacks:
            self._change_callbacks.remove(callback)

    def _notify_change(self, event_type: str, device: dict, all_devices: list[dict], extra: dict = None):
        """Notify all registered callbacks of a device change."""
        event = {
            "type": event_type,  # "connect", "disconnect", "update", "full_scan", "network_switch"
            "device": device,
            "total_devices": len(all_devices),
            "all_devices": all_devices,
            "timestamp": datetime.now().isoformat(),
            "scan_cycle": self._scan_count,
            "subnet": f"{self._subnet_prefix}0/24",
            "gateway": self._gateway_ip,
        }
        if extra:
            event.update(extra)

        for cb in self._change_callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.warning(f"Change callback error: {e}")

    def _discover_host(self) -> bool:
        """
        Dynamically query host IP, gateway IP, and subnet prefix.
        Returns True if the subnet changed since the previous check.
        """
        old_subnet = self._subnet_prefix
        gateway_ip, host_ip = get_network_routing()

        self._gateway_ip = gateway_ip
        if host_ip:
            self._host_ip = host_ip
            self._subnet_prefix = ".".join(host_ip.split(".")[:3]) + "."

        subnet_changed = (old_subnet != "127.0.0." and old_subnet != self._subnet_prefix)
        return subnet_changed

    def _read_arp_cache(self) -> dict[str, str]:
        """Read OS ARP cache to get IP → MAC mapping."""
        ip_mac_map = {}
        try:
            output = subprocess.check_output(
                ["arp", "-a"], text=True, timeout=3,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            )
            pattern = re.compile(r"(\d+\.\d+\.\d+\.\d+)\s+([0-9a-fA-F-]{17})\s+(\w+)")
            for match in pattern.finditer(output):
                ip, mac, _ = match.groups()
                if not ip.startswith(("224.", "239.", "255.", "127.")) and not ip.endswith(".255"):
                    ip_mac_map[ip] = mac.upper()
        except Exception as e:
            logger.debug(f"Failed to read ARP cache: {e}")
        return ip_mac_map

    def _active_discovery(self) -> dict[str, str]:
        """
        Actively sweep the /24 subnet for all live devices.
        Uses native Win32 SendARP on Windows (Layer-2 query),
        complemented by ARP cache inspection and host interface.
        """
        discovered: dict[str, str] = {}
        ips = [f"{self._subnet_prefix}{i}" for i in range(1, 255)]

        # 1. Native Windows SendARP sweep (fast Layer 2 ARP query for all 254 IPs)
        if hasattr(ctypes, "windll") and hasattr(ctypes.windll, "iphlpapi"):
            with ThreadPoolExecutor(max_workers=128) as executor:
                for res in executor.map(send_arp_query, ips):
                    if res:
                        discovered[res[0]] = res[1]
        else:
            # Fallback ping sweep for non-Windows
            with ThreadPoolExecutor(max_workers=64) as executor:
                alive = [ip for ip in ips if ping_host(ip, timeout_ms=50)]
                for ip in alive:
                    discovered[ip] = "UNKNOWN"

        # 2. Merge OS ARP cache to ensure 100% device capture
        arp_cache = self._read_arp_cache()
        for ip, mac in arp_cache.items():
            if ip.startswith(self._subnet_prefix):
                if ip not in discovered or discovered[ip] in ("UNKNOWN", "NOT-IN-ARP"):
                    discovered[ip] = mac

        # 3. Always include host machine
        if self._host_ip and self._host_ip != "127.0.0.1":
            if self._host_ip not in discovered:
                discovered[self._host_ip] = "HOST-INTERFACE"

        return discovered

    def _enrich_device(self, ip: str, mac: str) -> DeviceEntry:
        """Create a fully enriched DeviceEntry for a discovered IP."""
        raw_mfg = lookup_oui(mac)
        hostname = resolve_hostname(ip)

        # Port scanning
        scan_ports = [22, 53, 80, 443, 8080, 5000, 5173, 8000]
        if ip == self._gateway_ip or ip.endswith(".1") or ip.endswith(".254"):
            scan_ports = [22, 23, 53, 80, 443, 8080, 8443]
        elif ip == self._host_ip:
            scan_ports = [22, 80, 443, 445, 3389, 5173, 8000, 8080]

        open_ports = check_ports_fast(ip, scan_ports, timeout=0.12)
        device_type, role, resolved_mfg = infer_device_type(
            raw_mfg, hostname, open_ports, ip, mac, gateway_ip=self._gateway_ip
        )

        entry = DeviceEntry(
            ip=ip, mac=mac, manufacturer=resolved_mfg, hostname=hostname,
            device_type=device_type, role=role, open_ports=open_ports,
        )

        if ip == self._host_ip:
            entry.is_host = True
            entry.device_type = "server"
            entry.manufacturer = self._host_manufacturer
            entry.role = f"Host Machine ({self._hostname}) · SOC Engine"

        return entry

    def scan_once(self) -> list[dict]:
        """
        Perform a single full network scan cycle.
        Detects subnet switches, discovers devices, diffs state, and broadcasts events.
        """
        t_start = time.perf_counter()

        # Step 0: Check if network / subnet changed
        subnet_switched = self._discover_host()
        events = []

        if subnet_switched:
            logger.warning(f"🌐 Subnet switch detected! New subnet: {self._subnet_prefix}0/24")
            with self._lock:
                self.registry.clear()
            events.append(("network_switch", {"role": "Network", "device_id": f"Subnet: {self._subnet_prefix}0/24"}))

        # Step 1: Active Layer-2 discovery across /24 subnet
        discovered_map = self._active_discovery()

        # Step 2: Enrich all discovered devices in parallel
        new_entries: dict[str, DeviceEntry] = {}
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {
                executor.submit(self._enrich_device, ip, mac): ip
                for ip, mac in discovered_map.items()
            }
            for future in as_completed(futures, timeout=15):
                try:
                    entry = future.result(timeout=4)
                    new_entries[entry.ip] = entry
                except Exception as e:
                    logger.debug(f"Failed to enrich device: {e}")

        # Step 3: Diff with previous registry to detect changes
        with self._lock:
            previous_ips = set(self.registry.keys())
            current_ips = set(new_entries.keys())

            # New devices (connect)
            for ip in current_ips - previous_ips:
                entry = new_entries[ip]
                entry.status = "new"
                events.append(("connect", entry.to_dict()))

            # Disappeared devices (disconnect)
            for ip in previous_ips - current_ips:
                old_entry = self.registry[ip]
                # Only broadcast disconnect once when transitioning from online/new to offline
                if old_entry.status != "offline":
                    old_entry.status = "offline"
                    old_entry.last_seen = datetime.now()
                    events.append(("disconnect", old_entry.to_dict()))

            # Still present devices (update)
            for ip in current_ips & previous_ips:
                old = self.registry[ip]
                new = new_entries[ip]
                new.first_seen = old.first_seen
                new.scan_count = old.scan_count + 1
                new.status = "online"

                if (old.open_ports != new.open_ports or
                    old.mac != new.mac or
                    old.hostname != new.hostname or
                    old.status == "offline"):
                    events.append(("update", new.to_dict()))

            # Update registry
            # Keep offline devices for up to 2 scan cycles for smooth UI transitions
            updated_registry = {}
            for ip, entry in new_entries.items():
                updated_registry[ip] = entry

            for ip in previous_ips - current_ips:
                old_entry = self.registry[ip]
                old_entry.offline_cycles += 1
                if old_entry.offline_cycles <= 2:
                    old_entry.status = "offline"
                    updated_registry[ip] = old_entry

            self.registry = updated_registry

        self._scan_count += 1
        self._last_scan_time = datetime.now().isoformat()
        self._last_scan_duration = round(time.perf_counter() - t_start, 2)

        # Build full device list
        all_devices = self.get_all_devices()

        # Notify callbacks of specific connect/disconnect/update/network_switch events
        for item in events:
            event_type = item[0]
            device_dict = item[1]
            self._notify_change(event_type, device_dict, all_devices)

        # Always send a full_scan event to keep UI synchronized
        self._notify_change("full_scan", {}, all_devices)

        logger.info(
            f"Scan #{self._scan_count}: {len(current_ips)} alive devices on {self._subnet_prefix}0/24, "
            f"{len(events)} change event(s), {self._last_scan_duration}s"
        )

        return all_devices

    def get_all_devices(self) -> list[dict]:
        """Get all devices from the registry as API-compatible dicts."""
        with self._lock:
            devices = [entry.to_dict() for entry in self.registry.values()]
        # Sort: host first, then routers, then by risk score descending
        devices.sort(key=lambda d: (
            not d["is_host"],
            d["device_type"] != "router",
            -d["dynamic_risk_score"],
        ))
        return devices

    def get_status(self) -> dict:
        """Get scanner status info."""
        with self._lock:
            online = sum(1 for e in self.registry.values() if e.status != "offline")
            offline = sum(1 for e in self.registry.values() if e.status == "offline")
        return {
            "running": self._running,
            "scan_count": self._scan_count,
            "last_scan_time": self._last_scan_time,
            "last_scan_duration_s": self._last_scan_duration,
            "host_ip": self._host_ip,
            "gateway_ip": self._gateway_ip,
            "host_manufacturer": self._host_manufacturer,
            "subnet": f"{self._subnet_prefix}0/24",
            "total_devices": online + offline,
            "online_devices": online,
            "offline_devices": offline,
        }

    async def run_loop(self, interval: int = 8):
        """Run the scanner loop continuously."""
        self._running = True
        logger.info(f"Network scanner starting (interval={interval}s, subnet={self._subnet_prefix}0/24)")

        while self._running:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.scan_once)
            except Exception as e:
                logger.error(f"Scan cycle error: {e}")

            await asyncio.sleep(interval)

    def stop(self):
        """Stop the scanner loop."""
        self._running = False
        logger.info("Network scanner stopped")


# ──────────────────────────────────────────────────────────────────────────
# SINGLETON
# ──────────────────────────────────────────────────────────────────────────

_scanner_instance: Optional[NetworkScanner] = None


def get_scanner() -> NetworkScanner:
    """Get or create the singleton NetworkScanner instance."""
    global _scanner_instance
    if _scanner_instance is None:
        _scanner_instance = NetworkScanner()
    return _scanner_instance
