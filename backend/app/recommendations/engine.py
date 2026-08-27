"""
Rule-Based Recommendation Engine — M6.2
==========================================

Generates prioritized defensive action recommendations based on:
- Dynamic risk scores and attack probabilities
- SHAP feature explanations (which features drove the prediction)
- Device metadata (type, criticality, vulnerability, department)
- Topology exposure (centrality metrics)

Design philosophy:
    - Deterministic rule-based system (not ML) — explainable, auditable
    - Each rule has a condition and generates an action
    - Rules are configurable and extensible
    - Output is a prioritized list of defensive actions

Usage:
    from backend.app.recommendations.engine import RecommendationEngine

    engine = RecommendationEngine()
    recs = engine.generate_recommendations({
        "device_id": "WEB-SERVER-01",
        "attack_probability": 0.85,
        "risk_score": 0.9,
        "anomaly_score": 0.7,
        "vulnerability_score": 0.65,
        "criticality": 0.9,
        "device_type": "server",
        "department": "admin",
        "topology_exposure": 0.8,
        "top_features": ["neighbor_attack_count", "degree", "flow_bytes_per_s"],
    })
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


# ==============================================================================
# PRIORITY LEVELS
# ==============================================================================

PRIORITY_ORDER = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}


# ==============================================================================
# RECOMMENDATION ENGINE
# ==============================================================================

class RecommendationEngine:
    """
    Rule-based engine that generates defensive recommendations.

    Each rule is a dict with:
    - name: Human-readable rule identifier
    - category: Action category (isolation, access_control, monitoring, etc.)
    - condition: Callable(context) → bool
    - generate: Callable(context) → dict with {action, priority, reason, category}

    Args:
        custom_rules: Optional list of additional rules to add.
        thresholds: Override default thresholds for rule conditions.
    """

    def __init__(
        self,
        custom_rules: list[dict] | None = None,
        thresholds: dict | None = None,
    ):
        # Default thresholds (tunable)
        self.thresholds = {
            "attack_prob_critical": 0.9,
            "attack_prob_high": 0.7,
            "attack_prob_moderate": 0.4,
            "risk_score_critical": 0.85,
            "risk_score_high": 0.6,
            "anomaly_high": 0.7,
            "anomaly_moderate": 0.4,
            "vulnerability_high": 0.6,
            "vulnerability_moderate": 0.4,
            "criticality_high": 0.7,
            "topology_exposure_high": 0.6,
        }
        if thresholds:
            self.thresholds.update(thresholds)

        # Build rules
        self.rules = self._build_default_rules()
        if custom_rules:
            self.rules.extend(custom_rules)

    # ==========================================================================
    # DEFAULT RULES
    # ==========================================================================

    def _build_default_rules(self) -> list[dict]:
        """Build the default rule set."""
        t = self.thresholds
        return [
            # ── CRITICAL: Immediate SOC Escalation ──
            {
                "name": "immediate_soc_escalation",
                "category": "incident_response",
                "condition": lambda ctx: (
                    ctx.get("attack_probability", 0) > t["attack_prob_critical"]
                    and ctx.get("criticality", 0) > t["criticality_high"]
                ),
                "generate": lambda ctx: {
                    "action": (
                        f"IMMEDIATE: Escalate {ctx['device_id']} to SOC — "
                        f"critical asset under imminent attack threat "
                        f"(attack_prob={ctx.get('attack_probability', 0):.0%}, "
                        f"criticality={ctx.get('criticality', 0):.0%})"
                    ),
                    "priority": "critical",
                    "reason": (
                        "This device has both very high attack probability AND high "
                        "asset criticality. Compromise would cause significant damage."
                    ),
                    "category": "incident_response",
                },
            },

            # ── CRITICAL: Network Isolation for active threats ──
            {
                "name": "isolate_active_threat_source",
                "category": "network_isolation",
                "condition": lambda ctx: (
                    ctx.get("attack_probability", 0) > t["attack_prob_critical"]
                    and ctx.get("device_type", "") in ("workstation", "iot", "printer")
                ),
                "generate": lambda ctx: {
                    "action": (
                        f"ISOLATE: Move {ctx['device_id']} to quarantine VLAN — "
                        f"non-critical endpoint with attack_prob={ctx.get('attack_probability', 0):.0%}"
                    ),
                    "priority": "critical",
                    "reason": (
                        "Non-critical endpoint is very likely to be targeted. "
                        "Isolating prevents lateral movement to sensitive assets."
                    ),
                    "category": "network_isolation",
                },
            },

            # ── HIGH: Restrict access for high-risk devices ──
            {
                "name": "restrict_access_high_risk",
                "category": "access_control",
                "condition": lambda ctx: (
                    ctx.get("attack_probability", 0) > t["attack_prob_high"]
                    and _has_feature(ctx, ["dst_port", "src_port", "unique_dst_count",
                                          "unique_src_count", "protocol"])
                ),
                "generate": lambda ctx: {
                    "action": (
                        f"RESTRICT: Limit network access for {ctx['device_id']} — "
                        f"apply strict firewall rules on flagged ports/protocols"
                    ),
                    "priority": "high",
                    "reason": (
                        "Port/connection anomalies are among the top features driving "
                        "this prediction. Restricting access reduces the attack surface."
                    ),
                    "category": "access_control",
                },
            },

            # ── HIGH: Enhanced monitoring for anomalous devices ──
            {
                "name": "enhanced_monitoring_anomaly",
                "category": "monitoring",
                "condition": lambda ctx: (
                    ctx.get("anomaly_score", 0) > t["anomaly_high"]
                    and ctx.get("attack_probability", 0) > t["attack_prob_moderate"]
                ),
                "generate": lambda ctx: {
                    "action": (
                        f"MONITOR: Enable enhanced logging and IDS alerting for "
                        f"{ctx['device_id']} — anomaly_score={ctx.get('anomaly_score', 0):.0%}"
                    ),
                    "priority": "high",
                    "reason": (
                        "Device shows anomalous behavior patterns combined with "
                        "moderate-to-high attack probability. Detailed monitoring "
                        "can detect the attack in its early stages."
                    ),
                    "category": "monitoring",
                },
            },

            # ── HIGH: Patch vulnerable devices ──
            {
                "name": "patch_vulnerable_device",
                "category": "patch_management",
                "condition": lambda ctx: (
                    ctx.get("vulnerability_score", 0) > t["vulnerability_high"]
                    and ctx.get("attack_probability", 0) > t["attack_prob_moderate"]
                ),
                "generate": lambda ctx: {
                    "action": (
                        f"PATCH: Apply security updates to {ctx['device_id']} — "
                        f"vulnerability_score={ctx.get('vulnerability_score', 0):.0%}, "
                        f"consider emergency maintenance window"
                    ),
                    "priority": "high",
                    "reason": (
                        "Device has high vulnerability score and is at risk of attack. "
                        "Patching reduces exploitable weaknesses before the attacker "
                        "reaches this device."
                    ),
                    "category": "patch_management",
                },
            },

            # ── MEDIUM: Network micro-segmentation ──
            {
                "name": "segment_exposed_device",
                "category": "network_segmentation",
                "condition": lambda ctx: (
                    ctx.get("topology_exposure", 0) > t["topology_exposure_high"]
                    and ctx.get("risk_score", 0) > t["risk_score_high"]
                ),
                "generate": lambda ctx: {
                    "action": (
                        f"SEGMENT: Implement micro-segmentation around {ctx['device_id']} — "
                        f"high topology exposure (betweenness={ctx.get('topology_exposure', 0):.2f})"
                    ),
                    "priority": "medium",
                    "reason": (
                        "Device is highly connected in the network topology, making it "
                        "a bridge for attack propagation. Segmentation limits the blast "
                        "radius if this device is compromised."
                    ),
                    "category": "network_segmentation",
                },
            },

            # ── MEDIUM: Monitor lateral movement paths ──
            {
                "name": "monitor_lateral_movement",
                "category": "monitoring",
                "condition": lambda ctx: (
                    _has_feature(ctx, ["neighbor_attack_count", "pagerank",
                                       "betweenness_centrality"])
                    and ctx.get("attack_probability", 0) > t["attack_prob_moderate"]
                ),
                "generate": lambda ctx: {
                    "action": (
                        f"MONITOR: Track lateral movement attempts from/to "
                        f"{ctx['device_id']} — neighbor attack patterns detected"
                    ),
                    "priority": "medium",
                    "reason": (
                        "Graph topology features (neighbor attacks, centrality) are "
                        "significant prediction drivers. This suggests the device is "
                        "on a potential attack propagation path."
                    ),
                    "category": "monitoring",
                },
            },

            # ── MEDIUM: Backup critical data ──
            {
                "name": "backup_critical_asset",
                "category": "data_protection",
                "condition": lambda ctx: (
                    ctx.get("criticality", 0) > t["criticality_high"]
                    and ctx.get("risk_score", 0) > t["risk_score_high"]
                    and ctx.get("device_type", "") in ("server", "database")
                ),
                "generate": lambda ctx: {
                    "action": (
                        f"BACKUP: Initiate emergency backup of data on "
                        f"{ctx['device_id']} ({ctx.get('device_type', 'unknown')}) — "
                        f"critical asset at elevated risk"
                    ),
                    "priority": "medium",
                    "reason": (
                        "Critical server/database is at elevated risk. Backing up data "
                        "ensures recovery capability if the device is compromised."
                    ),
                    "category": "data_protection",
                },
            },

            # ── LOW: Review access policies ──
            {
                "name": "review_access_policies",
                "category": "access_control",
                "condition": lambda ctx: (
                    ctx.get("attack_probability", 0) > t["attack_prob_moderate"]
                    and ctx.get("department", "") in ("student", "guest")
                ),
                "generate": lambda ctx: {
                    "action": (
                        f"REVIEW: Audit access policies for {ctx['device_id']} "
                        f"({ctx.get('department', 'unknown')} department) — "
                        f"consider reducing privileges"
                    ),
                    "priority": "low",
                    "reason": (
                        "Device belongs to a lower-trust department and shows elevated "
                        "attack probability. Reviewing access policies may reveal "
                        "over-provisioned permissions."
                    ),
                    "category": "access_control",
                },
            },

            # ── LOW: General security hardening ──
            {
                "name": "general_hardening",
                "category": "hardening",
                "condition": lambda ctx: (
                    ctx.get("vulnerability_score", 0) > t["vulnerability_moderate"]
                    and ctx.get("risk_score", 0) > t["risk_score_high"]
                ),
                "generate": lambda ctx: {
                    "action": (
                        f"HARDEN: Review and close unnecessary services/ports on "
                        f"{ctx['device_id']} — reduce attack surface"
                    ),
                    "priority": "low",
                    "reason": (
                        "Device has moderate vulnerability combined with elevated risk. "
                        "Hardening the configuration reduces the number of potential "
                        "entry points for an attacker."
                    ),
                    "category": "hardening",
                },
            },
        ]

    # ==========================================================================
    # GENERATE RECOMMENDATIONS
    # ==========================================================================

    def generate_recommendations(
        self,
        context: dict[str, Any],
    ) -> list[dict]:
        """
        Generate all applicable recommendations for a device context.

        Args:
            context: Dictionary with device information:
                - device_id (str): Device identifier
                - attack_probability (float): ML model output [0, 1]
                - risk_score (float): Dynamic risk score [0, 1]
                - anomaly_score (float): Anomaly detection score [0, 1]
                - vulnerability_score (float): Device vulnerability [0, 1]
                - criticality (float): Asset criticality [0, 1]
                - device_type (str): workstation, server, router, etc.
                - department (str): student, faculty, admin, etc.
                - topology_exposure (float): Betweenness centrality [0, 1]
                - top_features (list[str]): Names of top SHAP features

        Returns:
            List of recommendation dicts sorted by priority,
            each with: action, priority, reason, category
        """
        recommendations = []

        for rule in self.rules:
            try:
                if rule["condition"](context):
                    rec = rule["generate"](context)
                    rec["rule_name"] = rule["name"]
                    recommendations.append(rec)
            except (KeyError, TypeError):
                # Skip rules that fail on missing context data
                continue

        # Sort by priority (critical first)
        recommendations.sort(key=lambda r: PRIORITY_ORDER.get(r["priority"], 99))

        return recommendations

    # ==========================================================================
    # BATCH RECOMMENDATIONS
    # ==========================================================================

    def generate_batch_recommendations(
        self,
        contexts: list[dict[str, Any]],
    ) -> dict[str, list[dict]]:
        """
        Generate recommendations for multiple devices.

        Args:
            contexts: List of device context dicts.

        Returns:
            Dict mapping device_id to list of recommendations.
        """
        results = {}
        for ctx in contexts:
            device_id = ctx.get("device_id", "unknown")
            recs = self.generate_recommendations(ctx)
            results[device_id] = recs
        return results

    def __repr__(self) -> str:
        return f"RecommendationEngine(rules={len(self.rules)})"


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _has_feature(context: dict, feature_names: list[str]) -> bool:
    """Check if any of the feature names appear in the context's top features."""
    top_features = context.get("top_features", [])
    if not top_features:
        return False
    return any(
        any(fname in feat for feat in top_features)
        for fname in feature_names
    )


# ==============================================================================
# SELF-TEST
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("M6.2 — Recommendation Engine Self-Test")
    print("=" * 60)

    engine = RecommendationEngine()
    print(f"Engine loaded: {engine}")

    # Test Case 1: Critical server under imminent threat
    print("\n--- Test 1: Critical server, high attack prob ---")
    ctx1 = {
        "device_id": "DATABASE-SERVER-01",
        "attack_probability": 0.95,
        "risk_score": 0.92,
        "anomaly_score": 0.8,
        "vulnerability_score": 0.7,
        "criticality": 0.95,
        "device_type": "server",
        "department": "admin",
        "topology_exposure": 0.75,
        "top_features": ["neighbor_attack_count", "degree", "total_fwd_packets"],
    }
    recs1 = engine.generate_recommendations(ctx1)
    for r in recs1:
        print(f"  [{r['priority'].upper():>8s}] {r['category']:<25s} {r['action'][:80]}")
    assert len(recs1) > 0, "Should generate recommendations for critical server"

    # Test Case 2: Student workstation, moderate risk
    print("\n--- Test 2: Student workstation, moderate risk ---")
    ctx2 = {
        "device_id": "PC-03",
        "attack_probability": 0.5,
        "risk_score": 0.45,
        "anomaly_score": 0.5,
        "vulnerability_score": 0.5,
        "criticality": 0.2,
        "device_type": "workstation",
        "department": "student",
        "topology_exposure": 0.3,
        "top_features": ["flow_bytes_per_s", "dst_port"],
    }
    recs2 = engine.generate_recommendations(ctx2)
    for r in recs2:
        print(f"  [{r['priority'].upper():>8s}] {r['category']:<25s} {r['action'][:80]}")

    # Test Case 3: Low-risk device
    print("\n--- Test 3: Low-risk device ---")
    ctx3 = {
        "device_id": "PRINTER-01",
        "attack_probability": 0.1,
        "risk_score": 0.15,
        "anomaly_score": 0.1,
        "vulnerability_score": 0.2,
        "criticality": 0.1,
        "device_type": "printer",
        "department": "faculty",
        "topology_exposure": 0.1,
        "top_features": [],
    }
    recs3 = engine.generate_recommendations(ctx3)
    if recs3:
        for r in recs3:
            print(f"  [{r['priority'].upper():>8s}] {r['category']:<25s} {r['action'][:80]}")
    else:
        print("  (No recommendations — device is low risk)")
    assert len(recs3) == 0, "Low-risk device should not trigger any rules"

    # Test Case 4: High attack prob workstation (should trigger isolation)
    print("\n--- Test 4: High-risk workstation (isolation) ---")
    ctx4 = {
        "device_id": "PC-17",
        "attack_probability": 0.95,
        "risk_score": 0.7,
        "anomaly_score": 0.8,
        "vulnerability_score": 0.4,
        "criticality": 0.2,
        "device_type": "workstation",
        "department": "student",
        "topology_exposure": 0.5,
        "top_features": ["neighbor_attack_count", "pagerank"],
    }
    recs4 = engine.generate_recommendations(ctx4)
    for r in recs4:
        print(f"  [{r['priority'].upper():>8s}] {r['category']:<25s} {r['action'][:80]}")
    assert any(r["category"] == "network_isolation" for r in recs4), \
        "High-risk workstation should trigger isolation"

    print(f"\n{'='*60}")
    print(f"[OK] All recommendation engine tests passed.")
    print(f"{'='*60}")
