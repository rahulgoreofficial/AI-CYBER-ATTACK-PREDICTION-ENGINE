"""
Graph Visualization — Debug and notebook-friendly plotting helpers.
====================================================================

Provides simple matplotlib/NetworkX visualizations for:
- Time-window graphs (colored by risk/attack status)
- Attack propagation chains

Usage:
    from graph.visualization import plot_window_graph, plot_attack_chain
    plot_window_graph(G, title="Window 42")
"""

import sys
from pathlib import Path

import numpy as np
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.config import PATHS


# Color palette
COLORS = {
    "normal": "#4CAF50",       # Green
    "attack_dst": "#F44336",   # Red
    "attack_src": "#FF9800",   # Orange
    "infrastructure": "#2196F3",  # Blue
    "external": "#9E9E9E",     # Grey
    "edge_normal": "#BDBDBD",
    "edge_attack": "#F44336",
    "edge_topology": "#E0E0E0",
}

# Device type shapes (for legend, not actual node shapes in matplotlib)
DEVICE_TYPE_MARKERS = {
    "firewall": "s",       # square
    "router": "D",         # diamond
    "switch": "^",         # triangle
    "server": "h",         # hexagon
    "workstation": "o",    # circle
    "laptop": "o",
    "access_point": "p",   # pentagon
    "external": "v",       # inverted triangle
}


def plot_window_graph(
    G: nx.DiGraph,
    title: str = "Network Graph",
    highlight_attacks: bool = True,
    node_size_by: str = "criticality",
    figsize: tuple = (14, 10),
    save_path: str | None = None,
) -> None:
    """
    Plot a time-window network graph with attack highlighting.

    Args:
        G: NetworkX DiGraph for one time window.
        title: Plot title.
        highlight_attacks: Color attack-related nodes/edges differently.
        node_size_by: Node attribute to scale size by ('criticality', 'degree').
        figsize: Figure size.
        save_path: If provided, save plot to this path.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[viz] matplotlib not available — skipping plot")
        return

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "Empty graph", ha="center", va="center",
                fontsize=16, color="white")
        ax.set_title(title, fontsize=16, fontweight="bold", color="white")
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
        plt.show()
        return

    # Layout
    try:
        pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)
    except Exception:
        pos = nx.circular_layout(G)

    # ── Node colors ──
    node_colors = []
    for node in G.nodes:
        attrs = G.nodes[node]
        dtype = attrs.get("device_type", "unknown")
        has_traffic = attrs.get("has_traffic", False)

        if highlight_attacks:
            # Check if node is involved in attacks
            inbound_attacks = sum(
                G.edges[(src, node)].get("attack_count", 0)
                for src in G.predecessors(node)
            )
            outbound_attacks = sum(
                G.edges[(node, dst)].get("attack_count", 0)
                for dst in G.successors(node)
            )

            if inbound_attacks > 0:
                node_colors.append(COLORS["attack_dst"])
            elif outbound_attacks > 0:
                node_colors.append(COLORS["attack_src"])
            elif dtype == "external":
                node_colors.append(COLORS["external"])
            elif dtype in ("firewall", "router", "switch"):
                node_colors.append(COLORS["infrastructure"])
            else:
                node_colors.append(COLORS["normal"])
        else:
            if dtype == "external":
                node_colors.append(COLORS["external"])
            elif dtype in ("firewall", "router", "switch"):
                node_colors.append(COLORS["infrastructure"])
            else:
                node_colors.append(COLORS["normal"])

    # ── Node sizes ──
    if node_size_by == "criticality":
        node_sizes = [
            max(200, G.nodes[n].get("criticality", 0.3) * 1500)
            for n in G.nodes
        ]
    elif node_size_by == "degree":
        max_deg = max(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 1
        node_sizes = [
            max(200, (G.degree(n) / max(max_deg, 1)) * 1500)
            for n in G.nodes
        ]
    else:
        node_sizes = [400] * G.number_of_nodes()

    # ── Edge colors ──
    edge_colors = []
    edge_widths = []
    for u, v, data in G.edges(data=True):
        if data.get("attack_count", 0) > 0:
            edge_colors.append(COLORS["edge_attack"])
            edge_widths.append(2.5)
        elif data.get("is_topology_link", False):
            edge_colors.append(COLORS["edge_topology"])
            edge_widths.append(0.5)
        else:
            edge_colors.append(COLORS["edge_normal"])
            edge_widths.append(1.0)

    # ── Draw ──
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.5,
    )

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.6,
        arrows=True,
        arrowsize=10,
        connectionstyle="arc3,rad=0.1",
    )

    # Labels
    labels = {}
    for node in G.nodes:
        # Shorten long names
        label = node.replace("-SERVER-", "-SRV-").replace("EXT-ATTACKER-", "ATK-")
        labels[node] = label

    nx.draw_networkx_labels(
        G, pos, labels, ax=ax,
        font_size=7,
        font_color="white",
        font_weight="bold",
    )

    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS["normal"], label="Normal"),
        mpatches.Patch(color=COLORS["attack_dst"], label="Attack Target"),
        mpatches.Patch(color=COLORS["attack_src"], label="Attack Source"),
        mpatches.Patch(color=COLORS["infrastructure"], label="Infrastructure"),
        mpatches.Patch(color=COLORS["external"], label="External"),
    ]
    ax.legend(
        handles=legend_elements, loc="upper left",
        fontsize=8, facecolor="#16213e", edgecolor="white",
        labelcolor="white",
    )

    ax.set_title(title, fontsize=16, fontweight="bold", color="white", pad=20)
    ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[viz] Saved: {save_path}")

    plt.show()


def plot_attack_chain(
    chain: dict,
    topology: dict | None = None,
    figsize: tuple = (12, 4),
    save_path: str | None = None,
) -> None:
    """
    Visualize a single attack propagation chain as a linear path.

    Args:
        chain: Dict with 'path_nodes' (list of device IDs).
        topology: Optional topology for node metadata.
        figsize: Figure size.
        save_path: If provided, save plot.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[viz] matplotlib not available — skipping plot")
        return

    path_nodes = chain.get("path_nodes", [])
    if not path_nodes:
        print("[viz] Empty chain")
        return

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    n = len(path_nodes)
    x_positions = np.linspace(0.1, 0.9, n)
    y = 0.5

    for i, node in enumerate(path_nodes):
        # Color: first node = source (orange), last = target (red), middle = yellow
        if i == 0:
            color = COLORS["attack_src"]
        elif i == n - 1:
            color = COLORS["attack_dst"]
        else:
            color = "#FFEB3B"  # Yellow for intermediate

        circle = plt.Circle((x_positions[i], y), 0.04, color=color,
                             ec="white", linewidth=2, zorder=5)
        ax.add_patch(circle)

        # Label
        label = node.replace("-SERVER-", "-SRV-").replace("EXT-ATTACKER-", "ATK-")
        ax.text(x_positions[i], y - 0.12, label, ha="center", va="top",
                fontsize=8, color="white", fontweight="bold")

        # Arrow to next node
        if i < n - 1:
            ax.annotate(
                "", xy=(x_positions[i + 1] - 0.05, y),
                xytext=(x_positions[i] + 0.05, y),
                arrowprops=dict(
                    arrowstyle="->", color=COLORS["edge_attack"],
                    lw=2.5, connectionstyle="arc3,rad=0",
                ),
                zorder=4,
            )

    # Title
    chain_id = chain.get("chain_id", "?")
    chain_len = chain.get("chain_length", len(path_nodes) - 1)
    attack_types = chain.get("attack_types", [])
    types_str = ", ".join(attack_types) if attack_types else "Unknown"

    ax.set_title(
        f"Attack Chain #{chain_id} — {chain_len} hops — {types_str}",
        fontsize=13, fontweight="bold", color="white", pad=15,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[viz] Saved: {save_path}")

    plt.show()


# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testing graph visualization...")
    print("(This will attempt to display plots — close windows to continue)")

    # Create a sample graph
    G = nx.DiGraph()
    G.add_node("ROUTER-01", device_type="router", criticality=0.95, has_traffic=True)
    G.add_node("PC-01", device_type="workstation", criticality=0.15, has_traffic=True)
    G.add_node("PC-02", device_type="workstation", criticality=0.15, has_traffic=True)
    G.add_node("FILE-SRV", device_type="server", criticality=0.90, has_traffic=True)
    G.add_node("ATK-01", device_type="external", criticality=0.1, has_traffic=True)

    G.add_edge("ATK-01", "PC-01", frequency=50, attack_count=30)
    G.add_edge("PC-01", "ROUTER-01", frequency=100, attack_count=0)
    G.add_edge("PC-01", "FILE-SRV", frequency=20, attack_count=5)
    G.add_edge("PC-02", "ROUTER-01", frequency=80, attack_count=0)
    G.add_edge("ROUTER-01", "FILE-SRV", frequency=200, attack_count=0)

    plot_window_graph(G, title="Sample Network — Window Test")

    # Sample chain
    chain = {
        "chain_id": 0,
        "path_nodes": ["ATK-01", "PC-01", "FILE-SRV"],
        "chain_length": 2,
        "attack_types": ["Brute Force", "DoS"],
    }
    plot_attack_chain(chain)

    print("\n[OK] Visualization test done.")
