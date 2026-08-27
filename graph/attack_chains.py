"""
Attack Chain Extraction — Reconstruct temporal attack propagation paths.
=========================================================================

Extracts ordered sequences of attack propagation through the network:
    If A→B at t1 and B→C at t2 (t2 > t1, within threshold),
    then A→B→C is a propagation chain.

These chains are used for:
1. Attack path visualization (Phase 8)
2. Recommendation engine context (Phase 6)
3. Feature engineering (chain proximity as a feature)

Usage:
    from graph.attack_chains import extract_attack_chains
    chains_df = extract_attack_chains(df_processed)
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.config import PATHS, TIME_CONFIG


def extract_attack_chains(
    df: pd.DataFrame,
    max_gap_minutes: int = 15,
    min_chain_length: int = 2,
) -> pd.DataFrame:
    """
    Extract temporal attack propagation chains from flow data.

    A chain is formed when the destination of one attack flow becomes the
    source of a subsequent attack flow within max_gap_minutes.

    Args:
        df: Preprocessed DataFrame with columns:
            src_device, dst_device, is_attack, timestamp, label, window_id
        max_gap_minutes: Maximum time gap (minutes) between consecutive
            chain links. Default 15 (matches prediction horizon).
        min_chain_length: Minimum number of links to keep a chain.

    Returns:
        DataFrame with columns:
            chain_id, path_nodes, path_edges, timestamps,
            attack_types, window_ids, chain_length, duration_seconds
    """
    print(f"\n{'='*70}")
    print("ATTACK CHAIN EXTRACTION")
    print(f"{'='*70}")
    print(f"  Max gap:         {max_gap_minutes} minutes")
    print(f"  Min chain length: {min_chain_length} links")

    # Filter to attack flows only
    attack_flows = df[df["is_attack"] == 1].copy()

    if len(attack_flows) == 0:
        print("  WARNING: No attack flows found!")
        return pd.DataFrame(columns=[
            "chain_id", "path_nodes", "path_edges", "timestamps",
            "attack_types", "window_ids", "chain_length", "duration_seconds",
        ])

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(attack_flows["timestamp"]):
        attack_flows["timestamp"] = pd.to_datetime(attack_flows["timestamp"])

    # Sort by timestamp
    attack_flows = attack_flows.sort_values("timestamp").reset_index(drop=True)

    print(f"  Attack flows:    {len(attack_flows):,}")
    print(f"  Time range:      {attack_flows['timestamp'].min()} -> {attack_flows['timestamp'].max()}")

    max_gap = pd.Timedelta(minutes=max_gap_minutes)

    # ── Build attack events as (timestamp, src, dst, label, window_id) tuples ──
    events = []
    for _, row in attack_flows.iterrows():
        events.append({
            "timestamp": row["timestamp"],
            "src": row["src_device"],
            "dst": row["dst_device"],
            "label": row.get("label", "ATTACK"),
            "window_id": row.get("window_id", -1),
        })

    # ── Greedy chain building ──
    # For each event, try to extend an existing chain where the last
    # destination matches this event's source, and the time gap is within
    # threshold. If no chain matches, start a new one.

    chains = []  # Each chain: list of events
    used = set()  # Indices of events already in chains

    # Build index: for each device, sorted list of event indices where
    # that device is the SOURCE
    src_index = {}
    for i, ev in enumerate(events):
        src_index.setdefault(ev["src"], []).append(i)

    # Start chains from each event not yet used
    for start_idx in range(len(events)):
        if start_idx in used:
            continue

        # Start a new chain
        chain = [events[start_idx]]
        used.add(start_idx)
        current_dst = events[start_idx]["dst"]
        current_time = events[start_idx]["timestamp"]

        # Try to extend the chain
        extending = True
        while extending:
            extending = False
            # Look for events where src == current_dst
            candidates = src_index.get(current_dst, [])
            best_next = None
            best_time_diff = None

            for cand_idx in candidates:
                if cand_idx in used:
                    continue
                cand = events[cand_idx]
                time_diff = cand["timestamp"] - current_time
                if time_diff > pd.Timedelta(0) and time_diff <= max_gap:
                    if best_next is None or time_diff < best_time_diff:
                        best_next = cand_idx
                        best_time_diff = time_diff

            if best_next is not None:
                chain.append(events[best_next])
                used.add(best_next)
                current_dst = events[best_next]["dst"]
                current_time = events[best_next]["timestamp"]
                extending = True

        if len(chain) >= min_chain_length:
            chains.append(chain)

    # ── Convert chains to DataFrame ──
    records = []
    for chain_id, chain in enumerate(chains):
        # Build path: first src, then all destinations
        path_nodes = [chain[0]["src"]] + [ev["dst"] for ev in chain]
        path_edges = [(ev["src"], ev["dst"]) for ev in chain]
        timestamps = [str(ev["timestamp"]) for ev in chain]
        attack_types = list(set(ev["label"] for ev in chain))
        window_ids = sorted(set(ev["window_id"] for ev in chain))

        duration = (chain[-1]["timestamp"] - chain[0]["timestamp"]).total_seconds()

        records.append({
            "chain_id": chain_id,
            "path_nodes": "|".join(path_nodes),
            "path_edges": "|".join(f"{s}->{d}" for s, d in path_edges),
            "timestamps": "|".join(timestamps),
            "attack_types": "|".join(sorted(attack_types)),
            "window_ids": "|".join(str(w) for w in window_ids),
            "chain_length": len(chain),
            "duration_seconds": duration,
            "start_time": str(chain[0]["timestamp"]),
            "end_time": str(chain[-1]["timestamp"]),
        })

    chains_df = pd.DataFrame(records)

    # ── Statistics ──
    print(f"\n  --- Chain Statistics ---")
    print(f"  Total chains found:     {len(chains_df)}")
    if len(chains_df) > 0:
        print(f"  Chain lengths:          min={chains_df['chain_length'].min()}, "
              f"max={chains_df['chain_length'].max()}, "
              f"mean={chains_df['chain_length'].mean():.1f}")
        print(f"  Chain duration (sec):   min={chains_df['duration_seconds'].min():.0f}, "
              f"max={chains_df['duration_seconds'].max():.0f}, "
              f"mean={chains_df['duration_seconds'].mean():.0f}")

        # Most common source nodes (chain starters)
        all_paths = chains_df["path_nodes"].str.split("|")
        chain_starts = all_paths.apply(lambda p: p[0])
        start_counts = chain_starts.value_counts()
        print(f"\n  --- Most Common Chain Sources ---")
        for dev, count in start_counts.head(5).items():
            print(f"    {dev:<25s} started {count} chains")

        # Most common final targets
        chain_ends = all_paths.apply(lambda p: p[-1])
        end_counts = chain_ends.value_counts()
        print(f"\n  --- Most Common Chain Targets (Final Hop) ---")
        for dev, count in end_counts.head(5).items():
            print(f"    {dev:<25s} final target in {count} chains")

        # Attack type distribution
        print(f"\n  --- Attack Types in Chains ---")
        all_types = set()
        for types_str in chains_df["attack_types"]:
            all_types |= set(types_str.split("|"))
        for t in sorted(all_types):
            if t:
                count = chains_df["attack_types"].str.contains(t, regex=False).sum()
                print(f"    {t:<30s} in {count} chains")
    else:
        print("  No chains found (attacks may be isolated, not chained)")

    # Count events that were NOT part of any chain
    unchained = len(events) - len(used)
    print(f"\n  Unchained attack events: {unchained}/{len(events)} "
          f"({unchained/max(len(events),1)*100:.1f}%)")

    print(f"{'='*70}")

    return chains_df


def get_chains_for_window(
    chains_df: pd.DataFrame,
    window_id: int,
) -> list[dict]:
    """
    Get all chains that pass through a specific time window.

    Args:
        chains_df: Output from extract_attack_chains().
        window_id: Window to query.

    Returns:
        List of chain dicts containing path, timestamps, etc.
    """
    if len(chains_df) == 0:
        return []

    matching = chains_df[
        chains_df["window_ids"].str.contains(str(window_id), regex=False)
    ]

    results = []
    for _, row in matching.iterrows():
        results.append({
            "chain_id": row["chain_id"],
            "path_nodes": row["path_nodes"].split("|"),
            "path_edges": row["path_edges"].split("|"),
            "attack_types": row["attack_types"].split("|"),
            "chain_length": row["chain_length"],
            "duration_seconds": row["duration_seconds"],
        })

    return results


def get_chain_summary(chains_df: pd.DataFrame) -> dict:
    """
    Get a high-level summary of all extracted chains.

    Returns:
        Dictionary with summary statistics.
    """
    if len(chains_df) == 0:
        return {
            "total_chains": 0,
            "avg_length": 0,
            "max_length": 0,
            "unique_sources": 0,
            "unique_targets": 0,
        }

    all_paths = chains_df["path_nodes"].str.split("|")
    sources = set(all_paths.apply(lambda p: p[0]))
    targets = set(all_paths.apply(lambda p: p[-1]))

    return {
        "total_chains": len(chains_df),
        "avg_length": chains_df["chain_length"].mean(),
        "max_length": chains_df["chain_length"].max(),
        "avg_duration_seconds": chains_df["duration_seconds"].mean(),
        "unique_sources": len(sources),
        "unique_targets": len(targets),
        "sources": sorted(sources),
        "targets": sorted(targets),
    }


# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testing attack chain extraction...")

    processed_path = PATHS["data_processed"] / "cicids2017_wednesday_processed.csv"
    if not processed_path.exists():
        print(f"[ERROR] Processed data not found: {processed_path}")
        sys.exit(1)

    df = pd.read_csv(processed_path, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"Loaded {len(df):,} rows, {df['is_attack'].sum():,} attack flows")

    # Extract chains
    chains_df = extract_attack_chains(df, max_gap_minutes=15)

    # Summary
    summary = get_chain_summary(chains_df)
    print(f"\nChain Summary: {summary}")

    # Check a specific window
    if len(chains_df) > 0:
        sample_window = df[df["is_attack"] == 1]["window_id"].mode().iloc[0]
        window_chains = get_chains_for_window(chains_df, sample_window)
        print(f"\nChains through window {sample_window}: {len(window_chains)}")
        for ch in window_chains[:3]:
            print(f"  Chain {ch['chain_id']}: {' -> '.join(ch['path_nodes'])}")

    # Save
    output_path = PATHS["data_processed"] / "attack_chains_wednesday.csv"
    chains_df.to_csv(output_path, index=False)
    print(f"\n[OK] Attack chains saved: {output_path}")
    print(f"\n[OK] Attack chain extraction test passed.")
