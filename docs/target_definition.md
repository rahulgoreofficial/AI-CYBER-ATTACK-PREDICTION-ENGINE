# Target Definition — Next-Target Prediction
# =============================================

## Core Concept

The prediction target is NOT the attack label on individual network flows.

It is a **temporal, device-level prediction**:

> **"Will device D become an attack destination within the next H minutes?"**

## Formal Definition

```
For a given time window T and prediction horizon H:

  Target(device_d, window_T) = 1
    IF device_d appears as an attack DESTINATION
    in any attack-labeled flow during time interval [T+1, T+H]

  Target(device_d, window_T) = 0
    OTHERWISE
```

## Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Window size | 5 minutes | Granular enough for sequence detection |
| Prediction horizon (H) | 15 minutes (3 windows) | Enough lead time for defensive action |
| Lookback (k) | 3 windows (15 minutes) | Sufficient historical context |

## Input → Output

```
INPUT:  Features computed from all events in windows [T-k, ..., T]
OUTPUT: For each candidate device, P(target in [T+1, ..., T+H])
```

## Anti-Leakage Guarantees

1. Features use ONLY events at or before time T
2. Target labels use ONLY events strictly after time T
3. Train/test split is TEMPORAL (train on earlier days, test on later)
4. No shuffling across time boundaries
5. Graph features computed from historical windows only
6. Anomaly scores computed from historical data only

## Label Generation Process

```python
# Pseudocode
for each time_window T:
    # Get future attack destinations
    future_events = events[(events.timestamp > T.end) &
                           (events.timestamp <= T.end + H)]
    attack_events = future_events[future_events.label != 'BENIGN']
    future_targets = set(attack_events.dst_id.unique())

    # Label each device
    for device in all_devices:
        target_label[device, T] = 1 if device in future_targets else 0
```

## Class Imbalance

The target will be heavily imbalanced (most devices are NOT targets in any given window).

Mitigation strategies:
- Class weights in XGBoost (`scale_pos_weight`)
- PR-AUC as primary metric (not accuracy)
- Top-K ranking evaluation (not just binary classification)
- Consider time windows containing attack activity only for training

## Evaluation as Ranking

The primary evaluation is RANKING, not binary classification:

| Metric | Question Answered |
|---|---|
| Top-1 Hit Rate | Did the actual target appear at rank 1? |
| Top-3 Hit Rate | Was it in the top 3? |
| Top-5 Hit Rate | Was it in the top 5? |
| MRR | How high was it ranked on average? |
| Early Warning | How much time before the actual event? |
| PR-AUC | How well does probability separate targets from non-targets? |
