# CICIDS2017 & CSE-CIC-IDS2018 — Dataset Notes
# ================================================

## CICIDS2017

### Source
- **Publisher**: Canadian Institute for Cybersecurity (CIC), University of New Brunswick
- **URL**: https://www.unb.ca/cic/datasets/ids-2017.html
- **Alternative Mirror**: https://www.kaggle.com/datasets/ciaboroiu/cicids2017

### Download Instructions

1. **Option A — Official UNB Site**:
   - Visit: https://www.unb.ca/cic/datasets/ids-2017.html
   - Download the "MachineLearningCVE" CSV files
   - Extract into: `data/raw/cicids2017/`

2. **Option B — Kaggle** (recommended for reliability):
   - Visit: https://www.kaggle.com/datasets/ciaboroiu/cicids2017
   - Download and extract into: `data/raw/cicids2017/`

### Files Expected After Download

```
data/raw/cicids2017/
├── Monday-WorkingHours.pcap_ISCX.csv
├── Tuesday-WorkingHours.pcap_ISCX.csv
├── Wednesday-workingHours.pcap_ISCX.csv        ← PRIMARY (start here)
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
├── Thursday-WorkingHours-Afternoon-Infiltration.pcap_ISCX.csv
├── Friday-WorkingHours-Morning.pcap_ISCX.csv
└── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

### Day-by-Day Attack Content

| Day | Attacks | Notes |
|---|---|---|
| Monday | None (benign only) | Baseline normal traffic |
| Tuesday | Brute Force (FTP, SSH) | Good for targeted attack sequences |
| **Wednesday** | **DoS (Slowloris, Hulk, GoldenEye), Heartbleed** | **Best variety — start here** |
| Thursday AM | Web Attacks (XSS, SQL Injection, Brute Force) | Application-layer attacks |
| Thursday PM | Infiltration | Lateral movement — excellent for propagation |
| Friday AM | Botnet (Ares) | Multi-host coordinated |
| Friday PM | DDoS (LOIT), Port Scan | Volume-based attacks |

### Known Issues & Preprocessing Notes

1. **Column names have leading spaces** (e.g., ` Destination Port` instead of `Destination Port`)
   - Fix: Strip whitespace from column names during preprocessing
2. **Infinity values** in `Flow Bytes/s` and `Flow Packets/s`
   - Fix: Replace with NaN, then impute or cap
3. **NaN values** scattered across multiple columns
   - Fix: Drop rows with excessive NaN, impute others
4. **Duplicate rows** exist
   - Fix: Deduplicate
5. **Timestamp format** varies; some entries may not parse cleanly
   - Fix: Use `pd.to_datetime` with `dayfirst=True` and error handling
6. **Class imbalance**: Benign traffic vastly outnumbers attack traffic
   - Handle via: class weights, oversampling, or threshold tuning
7. **No explicit device/host IDs**: Only IP addresses available
   - Fix: Map IPs to synthetic device IDs using the campus topology

### Feature Count
~78 flow-level features per row (after cleaning)

### Approximate Size
- Wednesday file: ~700K rows, ~200MB
- All files combined: ~2.8M rows

---

## CSE-CIC-IDS2018

### Source
- **Publisher**: Communications Security Establishment (CSE) + CIC
- **URL**: https://www.unb.ca/cic/datasets/ids-2018.html
- **AWS**: Available via AWS Open Data program

### When to Use
- After the CICIDS2017 pipeline is working
- For richer enterprise-style evaluation
- Contains more complex multi-stage attacks

### Attack Scenarios
- Brute Force, Heartbleed, Botnet, DoS, DDoS, Web Attacks, Infiltration
- Multi-department simulated organization

### Known Differences from CICIDS2017
- Larger dataset (~16M records across 10 days)
- Slightly different column naming
- More realistic enterprise topology simulation
- May require additional preprocessing adjustments

---

## Synthetic Campus Topology

### File
`data/synthetic/campus_topology.json`

### Contents
- 21 devices across 7 network segments
- Segments: Internet, DMZ, Core, Student VLAN, Faculty VLAN, Admin VLAN, Server VLAN
- Each device has: criticality, vulnerability, OS, department, open ports
- Physical and logical connections defined
- IP range mapping placeholders for CICIDS2017 integration

### Important Note
All criticality and vulnerability scores are **synthetic/assigned** values for demonstration purposes. They do NOT represent real-world security assessments.

---

## Data Versioning

| Dataset | Version Used | Date Acquired | Hash/Notes |
|---|---|---|---|
| CICIDS2017 | MachineLearningCVE | TBD | SHA256 TBD after download |
| CSE-CIC-IDS2018 | — | TBD | Phase 2 |
| Campus Topology | v1.0 | Created with project | 21 devices |
