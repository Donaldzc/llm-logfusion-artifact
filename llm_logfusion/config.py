"""Global configuration for the LLM-LogFusion reproduction.

All numbers are taken from the paper (csf2027-paper22.pdf) unless stated
otherwise.  Where the paper is internally inconsistent, the discrepancy is
recorded in the comment next to the value and surfaced in the generated report.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"

for _d in (DATA_DIR, CACHE_DIR, RESULTS_DIR, FIG_DIR, TABLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Nine heterogeneous platforms (Section IV-A / Table V)
# ---------------------------------------------------------------------------
PLATFORMS = ["HDFS", "BGL", "OpenStack", "Windows", "Apache",
             "Firewall", "Database", "SSH", "DNS"]

PLATFORM_COUNTS = {
    "HDFS": 5720, "BGL": 5100, "OpenStack": 4950, "Windows": 4780,
    "Apache": 4680, "Firewall": 4520, "Database": 4450, "SSH": 4370, "DNS": 4370,
}

PLATFORM_RATIOS = {
    "HDFS": 0.0470, "BGL": 0.0380, "OpenStack": 0.0401, "Windows": 0.0215,
    "Apache": 0.0850, "Firewall": 0.0248, "Database": 0.0990, "SSH": 0.0808,
    "DNS": 0.0748,
}
# NOTE: Table V also claims a 5.07 % aggregate anomaly ratio.  The per-platform
# ratios above imply 5.58 %; the two cannot both hold.  We keep the (more
# granular) per-platform ratios and report the derived aggregate ratio.
PAPER_TOTAL_LOGS = 42940
PAPER_TOTAL_RATIO = 0.0507

PLATFORM_DESC = {
    "HDFS": "Distributed File System", "BGL": "HPC Supercomputer",
    "OpenStack": "Cloud Platform", "Windows": "OS Event Logs",
    "Apache": "Web Server", "Firewall": "Network Firewall",
    "Database": "Database System", "SSH": "Secure Shell", "DNS": "DNS Server",
}

# ---------------------------------------------------------------------------
# MITRE ATT&CK 12-stage taxonomy (Table III)
# ---------------------------------------------------------------------------
STAGES = [
    "Initial Access", "Execution", "Persistence", "Privilege Escalation",
    "Defense Evasion", "Credential Access", "Discovery", "Lateral Movement",
    "Collection", "Command and Control", "Exfiltration", "Impact",
]
STAGE_INDEX = {s: i for i, s in enumerate(STAGES)}
N_STAGES = len(STAGES)

# Severity used only for knowledge-graph node colouring.
STAGE_SEVERITY = [0.90, 0.95, 0.70, 0.85, 0.60, 0.95,
                  0.45, 0.80, 0.65, 0.90, 0.95, 1.00]

# ---------------------------------------------------------------------------
# Feature dimensions (Table II)
# ---------------------------------------------------------------------------
N_TEMPORAL = 1
N_LEVEL = 3
N_PLATFORM = 9
N_CONTENT = 1
N_STRUCT = N_TEMPORAL + N_LEVEL + N_PLATFORM + N_CONTENT      # 14
N_SEMANTIC = 384                                               # all-MiniLM-L6-v2
N_FUSION = N_STRUCT + N_SEMANTIC                               # 398

# ---------------------------------------------------------------------------
# Embedding / linearised attention (Section III-C-1)
# ---------------------------------------------------------------------------
N_LANDMARKS = 64          # Nystrom landmarks k
ATTN_EPS = 1e-6           # regularisation added before inversion
EMBED_BATCH = 128         # batch processing size
SBERT_MODEL = os.environ.get("SBERT_MODEL", "all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Isolation Forest (Section III-C-2)
# ---------------------------------------------------------------------------
IF_CONTAMINATION = 0.10
IF_RANDOM_STATE = 42
IF_N_TREES = 200
IF_MAX_SAMPLES = 256
# Equal split probability per feature block.  Without this, the 384-d semantic
# block receives 384/398 of the random splits and the 14-d structural block is
# effectively ignored by the first layer.
IF_BLOCK_BALANCE = True
IF_BLOCK_SIZES = (N_STRUCT, N_SEMANTIC)

# ---------------------------------------------------------------------------
# PageRank knowledge graph (Section III-C-3)
# ---------------------------------------------------------------------------
PR_ALPHA = 0.85
PR_TOL = 1e-6
PR_MAX_ITER = 200
PR_PRUNE_FACTOR = 0.5     # prune nodes with PR(i) < 0.5 / N

# ---------------------------------------------------------------------------
# Second-stage LLM (Section IV-A)
# ---------------------------------------------------------------------------
LLM_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
LLM_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY_ENV = "DEEPSEEK_API_KEY"
LLM_TEMPERATURE = 0.0
# The paper's design value is 512.  Reasoning models spend completion tokens on
# hidden reasoning before emitting the JSON verdict, so the real-API validation
# raises the cap to 4096; the offline surrogate does not use this value.
LLM_MAX_TOKENS = int(os.environ.get("DEEPSEEK_MAX_TOKENS", "4096"))
LLM_TOP_P = 1.0
# The offline surrogate has no network latency.  To keep throughput numbers
# comparable with the paper we additionally report an *estimated* end-to-end
# time that charges every LLM call the paper's amortised latency:
#   235.95 s of LLM time / 3 822 calls = 61.7 ms per call (Section V-E).
LLM_CALL_LATENCY_S = 235.95 / 3822

# ---------------------------------------------------------------------------
# Experimental protocol
# ---------------------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]
TEST_FRACTION = 0.5
CONTAMINATION_GRID = [0.05, 0.10, 0.15, 0.20, 0.30]
FILTER_SUBSET = 10000     # LOF / OCSVM / DBSCAN cost O(n^2); evaluated on this subset
FILTER_BASELINES = ["Isolation Forest", "HBOS", "LOF", "One-Class SVM",
                    "AutoEncoder", "DBSCAN"]

# ---------------------------------------------------------------------------
# Dataset synthesis knobs
# ---------------------------------------------------------------------------
SUBTLE_ANOMALY_FRACTION = 0.05   # fraction of anomalies built from low-salience text
BENIGN_WARN_FRACTION = 0.04      # benign logs that look noisy (realistic false positives)
BENIGN_ERROR_FRACTION = 0.005
INVALID_TS_BENIGN = 0.005
INVALID_TS_ANOMALY = 0.35
ANOMALY_LEVEL_DIST = {"ERROR": 0.85, "WARN": 0.13, "INFO": 0.02}
LONG_PAYLOAD_PROB = 0.90
DATASET_START = "2024-03-12T00:00:00"
DATASET_HOURS = 72

DARPA_TRAIN_PAIRS = 2000
DARPA_TEST_PAIRS = 500
