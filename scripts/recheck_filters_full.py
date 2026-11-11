"""Full-corpus recheck of the three scalable first-layer filters (Table 2 subset)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".pylibs"))

from llm_logfusion import config                                # noqa: E402
from llm_logfusion.embedding import SemanticEncoder             # noqa: E402
from llm_logfusion.evaluation import classification_metrics     # noqa: E402
from llm_logfusion.filters import build_filter                  # noqa: E402
from scripts.run_reproduction import fusion_block, struct_features  # noqa: E402

NAMES = ["Isolation Forest", "HBOS", "AutoEncoder"]
rows = []
df = pd.read_csv(config.DATA_DIR / "extended_logs.csv")
emb = SemanticEncoder(backend="auto", attention="nystrom").fit(df["raw"].tolist()).encode_cached(
    df["raw"].tolist(), tag="repro")
for seed in range(5):
    tr, te = train_test_split(np.arange(len(df)), test_size=0.5,
                              stratify=df["label"], random_state=seed)
    s_tr, s_te = struct_features(df, tr, te)
    X_tr = fusion_block(emb, tr, s_tr)
    X_te = fusion_block(emb, te, s_te)
    y = df["label"].to_numpy()[te]
    for name in NAMES:
        f = build_filter(name, contamination=0.10, random_state=seed).fit(X_tr)
        m = classification_metrics(y, (f.score_samples(X_te) >= f.threshold_).astype(int))
        m.update({"filter": name, "seed": seed})
        rows.append(m)
        print(f"[{name}] seed={seed} F1={m['f1']:.3f}", flush=True)
out = pd.DataFrame(rows)
out.to_csv(config.TABLE_DIR / "review_filters_full.csv", index=False)
print(out.groupby("filter")[["f1", "precision", "recall"]].agg(["mean", "std"]).round(3).to_string())
