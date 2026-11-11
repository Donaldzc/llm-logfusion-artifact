"""LOPO target-domain threshold recalibration for BGL / DNS."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".pylibs"))

from llm_logfusion import config                              # noqa: E402
from llm_logfusion.embedding import SemanticEncoder           # noqa: E402
from llm_logfusion.evaluation import classification_metrics   # noqa: E402
from llm_logfusion.pipeline import LLMLogFusion, PipelineConfig  # noqa: E402
from scripts.run_reproduction import fusion_block, struct_features  # noqa: E402

df = pd.read_csv(config.DATA_DIR / "extended_logs.csv")
emb = SemanticEncoder(backend="auto", attention="nystrom").fit(df["raw"].tolist()).encode_cached(
    df["raw"].tolist(), tag="repro")
rows = []
for held in ("BGL", "DNS"):
    te = np.flatnonzero((df["platform"] == held).to_numpy())
    tr = np.flatnonzero((df["platform"] != held).to_numpy())
    s_tr, s_te = struct_features(df, tr, te)
    X_tr, X_te = fusion_block(emb, tr, s_tr), fusion_block(emb, te, s_te)
    train_df = df.iloc[tr].reset_index(drop=True)
    test_df = df.iloc[te].reset_index(drop=True)
    y = test_df["label"].to_numpy(dtype=int)
    model = LLMLogFusion(PipelineConfig(seed=0)).fit(train_df, X_tr, s_tr)
    scores = model._filter.score_samples(model._filter_matrix(X_te, s_te))
    base = model.predict(test_df, X_te, s_te)
    m0 = classification_metrics(y, base.y_pred)
    m0.update({"held_out": held, "setting": "source threshold",
               "call_rate": base.n_llm_calls / len(test_df)})
    rows.append(m0)
    print(f"[{held}] source thr={model._filter.threshold_:.4f} F1={m0['f1']:.3f} "
          f"P={m0['precision']:.3f} R={m0['recall']:.3f} FPR={m0['fpr']:.3f}", flush=True)
    for q in (0.90, 0.95, 1.0 - float(np.mean(y))):
        model._filter.threshold_ = float(np.quantile(scores, q))
        r = model.predict(test_df, X_te, s_te)
        m = classification_metrics(y, r.y_pred)
        m.update({"held_out": held, "setting": f"target quantile {q:.2f}",
                  "call_rate": r.n_llm_calls / len(test_df)})
        rows.append(m)
        print(f"[{held}] q={q:.2f} F1={m['f1']:.3f} P={m['precision']:.3f} "
              f"R={m['recall']:.3f} FPR={m['fpr']:.3f} call={m['call_rate']:.3f}", flush=True)
out = pd.DataFrame(rows)[["held_out", "setting", "precision", "recall", "f1", "fpr", "call_rate"]]
out.to_csv(config.TABLE_DIR / "review_lopo_recalibration.csv", index=False)
print(out.round(4).to_string(index=False))
