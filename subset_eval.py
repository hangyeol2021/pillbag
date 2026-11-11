"""학습셋에 같은 쉬운 문장이 있는 테스트(seen) vs 없는 테스트(unseen)로 나눠 평가 -> results/subset.json"""
import pandas as pd, glob, json, numpy as np
from metrics import evaluate_all
flag = pd.read_csv("results/test_leakflag.csv").tgt_in_train.values
out = {}
for f in sorted(glob.glob("results/pred_*.csv")):
    p = pd.read_csv(f).fillna(""); name = f.split("pred_")[1][:-4]
    cols = [c for c in ("plain", "fact_rerank", "llm") if c in p.columns]
    for c in cols:
        for lab, m in (("seen", flag), ("unseen", ~flag)):
            q = p[m]
            out[f"{name}|{c}|{lab}"] = evaluate_all(q.source.tolist(), q[c].tolist(), q.reference.tolist())
# 기준선과 정답도 unseen에서
import re
from baselines import rule_simplify
te = pd.read_csv("data/split_test.csv").fillna("")
src = te.source_expert.tolist(); ref = te.target_easy.tolist()
for lab, m in (("seen", flag), ("unseen", ~flag)):
    idx = np.where(m)[0]
    out[f"rule|-|{lab}"] = evaluate_all([src[i] for i in idx], [rule_simplify(src[i]) for i in idx], [ref[i] for i in idx])
    out[f"reference|-|{lab}"] = evaluate_all([src[i] for i in idx], [ref[i] for i in idx], [ref[i] for i in idx])
json.dump(out, open("results/subset.json", "w"), ensure_ascii=False, indent=1)
df = pd.DataFrame(out).T; df.index.name = "run"
df["model"] = [i.split("|")[0].rsplit("_s", 1)[0] for i in df.index]
df["kind"] = [i.split("|")[1] for i in df.index]; df["subset"] = [i.split("|")[2] for i in df.index]
g = df.groupby(["subset", "model", "kind"]).agg(["mean", "std"]).round(2)
print(g[[("SARI", "mean"), ("SARI", "std"), ("chrF", "mean"), ("num_recall", "mean"), ("num_halluc_per100", "mean"), ("neg_lost_pct", "mean")]].to_string())
