"""지표 v3(금지 표현에 '안 됩니다/안 돼요' 추가)로 저장된 예측 CSV를 다시 평가. 모델 출력은 바꾸지 않는다."""
import glob, json, pandas as pd
from metrics import evaluate_all
for f in sorted(glob.glob("results/pred_*.csv")):
    p = pd.read_csv(f).fillna(""); name = f.split("pred_")[1][:-4]
    js = f"results/{name}.json"
    try: d = json.load(open(js))
    except Exception: d = {}
    for c in ("plain", "fact_rerank", "llm"):
        if c in p.columns:
            key = c if c != "llm" else "llm"
            d[key] = evaluate_all(p.source.tolist(), p[c].tolist(), p.reference.tolist())
    d["metric_version"] = 3
    json.dump(d, open(js, "w"), ensure_ascii=False, indent=1); print(name, {k: d[k]["neg_lost_pct"] for k in d if isinstance(d.get(k), dict)})
