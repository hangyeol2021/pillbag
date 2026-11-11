import json, glob, collections, statistics as st
rows = collections.defaultdict(list)
for f in glob.glob("results/*_s*.json"):
    r = json.load(open(f))
    key = r["model"].split("/")[-1] + (f" frac{r['train_frac']}" if r.get("train_frac", 1) < 1 else "")
    for mode in ("plain", "fact_rerank"):
        rows[(key, mode)].append(r[mode])
base = json.load(open("results/baselines.json"))
out = {}
print(f"{'method':42s} {'n':>2s} " + " ".join(f"{k:>16s}" for k in ["SARI","chrF","num_recall","num_halluc_per100","neg_lost_pct","hard_terms_per_text","len_ratio"]))
for k, v in base.items():
    print(f"{k:42s} {1:>2d} " + " ".join(f"{v[m]:>16}" for m in ["SARI","chrF","num_recall","num_halluc_per100","neg_lost_pct","hard_terms_per_text","len_ratio"]))
    out[k] = {m: [v[m], 0.0] for m in v}
for (key, mode), lst in sorted(rows.items()):
    agg = {m: [round(st.mean(x[m] for x in lst), 2), round(st.pstdev([x[m] for x in lst]), 2)] for m in lst[0]}
    out[f"{key} | {mode}"] = agg | {"n_seeds": len(lst)}
    print(f"{key+' | '+mode:42s} {len(lst):>2d} " + " ".join(f"{agg[m][0]:>9}±{agg[m][1]:<6}" for m in ["SARI","chrF","num_recall","num_halluc_per100","neg_lost_pct","hard_terms_per_text","len_ratio"]))
json.dump(out, open("results/summary.json", "w"), ensure_ascii=False, indent=1)
