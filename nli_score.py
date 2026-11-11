"""후보 문장별 NLI 모순 확률 (원문=전제, 후보 문장=가설). 결과: results/cands/nli_<name>.json  {split_b8: [[max_contra per cand]]}"""
import sys, json, re, torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
MN = "Huffon/klue-roberta-base-nli"
tk = AutoTokenizer.from_pretrained(MN); md = AutoModelForSequenceClassification.from_pretrained(MN).to("cuda").eval()
lab = {v.lower(): int(k) for k, v in md.config.id2label.items()}; ci = lab.get("contradiction", 2); print("labels", md.config.id2label, flush=True)
D = {s: pd.read_csv(f"data/split_{s}.csv") for s in ["val", "test"]}
def sents(t): return [x for x in re.split(r"(?<=[.다요])\s+", str(t)) if len(x) > 5] or [str(t)]
for name in sys.argv[1:]:
    C = json.load(open(f"results/cands/{name}.json", encoding="utf-8")); out = {}
    for s, d in D.items():
        for key in (f"{s}_b4", f"{s}_b8"):
            pairs, owner = [], []
            for i, (src, cs) in enumerate(zip(d.source_expert, C[key])):
                for j, c in enumerate(cs):
                    for h in sents(c): pairs.append((str(src), h)); owner.append((i, j))
            P = []
            for b in range(0, len(pairs), 24):
                enc = tk([p for p, _ in pairs[b:b+24]], [h for _, h in pairs[b:b+24]], truncation="only_first", max_length=500, padding=True, return_tensors="pt").to("cuda"); enc.pop("token_type_ids", None)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    P += torch.softmax(md(**enc).logits.float(), -1)[:, ci].tolist()
            M = [[0.0] * len(cs) for cs in C[key]]
            for (i, j), p in zip(owner, P): M[i][j] = max(M[i][j], p)
            out[key] = M; print(name, key, len(pairs), flush=True)
    json.dump(out, open(f"results/cands/nli_{name}.json", "w"))
