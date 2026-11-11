"""저장된 체크포인트로 테스트셋 빔 후보 4개를 다시 만들고, 수정된 지표(목록 번호 제외)로 재순위화·평가.
train.py의 결과 json/pred csv를 덮어쓴다(학습 자체는 다시 하지 않음). 후보는 results/cands_<name>.json에 저장.
사용: python regen.py ckpt/kobart-base-v2_s0.pt ...
"""
import sys, json, torch, pandas as pd, os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from metrics import evaluate_all, fact_scores
tok = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")
te = pd.read_csv("data/split_test.csv").fillna("")
srcs, refs = te.source_expert.tolist(), te.target_easy.tolist()
def fact_key(src, c):
    r, h, n = fact_scores(src, c); return (r if r is not None else 1.0) - 0.5 * h - 1.0 * n
for ck in sys.argv[1:]:
    name = os.path.basename(ck)[:-3]
    base = "gogamza/kobart-summarization" if "summarization" in name else "gogamza/kobart-base-v2"
    m = AutoModelForSeq2SeqLM.from_pretrained(base); m.load_state_dict(torch.load(ck, map_location="cpu")); m.to("cuda").eval()
    cands = []
    for i in range(0, len(te), 16):
        ch = te.iloc[i:i + 16]
        enc = tok([f"[{r.section}] {r.source_expert}" for r in ch.itertuples()], max_length=512, truncation=True, padding=True, return_tensors="pt").to("cuda")
        enc.pop("token_type_ids", None)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            g = m.generate(**enc, num_beams=4, num_return_sequences=4, max_new_tokens=256, early_stopping=True, no_repeat_ngram_size=4)
        t = tok.batch_decode(g, skip_special_tokens=True)
        cands += [[x.strip() for x in t[k * 4:(k + 1) * 4]] for k in range(len(ch))]
    plain = [c[0] for c in cands]
    rr = [max(c, key=lambda x, s=s: fact_key(s, x)) for c, s in zip(cands, srcs)]
    js = f"results/{name}.json"; d = json.load(open(js)) if os.path.exists(js) else {"model": base}
    d["plain"] = evaluate_all(srcs, plain, refs); d["fact_rerank"] = evaluate_all(srcs, rr, refs); d["metric_version"] = 2
    json.dump(d, open(js, "w"), ensure_ascii=False, indent=1)
    json.dump(cands, open(f"results/cands_{name}.json", "w"), ensure_ascii=False)
    pd.DataFrame({"section": te.section, "item_name": te.item_name, "source": srcs, "reference": refs, "plain": plain, "fact_rerank": rr}).to_csv(f"results/pred_{name}.csv", index=False, encoding="utf-8-sig")
    print(name, "plain", d["plain"], "\n  rerank", d["fact_rerank"], flush=True)
    del m; torch.cuda.empty_cache()
