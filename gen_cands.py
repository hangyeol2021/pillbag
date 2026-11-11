"""저장된 체크포인트로 val/test 빔 후보 생성 (beam4x4, beam8x8). 재순위화 실험용.
사용: python gen_cands.py kobart-base-v2_s0 [kobart-base-v2_s1 ...]"""
import sys, json, os, time, torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tok = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")
os.makedirs("results/cands", exist_ok=True)
D = {s: pd.read_csv(f"data/split_{s}.csv") for s in ["val", "test"]}
for name in sys.argv[1:]:
    out_p = f"results/cands/{name}.json"
    if os.path.exists(out_p): print("skip", name); continue
    base = "gogamza/kobart-summarization" if "summarization" in name else "gogamza/kobart-base-v2"
    m = AutoModelForSeq2SeqLM.from_pretrained(base); m.load_state_dict(torch.load(f"ckpt/{name}.pt", map_location="cpu")); m.to("cuda").eval()
    res, t0 = {}, time.time()
    for s, d in D.items():
        src = [f"[{a}] {b}" for a, b in zip(d.section, d.source_expert)]
        order = sorted(range(len(src)), key=lambda i: len(src[i]))
        for nb in (4, 8):
            outs = [None] * len(src)
            for i in range(0, len(order), 24):
                idx = order[i:i + 24]
                enc = tok([src[j] for j in idx], max_length=512, truncation=True, padding=True, return_tensors="pt").to("cuda")
                enc.pop("token_type_ids", None)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    g = m.generate(**enc, num_beams=nb, num_return_sequences=nb, max_new_tokens=256, early_stopping=True, no_repeat_ngram_size=4)
                txt = tok.batch_decode(g, skip_special_tokens=True)
                for k, j in enumerate(idx): outs[j] = [t.strip() for t in txt[k * nb:(k + 1) * nb]]
            res[f"{s}_b{nb}"] = outs
            print(name, s, nb, f"{time.time()-t0:.0f}s", flush=True)
    json.dump(res, open(out_p, "w"), ensure_ascii=False)
    del m; torch.cuda.empty_cache()
