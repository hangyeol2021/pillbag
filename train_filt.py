"""한국어 seq2seq 트랜스포머 미세조정 (전문 설명 -> 쉬운 설명).
사용: python train.py --model gogamza/kobart-base-v2 --seed 0
      python train.py --model paust/pko-t5-base --seed 0 --lr 3e-4
- 입력 앞에 항목 표시([효능]/[용법]/[주의사항])를 붙인다.
- 검증 손실로 조기 종료, 가장 좋은 가중치 저장.
- 테스트: 빔 서치(beam=4) 기본 출력 + '사실 보존 재순위화'(제안):
  빔 후보 4개 중 원문의 숫자 보존·부정 보존 점수가 가장 높은 후보를 고르고, 동점이면 모델 점수 순.
"""
import argparse, json, os, time, math, random
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from metrics import evaluate_all, fact_scores

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gogamza/kobart-base-v2")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--max_src", type=int, default=512)
ap.add_argument("--max_tgt", type=int, default=256)
ap.add_argument("--beams", type=int, default=4)
ap.add_argument("--train_frac", type=float, default=1.0, help="학습 데이터 비율(데이터 양 실험)")
ap.add_argument("--tag", default="")
A = ap.parse_args()
torch.manual_seed(A.seed); np.random.seed(A.seed); random.seed(A.seed)
dev = "cuda"
name = A.model.split("/")[-1] + f"_s{A.seed}" + (f"_f{A.train_frac}" if A.train_frac < 1 else "") + A.tag
os.makedirs("results", exist_ok=True); os.makedirs("ckpt", exist_ok=True)

tok = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")  # summarization 저장소 토크나이저는 잘못 로드됨(RobertaTokenizer) -> 같은 어휘의 base 토크나이저 사용
model = AutoModelForSeq2SeqLM.from_pretrained(A.model).to(dev)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
is_bart = "bart" in A.model.lower()

def inp(row):
    return f"[{row['section']}] {row['source_expert']}"

def load(s):
    d = pd.read_csv(f"data/split_{s}.csv")
    return d

tr, va, te = load("train"), load("val"), load("test")
_fs = [fact_scores(a, b) for a, b in zip(tr.source_expert, tr.target_easy)]
_keep = [not ((r is not None and r < 0.5) or n) for r, h, n in _fs]
print(f"[filter] train {len(tr)} -> {sum(_keep)} (dropped {len(tr)-sum(_keep)})", flush=True)
tr = tr[_keep].reset_index(drop=True)
if A.train_frac < 1:
    keys = tr.src_key.unique(); rng = np.random.default_rng(A.seed); rng.shuffle(keys)
    tr = tr[tr.src_key.isin(set(keys[:int(len(keys) * A.train_frac)]))]
print(f"train {len(tr)} val {len(va)} test {len(te)}", flush=True)

def encode(d):
    x = tok([inp(r) for _, r in d.iterrows()], max_length=A.max_src, truncation=True)
    y = tok(d.target_easy.astype(str).tolist(), max_length=A.max_tgt, truncation=True)
    if is_bart:  # KoBART: 문장 끝 토큰 보장
        y["input_ids"] = [ids if ids[-1] == tok.eos_token_id else ids + [tok.eos_token_id] for ids in y["input_ids"]]
    return list(zip(x["input_ids"], y["input_ids"]))

def batches(data, bs, shuffle):
    idx = list(range(len(data)))
    if shuffle: random.shuffle(idx)
    else: idx.sort(key=lambda i: len(data[i][0]))
    for i in range(0, len(idx), bs):
        b = [data[j] for j in idx[i:i + bs]]
        L = max(len(a) for a, _ in b); T = max(len(c) for _, c in b)
        x = torch.full((len(b), L), tok.pad_token_id); m = torch.zeros((len(b), L), dtype=torch.long)
        y = torch.full((len(b), T), -100)
        for k, (a, c) in enumerate(b):
            x[k, :len(a)] = torch.tensor(a); m[k, :len(a)] = 1; y[k, :len(c)] = torch.tensor(c)
        yield x.to(dev), m.to(dev), y.to(dev)

Dtr, Dva = encode(tr), encode(va)
opt = torch.optim.AdamW(model.parameters(), lr=A.lr, weight_decay=0.01)
steps = A.epochs * math.ceil(len(Dtr) / A.bs); warm = int(0.06 * steps)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / max(1, warm)) * max(0.0, (steps - s) / max(1, steps - warm)))
scaler_dtype = torch.bfloat16
best, bad, t0 = 1e9, 0, time.time()
for ep in range(A.epochs):
    model.train(); tl = 0; nb = 0
    for x, m, y in batches(Dtr, A.bs, True):
        with torch.autocast("cuda", dtype=scaler_dtype):
            loss = model(input_ids=x, attention_mask=m, labels=y).loss
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        tl += loss.item(); nb += 1
    model.eval(); vl = 0; vn = 0
    with torch.no_grad():
        for x, m, y in batches(Dva, 32, False):
            with torch.autocast("cuda", dtype=scaler_dtype):
                vl += model(input_ids=x, attention_mask=m, labels=y).loss.item(); vn += 1
    vl /= vn
    print(f"epoch {ep+1} train {tl/nb:.4f} val {vl:.4f} ({time.time()-t0:.0f}s)", flush=True)
    if vl < best - 1e-4:
        best, bad = vl, 0; torch.save(model.state_dict(), f"ckpt/{name}.pt")
    else:
        bad += 1
        if bad >= 2: break
model.load_state_dict(torch.load(f"ckpt/{name}.pt")); model.eval()

def generate(d, nret=1):
    outs = []
    rows = list(d.iterrows())
    for i in range(0, len(rows), 16):
        chunk = rows[i:i + 16]
        enc = tok([inp(r) for _, r in chunk], max_length=A.max_src, truncation=True, padding=True, return_tensors="pt").to(dev)
        enc.pop("token_type_ids", None)
        with torch.no_grad(), torch.autocast("cuda", dtype=scaler_dtype):
            g = model.generate(**enc, num_beams=A.beams, num_return_sequences=nret, max_new_tokens=A.max_tgt,
                               early_stopping=True, no_repeat_ngram_size=4, output_scores=False)
        txt = tok.batch_decode(g, skip_special_tokens=True)
        for k in range(len(chunk)):
            outs.append([t.strip() for t in txt[k * nret:(k + 1) * nret]])
    return outs

cands = generate(te, nret=A.beams)
plain = [c[0] for c in cands]
def fact_key(src, c):
    rec, hal, neg = fact_scores(src, c)
    return ((rec if rec is not None else 1.0) - 0.5 * hal - 1.0 * neg)
rerank = []
for (_, r), c in zip(te.iterrows(), cands):
    sc = [fact_key(r.source_expert, x) for x in c]
    best_i = max(range(len(c)), key=lambda i: (sc[i], -i))
    rerank.append(c[best_i])
srcs, refs = te.source_expert.tolist(), te.target_easy.tolist()
res = {"model": A.model, "seed": A.seed, "train_frac": A.train_frac, "n_train": len(tr), "best_val_loss": best,
       "plain": evaluate_all(srcs, plain, refs), "fact_rerank": evaluate_all(srcs, rerank, refs),
       "train_sec": round(time.time() - t0)}
print(json.dumps(res, ensure_ascii=False), flush=True)
json.dump(res, open(f"results/{name}.json", "w"), ensure_ascii=False, indent=1)
pd.DataFrame({"section": te.section, "item_name": te.item_name, "source": srcs, "reference": refs, "plain": plain, "fact_rerank": rerank}).to_csv(
    f"results/pred_{name}.csv", index=False, encoding="utf-8-sig")
