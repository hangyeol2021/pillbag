"""대형 언어모델(무료 공개 Qwen2.5-7B-Instruct, 학교 GPU에서 실행) 지시문만으로 변환(학습 없음).
사용: python llm_baseline.py [--n 0(전체)]
"""
import argparse, json, os, time, torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from metrics import evaluate_all
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=0); ap.add_argument("--bs", type=int, default=16)
A = ap.parse_args()
M = "Qwen/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16, device_map="cuda"); model.eval()
te = pd.read_csv("data/split_test.csv")
if A.n: te = te.head(A.n)
SYS = ("당신은 의약품 설명서를 환자용 쉬운 한국어로 바꾸는 도우미입니다. "
       "의미를 바꾸지 말고, 숫자·용량·횟수·나이·기간과 금지/주의 사항은 그대로 유지하세요. "
       "새로운 정보를 추가하지 마세요. 존댓말(-하십시오/-합니다)로, 바꾼 문장만 출력하세요.")
def prompt(r):
    msg = [{"role": "system", "content": SYS},
           {"role": "user", "content": f"항목: {r['section']}\n원문: {r['source_expert']}\n쉬운 설명:"}]
    return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
outs, t0 = [], time.time()
rows = list(te.iterrows())
for i in range(0, len(rows), A.bs):
    ps = [prompt(r) for _, r in rows[i:i + A.bs]]
    enc = tok(ps, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=320, do_sample=False)
    outs += [t.strip() for t in tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)]
    if (i // A.bs) % 10 == 0: print(i + len(ps), "/", len(rows), f"{time.time()-t0:.0f}s", flush=True)
res = evaluate_all(te.source_expert.tolist(), outs, te.target_easy.tolist())
print(json.dumps(res, ensure_ascii=False))
os.makedirs("results", exist_ok=True)
json.dump({"model": M, "n": len(te), "zero_shot": res}, open("results/llm_qwen7b.json", "w"), ensure_ascii=False, indent=1)
pd.DataFrame({"section": te.section, "source": te.source_expert, "reference": te.target_easy, "llm": outs}).to_csv("results/pred_llm_qwen7b.csv", index=False, encoding="utf-8-sig")
