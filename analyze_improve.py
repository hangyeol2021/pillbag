"""성능 개선 실험 분석: 저장된 빔 후보로 재순위화 변형 비교 (가중치는 검증셋에서만 선택, 테스트는 마지막에 1회 평가).
V0 빔4 1등 / V1 빔4+기존 재순위화 / V2 빔8+기존 가중치 / V3 +가중치 튜닝 / V4 +숫자·단위 묶음 / V5 +NLI 모순 감점 / V6 +용어 풀이 후처리
학습 쪽: base(기존 학습) vs filt(정답 품질 필터 재학습)"""
import json, os, re, itertools, collections, statistics as st
import pandas as pd
import metrics as M

D = {s: pd.read_csv(f"data/split_{s}.csv") for s in ["val", "test"]}
SRC = {s: D[s].source_expert.astype(str).tolist() for s in D}; REF = {s: D[s].target_easy.astype(str).tolist() for s in D}
seen = None
try:
    lf = pd.read_csv("results/test_leakflag.csv"); col = [c for c in lf.columns if c.lower() in ("seen", "leak", "in_train", "tgt_in_train", "leak_flag")]
    if col and len(lf) == len(D["test"]): seen = lf[col[0]].astype(bool).tolist()
    print("leak col:", col, list(lf.columns)[:8])
except Exception as e: print("no leakflag", e)

UN = {"㎎": "mg", "ml": "mL", "㎖": "mL", "㎏": "kg", "번": "회", "살": "세", "mcg": "μg"}
def nums_u(t):
    t = str(t)
    for a, b in M.KNUM: t = t.replace(a, b)
    t = M.LIST_MARK.sub(" ", t); out = []
    for m in M.NUM_UNIT.finditer(t):
        v = m.group(1).replace(",", "")
        try: v = str(float(v)).rstrip("0").rstrip(".")
        except ValueError: pass
        u = m.group(2) or ""; out.append((v, UN.get(u, u)))
    return collections.Counter(out)
CACHE = {}
def feats(src, c):
    k = (src, c)
    if k not in CACHE:
        r, h, n = M.fact_scores(src, c)
        s, o = nums_u(src), nums_u(c)
        ru = sum(min(v, o[x]) for x, v in s.items()) / sum(s.values()) if s else None
        CACHE[k] = (r, h, n, ru)
    return CACHE[k]

def pick(srcs, cands, nli, w):
    a, b, g, d, unit = w; out = []
    for i, (src, cs) in enumerate(zip(srcs, cands)):
        best, bs = None, -1e9
        for j, c in enumerate(cs):
            r, h, n, ru = feats(src, c); rr = ru if unit else r
            sc = (rr if rr is not None else 1.0) - a * h - b * n - g * j - (d * nli[i][j] if (d and nli) else 0)
            if sc > bs + 1e-9: best, bs = c, sc
        out.append(best)
    return out

def quick(srcs, preds, refs):
    fr = [M.fact_scores(s, p) for s, p in zip(srcs, preds)]; rec = [x[0] for x in fr if x[0] is not None]
    return dict(SARI=st.mean(M.sari_one(s, p, r) for s, p, r in zip(srcs, preds, refs)),
                rec=100 * st.mean(rec), hal=100 * sum(x[1] for x in fr) / len(fr), neg=100 * sum(x[2] for x in fr) / len(fr))
def J(q, q0): return -1e9 if q["SARI"] < q0["SARI"] - 0.5 else q["rec"] - q["neg"] - 0.1 * q["hal"]

EASY = [(r"경구\s?투여", "복용"), (r"투여", "복용"), (r"소아", "어린이"), (r"이상반응", "부작용"), (r"과민증", "과민 반응(알레르기)"),
        (r"병용", "함께 사용"), (r"연용", "계속 사용"), (r"증량", "양을 늘림"), (r"감량", "양을 줄임"), (r"도포", "바름"),
        (r"점안", "눈에 넣음"), (r"작열감", "화끈거림"), (r"소양감", "가려움"), (r"구역(?!질)", "메스꺼움"), (r"현훈", "어지러움"),
        (r"동통", "통증"), (r"환부", "아픈 부위")]
def easy(t):
    for a, b in EASY: t = re.sub(a, b, t)
    return t

def run_family(names):
    C = {n: json.load(open(f"results/cands/{n}.json", encoding="utf-8")) for n in names if os.path.exists(f"results/cands/{n}.json")}
    N = {n: (json.load(open(f"results/cands/nli_{n}.json")) if os.path.exists(f"results/cands/nli_{n}.json") else None) for n in C}
    if not C: return None
    print("family", list(C), "nli:", [n for n in N if N[n]], flush=True)
    def val_score(w, key):
        qs = [quick(SRC["val"], pick(SRC["val"], C[n][f"val_{key}"], (N[n] or {}).get(f"val_{key}"), w), REF["val"]) for n in C]
        return {k: st.mean(q[k] for q in qs) for k in qs[0]}
    q0 = val_score((0, 0, 99, 0, False), "b4")  # g=99 -> 항상 1등
    grids = {
        "V3": [(a, b, g, 0, False) for a, b, g in itertools.product([0, .25, .5, 1], [.5, 1, 2], [0, .02, .05, .1])],
        "V4": [(a, b, g, 0, True) for a, b, g in itertools.product([0, .25, .5, 1], [.5, 1, 2], [0, .02, .05, .1])]}
    if all(N[n] for n in C):
        grids["V5"] = [(a, b, g, d, True) for a, b, g, d in itertools.product([0, .25, .5], [.5, 1, 2], [0, .02, .05], [.25, .5, 1, 2])]
    chosen = {"V0": ((0, 0, 99, 0, False), "b4"), "V1": ((.5, 1, 0, 0, False), "b4"), "V2": ((.5, 1, 0, 0, False), "b8")}
    for v, gr in grids.items():
        best = max(gr, key=lambda w: J(val_score(w, "b8"), q0)); chosen[v] = (best, "b8"); print(v, "val-best", best, flush=True)
    # 테스트 평가 (시드별 -> 평균±표준편차)
    rows = {}
    for v, (w, key) in chosen.items():
        per = []
        for n in C:
            preds = pick(SRC["test"], C[n][f"test_{key}"], (N[n] or {}).get(f"test_{key}"), w)
            e = M.evaluate_all(SRC["test"], preds, REF["test"])
            if seen is not None:
                idx = [i for i, s in enumerate(seen) if not s]
                u = quick([SRC["test"][i] for i in idx], [preds[i] for i in idx], [REF["test"][i] for i in idx])
                e.update(unseen_SARI=round(u["SARI"], 2), unseen_rec=round(u["rec"], 2), unseen_neg=round(u["neg"], 2))
            per.append((e, preds))
        rows[v] = dict(weights=w, beam=key, metrics={k: (round(st.mean(p[0][k] for p in per), 2), round(st.pstdev(p[0][k] for p in per), 2)) for k in per[0][0]})
        last = (v, per)
    best_v = "V5" if "V5" in rows else "V4"
    per = []
    for n in C:
        w, key = chosen[best_v]
        preds = [easy(p) for p in pick(SRC["test"], C[n][f"test_{key}"], (N[n] or {}).get(f"test_{key}"), w)]
        per.append(M.evaluate_all(SRC["test"], preds, REF["test"]))
    rows["V6"] = dict(weights=chosen[best_v][0], beam="b8", base=best_v, metrics={k: (round(st.mean(p[k] for p in per), 2), round(st.pstdev(p[k] for p in per), 2)) for k in per[0]})
    return rows

out = {}
for fam, model in [("base", "kobart-base-v2"), ("summ", "kobart-summarization")]:
    for tr in ["", "_filt"]:
        r = run_family([f"{model}_s{s}{tr}" for s in range(3)])
        if r: out[fam + tr] = r
json.dump(out, open("results/improve_summary.json", "w"), ensure_ascii=False, indent=1)
K = ["SARI", "chrF", "num_recall", "num_halluc_per100", "neg_lost_pct", "hard_terms_per_text", "unseen_SARI", "unseen_rec", "unseen_neg"]
for fam, rows in out.items():
    print(f"\n## {fam}")
    print("| 변형 | " + " | ".join(K) + " |")
    for v, r in rows.items():
        print(f"| {v} {r['beam']} {r['weights']} | " + " | ".join(f"{r['metrics'][k][0]}±{r['metrics'][k][1]}" if k in r["metrics"] else "-" for k in K) + " |")
