"""평가 지표.
- SARI (Xu et al., 2016): 원문 대비 무엇을 지키고/지우고/더했는지를 정답과 비교. 어절 단위.
- chrF (sacrebleu): 음절(문자) n-gram 기반 정답 유사도. 한국어는 어절 BLEU가 희소하므로 함께 보고.
- 사실 보존: 원문의 숫자+단위(용량·횟수·나이·기간), 부정·금지 표현이 출력에 남아 있는지.
- 전문용어 잔존: 미리 정한 어려운 의약 표현이 출력에 남은 개수.
"""
import re, collections
import sacrebleu

NUM_UNIT = re.compile(r"(\d+(?:[.,]\d+)?)\s*(mg|㎎|g|mL|ml|㎖|L|μg|mcg|정|캡슐|포|병|회|번|시간|일|주|개월|세|살|%|매|방울|스푼|cc|kg|㎏)?")
NEG = re.compile(r"(않|말 것|말것|마십시오|마세요|금지|금한다|안 된다|안된다|하지 말|피하|삼가)")
HARD = ["투여", "경구", "소아", "이상반응", "금기", "과민증", "복용시키", "투약", "병용", "주의할 것", "연용", "증량", "감량",
        "도포", "흡입", "점안", "점이", "점비", "질환", "작열감", "발진", "소양감", "구역", "현훈", "동통", "울혈", "환부",
        "1일", "1회", "하는 것이 바람직하다", "할 것", "않는다", "한다."]


KNUM = [("하루", "1일"), ("이틀", "2일"), ("사흘", "3일"), ("한 번", "1회"), ("한번", "1회"), ("두 번", "2회"), ("두번", "2회"),
        ("세 번", "3회"), ("세번", "3회"), ("네 번", "4회"), ("네번", "4회"), ("다섯 번", "5회"), ("여섯 번", "6회"),
        ("한 알", "1정"), ("두 알", "2정"), ("세 알", "3정"), ("한 포", "1포"), ("두 포", "2포"), ("한 병", "1병"), ("한 캡슐", "1캡슐")]


def nums(t):
    t = str(t)
    for a, b in KNUM:
        t = t.replace(a, b)
    out = []
    for m in NUM_UNIT.finditer(t):
        v = m.group(1).replace(",", "")
        try:
            v = str(float(v)).rstrip("0").rstrip(".")
        except ValueError:
            pass
        out.append(v)
    return collections.Counter(out)


def fact_scores(src, out):
    s, o = nums(src), nums(out)
    kept = sum(min(c, o[k]) for k, c in s.items())
    recall = kept / sum(s.values()) if s else None                  # 원문 숫자를 얼마나 남겼나
    halluc = sum(c for k, c in o.items() if k not in s)              # 원문에 없는 숫자를 만들어 냈나
    neg_src, neg_out = bool(NEG.search(str(src))), bool(NEG.search(str(out)))
    neg_lost = int(neg_src and not neg_out)                           # 원문의 금지/부정이 사라졌나
    return recall, halluc, neg_lost


def hard_count(t):
    t = str(t)
    return sum(t.count(h) for h in HARD)


# ---- SARI (Xu et al. 2016) 단일 정답, 어절 n-gram 1~4 ----
def _ngrams(toks, n):
    return collections.Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def sari_one(src, pred, ref):
    s, p, r = str(src).split(), str(pred).split(), str(ref).split()
    keep_s, del_s, add_s = [], [], []
    for n in range(1, 5):
        sg, pg, rg = _ngrams(s, n), _ngrams(p, n), _ngrams(r, n)
        # add
        add_p = set(pg) - set(sg); add_r = set(rg) - set(sg)
        tp = len(add_p & add_r)
        pa = tp / len(add_p) if add_p else 0.0; ra = tp / len(add_r) if add_r else 0.0
        add_s.append(2 * pa * ra / (pa + ra) if pa + ra > 0 else 0.0)
        # keep
        keep_p = set(pg) & set(sg); keep_r = set(rg) & set(sg)
        tpk = len(keep_p & keep_r)
        pk = tpk / len(keep_p) if keep_p else 0.0; rk = tpk / len(keep_r) if keep_r else 0.0
        keep_s.append(2 * pk * rk / (pk + rk) if pk + rk > 0 else 0.0)
        # delete (precision only, as in the original)
        del_p = set(sg) - set(pg); del_r = set(sg) - set(rg)
        tpd = len(del_p & del_r)
        del_s.append(tpd / len(del_p) if del_p else 0.0)
    avg = lambda x: sum(x) / len(x)
    return 100 * (avg(keep_s) + avg(del_s) + avg(add_s)) / 3


def evaluate_all(srcs, preds, refs):
    sari = sum(sari_one(s, p, r) for s, p, r in zip(srcs, preds, refs)) / len(preds)
    chrf = sacrebleu.corpus_chrf(list(map(str, preds)), [list(map(str, refs))]).score
    bleu = sacrebleu.corpus_bleu(list(map(str, preds)), [list(map(str, refs))], tokenize="char").score
    fr = [fact_scores(s, p) for s, p in zip(srcs, preds)]
    rec = [x[0] for x in fr if x[0] is not None]
    return dict(SARI=round(sari, 2), chrF=round(chrf, 2), BLEU_char=round(bleu, 2),
                num_recall=round(100 * sum(rec) / max(1, len(rec)), 2),
                num_halluc_per100=round(100 * sum(x[1] for x in fr) / len(fr), 2),
                neg_lost_pct=round(100 * sum(x[2] for x in fr) / len(fr), 2),
                hard_terms_per_text=round(sum(map(hard_count, preds)) / len(preds), 3),
                len_ratio=round(sum(len(str(p)) for p in preds) / max(1, sum(len(str(s)) for s in srcs)), 3))
