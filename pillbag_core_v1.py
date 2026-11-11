"""발표용 약봉투 시연 핵심 함수 (pillbag_demo.py 로직을 함수로 분리 + 재순위화 전/후 비교 + API 캐시).
노트북 pillbag_app.ipynb 에서 import 해서 쓴다."""
import os, re, json, difflib, io
import torch
import fetch_mfds as F
from metrics import fact_scores
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

URL = "https://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService08/getDrugPrdtPrmsnDtlInq08"
CACHE_PATH = "results/demo_api_cache.json"
S = {}  # reader, tok, model, base, key, cache


def load(ck="ckpt/kobart-summarization_s0.pt", ocr=True):
    import easyocr
    base = "gogamza/kobart-summarization" if "summarization" in ck else "gogamza/kobart-base-v2"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")  # 요약판 토크나이저 오류 때문에 base로 고정
    model = AutoModelForSeq2SeqLM.from_pretrained(base)
    model.load_state_dict(torch.load(ck, map_location="cpu")); model.to(dev).eval()
    S.update(tok=tok, model=model, base=base, dev=dev, ck=ck)
    if ocr:
        S["reader"] = easyocr.Reader(["ko", "en"], gpu=(dev == "cuda"))
    try:
        S["key"] = F.load_key()
    except Exception as e:
        S["key"] = None; print("API 키 없음(캐시만 사용):", e)
    try:
        S["cache"] = json.load(open(CACHE_PATH, encoding="utf-8"))
    except Exception:
        S["cache"] = {}
    print(f"준비 완료: {base} ({dev}), 캐시 {len(S['cache'])}건")


def prep_image(img_bytes, target=1800):
    """작은/흐린 사진 보정: 회전 정보 반영 -> 긴 변 target px로 확대 -> 흑백 + 대비 + 선명하게.
    반환: (보정된 PNG bytes, 원본 크기, 경고 문구 or None)"""
    from PIL import Image, ImageOps, ImageEnhance, ImageFilter
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes))).convert("RGB")
    size = im.size; warn = None
    if max(size) < 1000:
        warn = f"사진 해상도가 낮습니다({size[0]}x{size[1]}). 글자가 작으면 OCR이 거의 못 읽습니다. 원본 사진(긴 변 1500px 이상)을 올려 주세요."
    s = target / max(size)
    if s > 1: im = im.resize((int(size[0] * s), int(size[1] * s)), Image.LANCZOS)
    g = ImageOps.autocontrast(ImageOps.grayscale(im), cutoff=1)
    g = ImageEnhance.Sharpness(g).enhance(2.0).filter(ImageFilter.MedianFilter(3))
    buf = io.BytesIO(); g.save(buf, "PNG")
    return buf.getvalue(), size, warn


def ocr(img_bytes, prep=True, min_conf=0.2):
    """반환: [(box, text, conf)]  (prep=True면 보정한 이미지 좌표)"""
    if prep: img_bytes = prep_image(img_bytes)[0]
    return [(b, t, c) for b, t, c in S["reader"].readtext(img_bytes, mag_ratio=1.5, contrast_ths=0.05, adjust_contrast=0.7) if c > min_conf]


def candidates(lines):
    cand = []
    for _, l, _ in lines:
        for m in re.finditer(r"[가-힣A-Za-z0-9\-]+(?:정|캡슐|캅셀|시럽|현탁액|연고|크림|액|산|과립|패치)(?:\s*\d+(?:\.\d+)?\s*(?:mg|밀리그램|그램|g))?", l):
            cand.append(m.group(0).replace(" ", ""))
    # '7정', '1정' 같은 복용량 표기 제외 + 약 이름이 아닌 흔한 단어(영수증·안내문) 제외
    STOP = ("계산", "금액", "전액", "생산", "일정", "관계없이", "결정", "고정", "조정", "측정", "예정", "지정", "인정", "안정", "규정", "확정", "청산", "합산", "정산", "환액", "총액", "잔액")
    out = []
    for c in dict.fromkeys(cand):
        if not re.match(r"^[가-힣A-Za-z]{2,}", c): continue
        if any(w in c for w in STOP): continue
        if len(re.sub(r"\d.*$", "", c)) < 3: continue  # '산', '액' 등 너무 짧은 것
        out.append(c)
    return out


def _api(q):
    if q in S["cache"]:
        return S["cache"][q]
    if not S.get("key"):
        return []
    j = F.get(URL, {"serviceKey": S["key"], "item_name": q, "type": "json", "numOfRows": 50})
    items = F.items_of(j)[0] if j else []
    keep = ["ITEM_NAME", "ETC_OTC_CODE", "EE_DOC_DATA", "UD_DOC_DATA", "NB_DOC_DATA"]
    items = [{k: it.get(k) for k in keep} for it in items]
    S["cache"][q] = items
    os.makedirs("results", exist_ok=True)
    json.dump(S["cache"], open(CACHE_PATH, "w", encoding="utf-8"), ensure_ascii=False)
    return items


def lookup(name):
    """OCR 약 이름 -> (허가 품목, 보정 유사도 or None). 못 찾으면 (None, None)"""
    q = re.sub(r"\d.*$", "", name)
    # OCR 오타 대비 검색어 후보: ① 그대로 ② 앞 글자 오인식 보정(수↔슈, 두↔듀 등) ③ 끝을 줄이기 ④ 앞을 줄이기
    SWAP = {"수": "슈", "슈": "수", "두": "듀", "듀": "두", "주": "쥬", "추": "츄", "부": "뷰", "무": "뮤", "르": "를", "로": "료", "토": "툐", "디": "다", "짇": "질", "질": "짇"}
    qs = [q] + [q[:i] + SWAP[ch] + q[i + 1:] for i, ch in enumerate(q[:2]) if ch in SWAP]
    qs += [q[:k] for k in range(len(q) - 1, 2, -1)] + [q[k:] for k in range(1, max(1, len(q) - 3))]
    items = []
    for qq in dict.fromkeys(qs):
        items = _api(qq)
        if items: break
    # 용량·괄호(성분명)를 떼고 이름끼리만 비교 (예: '슈가메트서방정5/1000밀리그램' -> '슈가메트서방정')
    core = lambda s: re.sub(r"\d.*$", "", re.sub(r"\(.*?\)", "", str(s)))
    sim = lambda it: difflib.SequenceMatcher(None, q, core(it.get("ITEM_NAME"))).ratio()
    best = max(items, key=sim) if items else None
    if best is None or (sim(best) < 0.6 and not str(best.get("ITEM_NAME")).startswith(q)):
        return None, None
    return best, (None if q in str(best.get("ITEM_NAME")) else round(sim(best), 2))


def _score(src, x):
    r, h, n = fact_scores(src, x)
    return (r if r is not None else 1) - 0.5 * h - n


def simplify(section, text):
    """반환: dict(top1=빔 1등 결과, rerank=재순위화 결과)"""
    tok, model = S["tok"], S["model"]
    sents = re.split(r"(?<=[.다])\s+|\n", text); chunks, cur = [], ""
    for s in sents:
        if len(tok(cur + " " + s).input_ids) > 480 and cur:
            chunks.append(cur); cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur: chunks.append(cur)
    top1, rr = [], []
    for c in chunks:
        enc = tok(f"[{section}] {c}", return_tensors="pt", truncation=True, max_length=512).to(S["dev"])
        enc.pop("token_type_ids", None)
        with torch.no_grad():
            g = model.generate(**enc, num_beams=4, num_return_sequences=4, max_new_tokens=256, no_repeat_ngram_size=4)
        cs = [t.strip() for t in tok.batch_decode(g, skip_special_tokens=True)]
        top1.append(cs[0]); rr.append(max(cs, key=lambda x: _score(c, x)))
    return dict(top1=" ".join(top1), rerank=" ".join(rr))


def check(src, out):
    r, h, n = fact_scores(src, out)
    ok = not ((r is not None and r < 1) or h or n)
    return dict(ok=ok, recall=None if r is None else round(100 * r), halluc=h, neg_lost=n)


def explain_drug(item, max_nb=1500):
    secs = []
    for sec, k in [("효능", "EE_DOC_DATA"), ("용법", "UD_DOC_DATA"), ("주의사항", "NB_DOC_DATA")]:
        src = F.doc_to_text(item.get(k))
        if not src: continue
        if sec == "주의사항": src = src[:max_nb]
        o = simplify(sec, src)
        secs.append(dict(section=sec, src=src, top1=o["top1"], rerank=o["rerank"],
                         chk_top1=check(src, o["top1"]), chk=check(src, o["rerank"])))
    return secs
