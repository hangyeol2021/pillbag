"""약봉투 사진 -> 약 이름 인식(OCR) -> 식약처 허가정보 조회 -> 쉬운 설명 생성 -> 사실 보존 검사.
사용: python pillbag_demo.py 사진.jpg [ckpt/모델.pt]
- OCR: EasyOCR(무료, 한국어). 개인정보(이름·병원 등)는 저장하지 않고 약 이름만 사용한다.
- 처방약(전문의약품)은 쉬운 설명(e약은요)이 없으므로, 학습한 모델이 바로 이 경우를 돕는다.
결과: results/demo_<사진이름>.md
"""
import sys, os, re, json, difflib, torch
import easyocr, requests
import fetch_mfds as F
from metrics import fact_scores, NEG
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

img = sys.argv[1]
ck = sys.argv[2] if len(sys.argv) > 2 else "ckpt/kobart-summarization_s0.pt"
base = "gogamza/kobart-summarization" if "summarization" in ck else "gogamza/kobart-base-v2"
key = F.load_key()
URL = "https://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService08/getDrugPrdtPrmsnDtlInq08"

reader = easyocr.Reader(["ko", "en"], gpu=True)
lines = [t for _, t, c in reader.readtext(img) if c > 0.2]
# 약 이름 후보: '정/캡슐/시럽/주/연고/액' 등으로 끝나는 덩어리
cand = []
for l in lines:
    for m in re.finditer(r"[가-힣A-Za-z0-9\-]+(?:정|캡슐|캅셀|시럽|현탁액|연고|크림|액|산|과립|패치)(?:\s*\d+(?:\.\d+)?\s*(?:mg|밀리그램|그램|g))?", l):
        cand.append(m.group(0).replace(" ", ""))
# 숫자로 시작하는 '7정', '1정' 같은 복용량 표기는 약 이름이 아니므로 제외 (한글 2자 이상으로 시작해야 함)
cand = [c for c in dict.fromkeys(cand) if re.match(r"^[가-힣A-Za-z]{2,}", c)]
print("OCR 줄 수:", len(lines), "약 이름 후보:", cand, flush=True)

tok = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2"); model = AutoModelForSeq2SeqLM.from_pretrained(base)
model.load_state_dict(torch.load(ck, map_location="cpu")); model.to("cuda").eval()

def simplify(section, text):
    # 문단이 길면 문장 단위로 나눠 512토큰 이내 덩어리로 처리
    sents = re.split(r"(?<=[.다])\s+|\n", text); chunks, cur = [], ""
    for s in sents:
        if len(tok(cur + " " + s).input_ids) > 480 and cur:
            chunks.append(cur); cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur: chunks.append(cur)
    outs = []
    for c in chunks:
        enc = tok(f"[{section}] {c}", return_tensors="pt", truncation=True, max_length=512).to("cuda"); enc.pop("token_type_ids", None)
        with torch.no_grad():
            g = model.generate(**enc, num_beams=4, num_return_sequences=4, max_new_tokens=256, no_repeat_ngram_size=4)
        cs = [t.strip() for t in tok.batch_decode(g, skip_special_tokens=True)]
        def score(x):
            r, h, n = fact_scores(c, x); return (r if r is not None else 1) - 0.5 * h - n
        outs.append(max(cs, key=score))
    return " ".join(outs)

md = [f"# 약봉투 쉬운 설명 (시연)\n\n- 사진: {os.path.basename(img)}\n- 인식한 약 이름 후보: {', '.join(cand) or '없음'}\n- 모델: {base} 미세조정 + 사실 보존 재순위화\n\n> 이 설명은 학습용 시연이며 의사·약사의 복약 지도를 대신하지 않습니다.\n"]
for name in cand[:6]:
    q = re.sub(r"\d.*$", "", name)
    items = []
    # OCR 오타(예: 타이레놀→타이레늘)에 대비해, 못 찾으면 이름 끝을 한 글자씩 줄여 다시 검색
    for k in range(len(q), 2, -1):
        j = F.get(URL, {"serviceKey": key, "item_name": q[:k], "type": "json", "numOfRows": 50})
        items, _ = F.items_of(j) if j else ([], 0)
        if items: break
    sim = lambda it: difflib.SequenceMatcher(None, name, re.sub(r"\(.*?\)", "", str(it.get("ITEM_NAME")))).ratio()
    best = max(items, key=sim) if items else None
    if best is None or (sim(best) < 0.6 and not str(best.get("ITEM_NAME")).startswith(q)):
        md.append(f"\n## {name}\n허가정보에서 찾지 못했습니다.\n"); continue
    if q not in str(best.get("ITEM_NAME")):
        md.append(f"\n> OCR 결과 '{name}'을(를) 가장 비슷한 허가 제품명으로 보정했습니다 (유사도 {sim(best):.2f}).\n")
    md.append(f"\n## {best.get('ITEM_NAME')} ({'전문의약품' if '전문' in str(best.get('ETC_OTC_CODE')) else '일반의약품'})\n")
    for sec, k in [("효능", "EE_DOC_DATA"), ("용법", "UD_DOC_DATA"), ("주의사항", "NB_DOC_DATA")]:
        src = F.doc_to_text(best.get(k))
        if not src: continue
        if sec == "주의사항": src = src[:1500]
        out = simplify(sec, src)
        r, h, n = fact_scores(src, out)
        flag = "확인 필요" if (r is not None and r < 1) or h or n else "숫자·금지 표현 보존됨"
        md.append(f"\n### {sec}\n- **쉬운 설명:** {out}\n- 원문(일부): {src[:300]}{'…' if len(src) > 300 else ''}\n- 사실 검사: {flag} (숫자 보존 {'-' if r is None else round(100*r)}%, 새로 생긴 숫자 {h}개, 금지 표현 누락 {n})\n")
os.makedirs("results", exist_ok=True)
out_path = f"results/demo_{os.path.splitext(os.path.basename(img))[0]}.md"
open(out_path, "w", encoding="utf-8").write("".join(md)); print("saved", out_path)
