"""식약처 공공데이터 수집: e약은요(쉬운 설명) + 의약품 제품 허가정보(전문 설명)를 품목기준코드로 짝짓기.

사용법:
  1) 같은 폴더에 .env 파일을 만들고 한 줄 입력:  DATA_GO_KR_KEY=발급받은_일반인증키(Decoding)
  2) python fetch_mfds.py            (전체 수집)
     python fetch_mfds.py --limit 30  (시험용: 30개만)
결과:
  data/easy_raw.jsonl     e약은요 원본
  data/permit_raw.jsonl   허가정보 상세 원본
  data/pairs_section.csv  품목·항목별 (전문문장, 쉬운문장) 짝
출처: 식품의약품안전처, 공공데이터포털(data.go.kr) 15075057, 15095677
"""
import argparse, json, os, re, time, html
import pandas as pd, requests

BASE = "https://apis.data.go.kr/1471000"
EASY = f"{BASE}/DrbEasyDrugInfoService/getDrbEasyDrugList"
PERMIT_CANDIDATES = [f"{BASE}/DrugPrdtPrmsnInfoService{s}/getDrugPrdtPrmsnDtlInq{o}"
                     for s in ("08", "07", "06") for o in ("08", "07", "06", "05")]
S = requests.Session()


def load_key():
    k = os.environ.get("DATA_GO_KR_KEY")
    if not k and os.path.exists(".env"):
        for line in open(".env", encoding="utf-8"):
            if line.strip().startswith("DATA_GO_KR_KEY="):
                k = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not k:
        raise SystemExit(".env 파일에 DATA_GO_KR_KEY=... 를 넣어 주세요.")
    return k


def get(url, params, tries=5):
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=30)
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError:
                    return {"_raw": r.text[:300]}
            time.sleep(1 + i)
        except requests.RequestException:
            time.sleep(2 + i)
    return None


def items_of(j):
    try:
        it = j["body"]["items"]
    except (KeyError, TypeError):
        try:
            it = j["response"]["body"]["items"]
        except (KeyError, TypeError):
            return [], 0
    if isinstance(it, dict):
        it = it.get("item", [])
    if isinstance(it, dict):
        it = [it]
    body = j.get("body") or j.get("response", {}).get("body", {})
    return it or [], int(body.get("totalCount", 0) or 0)


def doc_to_text(x):
    if not x:
        return ""
    x = html.unescape(str(x))
    paras = re.findall(r"<PARAGRAPH[^>]*>(.*?)</PARAGRAPH>", x, flags=re.S) or [x]
    out = []
    for p in paras:
        p = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", p, flags=re.S)
        p = re.sub(r"<[^>]+>", " ", p)
        p = re.sub(r"\s+", " ", p).strip()
        if p:
            out.append(p)
    return "\n".join(out)


def clean(t):
    return re.sub(r"\s+", " ", html.unescape(str(t or ""))).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    key = load_key()
    os.makedirs("data", exist_ok=True)
    easy, page = [], 1
    while True:
        j = get(EASY, {"serviceKey": key, "pageNo": page, "numOfRows": 100, "type": "json"})
        if j is None or "_raw" in (j or {}):
            raise SystemExit(f"e약은요 호출 실패: {j}")
        it, total = items_of(j)
        easy += it
        print(f"e약은요 {len(easy)}/{total}", flush=True)
        if not it or len(easy) >= total or (a.limit and len(easy) >= a.limit):
            break
        page += 1
        time.sleep(0.1)
    if a.limit:
        easy = easy[:a.limit]
    with open("data/easy_raw.jsonl", "w", encoding="utf-8") as f:
        for e in easy:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    seq0 = easy[0]["itemSeq"]
    permit_url = None
    for u in PERMIT_CANDIDATES:
        j = get(u, {"serviceKey": key, "item_seq": seq0, "type": "json", "numOfRows": 1}, tries=1)
        it, _ = items_of(j) if j else ([], 0)
        if it and any(k in it[0] for k in ("EE_DOC_DATA", "UD_DOC_DATA", "NB_DOC_DATA")):
            permit_url = u
            break
    if not permit_url:
        raise SystemExit("허가정보 상세 조회 주소를 찾지 못했습니다.")
    print("허가정보 주소:", permit_url, flush=True)
    rows = []
    fout = open("data/permit_raw.jsonl", "w", encoding="utf-8")
    sections = [("효능", "EE_DOC_DATA", "efcyQesitm"),
                ("용법", "UD_DOC_DATA", "useMethodQesitm"),
                ("주의사항", "NB_DOC_DATA", "atpnQesitm")]
    for n, e in enumerate(easy, 1):
        j = get(permit_url, {"serviceKey": key, "item_seq": e["itemSeq"], "type": "json", "numOfRows": 1})
        it, _ = items_of(j) if j else ([], 0)
        if it:
            p = it[0]
            fout.write(json.dumps({k: p.get(k) for k in ("ITEM_SEQ", "ITEM_NAME", "ETC_OTC_CODE", "EE_DOC_DATA", "UD_DOC_DATA", "NB_DOC_DATA")}, ensure_ascii=False) + "\n")
            for name, pk, ek in sections:
                src, tgt = doc_to_text(p.get(pk)), clean(e.get(ek))
                if src and tgt:
                    rows.append({"item_seq": e["itemSeq"], "item_name": clean(e.get("itemName")),
                                 "section": name, "source_expert": src, "target_easy": tgt})
        if n % 50 == 0:
            print(f"허가정보 {n}/{len(easy)}, 짝 {len(rows)}", flush=True)
        time.sleep(0.05)
    fout.close()
    df = pd.DataFrame(rows)
    df.to_csv("data/pairs_section.csv", index=False, encoding="utf-8-sig")
    print("완료: 품목", df.item_seq.nunique() if len(df) else 0, "짝", len(df))
    if len(df):
        print(df.groupby("section").size().to_string())
        for _, r in df.head(3).iterrows():
            print("-", r.section, "|", r.source_expert[:120], "=>", r.target_easy[:120])


if __name__ == "__main__":
    main()
