"""허가정보를 동시에 여러 개 받아 이어서 수집하고, 마지막에 짝 파일을 만든다."""
import json, os, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import fetch_mfds as F

key = F.load_key()
URL = "https://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService08/getDrugPrdtPrmsnDtlInq08"
easy = [json.loads(l) for l in open("data/easy_raw.jsonl", encoding="utf-8")]
done = {}
if os.path.exists("data/permit_raw.jsonl"):
    for l in open("data/permit_raw.jsonl", encoding="utf-8"):
        try:
            p = json.loads(l); done[str(p["ITEM_SEQ"])] = p
        except Exception:
            pass
todo = [e for e in easy if str(e["itemSeq"]) not in done]
print("already", len(done), "todo", len(todo), flush=True)
lock = threading.Lock()
fout = open("data/permit_raw.jsonl", "a", encoding="utf-8")

def job(e):
    j = F.get(URL, {"serviceKey": key, "item_seq": e["itemSeq"], "type": "json", "numOfRows": 1})
    it, _ = F.items_of(j) if j else ([], 0)
    if not it:
        return None
    p = it[0]
    rec = {k: p.get(k) for k in ("ITEM_SEQ", "ITEM_NAME", "ETC_OTC_CODE", "EE_DOC_DATA", "UD_DOC_DATA", "NB_DOC_DATA")}
    with lock:
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
        done[str(rec["ITEM_SEQ"])] = rec
    return 1

n = 0
with ThreadPoolExecutor(max_workers=8) as ex:
    for f in as_completed([ex.submit(job, e) for e in todo]):
        n += 1
        if n % 200 == 0:
            print("done", n, "/", len(todo), flush=True)
fout.close()
rows = []
sections = [("효능", "EE_DOC_DATA", "efcyQesitm"), ("용법", "UD_DOC_DATA", "useMethodQesitm"), ("주의사항", "NB_DOC_DATA", "atpnQesitm")]
for e in easy:
    p = done.get(str(e["itemSeq"]))
    if not p:
        continue
    for name, pk, ek in sections:
        src, tgt = F.doc_to_text(p.get(pk)), F.clean(e.get(ek))
        if src and tgt:
            rows.append({"item_seq": e["itemSeq"], "item_name": F.clean(e.get("itemName")), "section": name,
                         "source_expert": src, "target_easy": tgt})
df = pd.DataFrame(rows)
df.to_csv("data/pairs_section.csv", index=False, encoding="utf-8-sig")
print("완료: 품목", df.item_seq.nunique(), "짝", len(df), flush=True)
print(df.groupby("section").size().to_string())
