"""지금까지 받은 허가정보로 짝 파일을 만든다(수집과 별개로 실행 가능)."""
import json, pandas as pd, time
import fetch_mfds as F
easy = [json.loads(l) for l in open("data/easy_raw.jsonl", encoding="utf-8")]
done = {}
for l in open("data/permit_raw.jsonl", encoding="utf-8"):
    try:
        p = json.loads(l); done[str(p["ITEM_SEQ"])] = p
    except Exception:
        pass
rows = []
for e in easy:
    p = done.get(str(e["itemSeq"]))
    if not p: continue
    for name, pk, ek in [("효능", "EE_DOC_DATA", "efcyQesitm"), ("용법", "UD_DOC_DATA", "useMethodQesitm"), ("주의사항", "NB_DOC_DATA", "atpnQesitm")]:
        src, tgt = F.doc_to_text(p.get(pk)), F.clean(e.get(ek))
        if src and tgt:
            rows.append({"item_seq": e["itemSeq"], "item_name": F.clean(e.get("itemName")), "section": name, "source_expert": src, "target_easy": tgt})
df = pd.DataFrame(rows); df.to_csv("data/pairs_section.csv", index=False, encoding="utf-8-sig")
json.dump({"snapshot_time_utc": time.strftime("%Y-%m-%d %H:%M"), "easy_items": len(easy), "permit_items": len(done), "pairs": len(df)}, open("data/snapshot.json", "w"))
print("snapshot: e약은요", len(easy), "허가정보", len(done), "짝", len(df))
