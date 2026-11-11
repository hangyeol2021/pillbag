s=open("pillbag_demo.py",encoding="utf-8").read()
s=s.replace('cand = list(dict.fromkeys(cand))','''# 숫자로 시작하는 '7정', '1정' 같은 복용량 표기는 약 이름이 아니므로 제외 (한글 2자 이상으로 시작해야 함)
cand = [c for c in dict.fromkeys(cand) if re.match(r"^[가-힣A-Za-z]{2,}", c)]''')
old='''    q = re.sub(r"\\d.*$", "", name)
    j = F.get(URL, {"serviceKey": key, "item_name": q, "type": "json", "numOfRows": 20})
    items, _ = F.items_of(j) if j else ([], 0)
    if not items:
        md.append(f"\\n## {name}\\n허가정보에서 찾지 못했습니다.\\n"); continue
    best = max(items, key=lambda it: difflib.SequenceMatcher(None, name, str(it.get("ITEM_NAME"))).ratio())'''
assert old in s, "old"
new='''    q = re.sub(r"\\d.*$", "", name)
    items = []
    # OCR 오타(예: 타이레놀→타이레늘)에 대비해, 못 찾으면 이름 끝을 한 글자씩 줄여 다시 검색
    for k in range(len(q), 2, -1):
        j = F.get(URL, {"serviceKey": key, "item_name": q[:k], "type": "json", "numOfRows": 50})
        items, _ = F.items_of(j) if j else ([], 0)
        if items: break
    sim = lambda it: difflib.SequenceMatcher(None, name, re.sub(r"\\(.*?\\)", "", str(it.get("ITEM_NAME")))).ratio()
    best = max(items, key=sim) if items else None
    if best is None or sim(best) < 0.6:
        md.append(f"\\n## {name}\\n허가정보에서 찾지 못했습니다.\\n"); continue
    if q not in str(best.get("ITEM_NAME")):
        md.append(f"\\n> OCR 결과 '{name}'을(를) 가장 비슷한 허가 제품명으로 보정했습니다 (유사도 {sim(best):.2f}).\\n")'''
s=s.replace(old,new)
open("pillbag_demo.py","w",encoding="utf-8").write(s); print("ok")
