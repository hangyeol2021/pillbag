"""train.py 를 복사해 '정답 품질 필터'를 넣은 train_filt.py 를 만든다 (원본은 그대로).
필터: 학습셋에서 정답(e약은요)이 원문 숫자를 절반 미만 남겼거나 원문의 금지 표현을 잃은 쌍을 뺀다."""
s = open("train.py", encoding="utf-8").read()
anchor = 'tr, va, te = load("train"), load("val"), load("test")'
assert anchor in s
s = s.replace(anchor, anchor + '''
_fs = [fact_scores(a, b) for a, b in zip(tr.source_expert, tr.target_easy)]
_keep = [not ((r is not None and r < 0.5) or n) for r, h, n in _fs]
print(f"[filter] train {len(tr)} -> {sum(_keep)} (dropped {len(tr)-sum(_keep)})", flush=True)
tr = tr[_keep].reset_index(drop=True)''')
open("train_filt.py", "w", encoding="utf-8").write(s); print("ok")
