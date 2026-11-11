"""짝 데이터 정리 + 분할.
- 같은 원문(전문 설명)이 여러 품목에 반복되는 경우(같은 성분 제네릭) 학습/평가에 섞이지 않도록
  '정규화한 원문' 단위로 묶어서 분할한다(정보 누수 방지).
- 토큰 길이 제한(원문 512, 정답 256 KoBART 토큰)을 넘는 짝은 제외한다.
출력: data/split_{train,val,test}.csv, data/prep_stats.json
"""
import pandas as pd, numpy as np, re, json, hashlib
from transformers import AutoTokenizer

df = pd.read_csv("data/pairs_section.csv")
n0 = len(df)
norm = lambda s: re.sub(r"[\s\W_]+", "", str(s))
df = df[df.source_expert.str.len() > 5]
df = df[df.target_easy.str.len() > 5]
df["src_key"] = df.source_expert.map(lambda s: hashlib.md5(norm(s).encode()).hexdigest())
# 같은 (원문,정답) 완전 중복 제거
df = df.drop_duplicates(["src_key", "target_easy"])
tok = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")
df["src_tok"] = df.source_expert.map(lambda s: len(tok(s).input_ids))
df["tgt_tok"] = df.target_easy.map(lambda s: len(tok(s).input_ids))
n1 = len(df)
df = df[(df.src_tok <= 512) & (df.tgt_tok <= 256)]
# 원문 단위 그룹 분할 80/10/10
keys = df.src_key.unique()
rng = np.random.default_rng(42)
keys = np.array(keys); rng.shuffle(keys)
a, b = int(len(keys) * .8), int(len(keys) * .9)
grp = {k: ("train" if i < a else "val" if i < b else "test") for i, k in enumerate(keys)}
df["split"] = df.src_key.map(grp)
for s in ["train", "val", "test"]:
    df[df.split == s].to_csv(f"data/split_{s}.csv", index=False, encoding="utf-8-sig")
stats = dict(raw_pairs=n0, after_dedup=n1, after_len_filter=len(df), items=int(df.item_seq.nunique()),
             unique_sources=int(len(keys)),
             by_section=df.groupby("section").size().to_dict(),
             by_split=df.groupby("split").size().to_dict(),
             src_tok_median=float(df.src_tok.median()), tgt_tok_median=float(df.tgt_tok.median()),
             src_chars_mean=float(df.source_expert.str.len().mean()), tgt_chars_mean=float(df.target_easy.str.len().mean()))
json.dump(stats, open("data/prep_stats.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(stats, ensure_ascii=False, indent=1))
