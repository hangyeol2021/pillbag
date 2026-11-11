s=open("metrics_v1.py",encoding="utf-8").read()
old='''    for a, b in KNUM:
        t = t.replace(a, b)
    out = []'''
new='''    for a, b in KNUM:
        t = t.replace(a, b)
    # 목록 번호(1) 2) (3) 1. ①)는 사실 정보가 아니므로 제외
    t = LIST_MARK.sub(" ", t)
    out = []'''
assert old in s, "pattern"
s=s.replace(old,new)
s=s.replace("def nums(t):", 'LIST_MARK = re.compile(r"(?:^|(?<=[\\s:·*\\-]))[\\(\\[]?\\d{1,2}[\\)\\]]|^\\s*\\d{1,2}\\.(?!\\d)|[\\u2460-\\u2473]", re.M)\n\n\ndef nums(t):',1)
open("metrics.py","w",encoding="utf-8").write(s); print("patched")
