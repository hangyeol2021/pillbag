import pandas as pd, json
from metrics import fact_scores
p=pd.read_csv("results/pred_kobart-base-v2_s0.csv").fillna("")
flag=pd.read_csv("results/test_leakflag.csv").tgt_in_train.values
p["unseen"]=~flag
rows=[]
for i,r in p.iterrows():
    a=fact_scores(r.source,r.plain); b=fact_scores(r.source,r.fact_rerank); c=fact_scores(r.source,r.reference)
    rows.append(dict(i=i,sec=r.section,unseen=bool(r.unseen),src=r.source,ref=r.reference,plain=r.plain,rr=r.fact_rerank,fp=a,fr=b,fref=c))
ex={}
ex["rerank_fix"]=[x for x in rows if x["plain"]!=x["rr"] and x["unseen"] and (x["fr"][0] or 0)>(x["fp"][0] or 0) and len(x["src"])<260][:4]
ex["ref_drops"]=[x for x in rows if x["unseen"] and x["fref"][0] is not None and x["fref"][0]<0.6 and len(x["src"])<260][:3]
ex["good"]=[x for x in rows if x["unseen"] and x["fp"][0]==1 and x["fp"][1]==0 and x["sec"]=="주의사항" and len(x["src"])<300][:2]
ex["neg_lost"]=[x for x in rows if x["unseen"] and x["fp"][2] and len(x["src"])<300][:2]
json.dump(ex,open("results/examples.json","w"),ensure_ascii=False,indent=1)
print({k:len(v) for k,v in ex.items()})
