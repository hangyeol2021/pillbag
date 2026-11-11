"""KoBART 디코더의 cross-attention 시각화 (출력 토큰이 원문의 어느 토큰을 봤는지).
사용: python attn_viz.py ckpt/kobart-base-v2_s0.pt "[용법] 1회 1정, 1일 3회 식후 경구 투여한다."
"""
import sys, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import koreanize_matplotlib
from matplotlib import font_manager
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
ck, text = sys.argv[1], sys.argv[2]
out_png = sys.argv[3] if len(sys.argv) > 3 else "results/attn.png"
base = "gogamza/kobart-summarization" if "summarization" in ck else "gogamza/kobart-base-v2"
tok = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2"); m = AutoModelForSeq2SeqLM.from_pretrained(base, attn_implementation="eager")
m.load_state_dict(torch.load(ck, map_location="cpu")); m.eval()
enc = tok(text, return_tensors="pt"); enc.pop("token_type_ids", None)
g = m.generate(**enc, num_beams=1, max_new_tokens=60)
out = m(**enc, decoder_input_ids=g[:, :-1], output_attentions=True)
att = torch.stack(out.cross_attentions)[-2:].mean(0).mean(1)[0]   # 마지막 2층, 헤드 평균: [tgt, src]
src = [t.replace("▁", " ") for t in tok.convert_ids_to_tokens(enc.input_ids[0])]
tgt = [t.replace("▁", " ") for t in tok.convert_ids_to_tokens(g[0, 1:])]
for f in font_manager.findSystemFonts():
    if any(k in f for k in ("Nanum", "Noto", "CJK", "Malgun", "Unfonts")):
        font_manager.fontManager.addfont(f); plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name(); break
fig, ax = plt.subplots(figsize=(max(6, len(src) * 0.45), max(4, len(tgt) * 0.35)))
ax.imshow(att.detach().numpy(), cmap="Greens", aspect="auto")
ax.set_xticks(range(len(src))); ax.set_xticklabels(src, rotation=70, fontsize=8)
ax.set_yticks(range(len(tgt))); ax.set_yticklabels(tgt, fontsize=8)
ax.set_xlabel("원문 토큰"); ax.set_ylabel("생성 토큰")
plt.tight_layout(); plt.savefig(out_png, dpi=160)
print("output:", tok.decode(g[0], skip_special_tokens=True)); print("saved", out_png)
import json
json.dump({"src": src, "tgt": tgt, "att": [[round(float(v), 3) for v in r] for r in att.detach()]}, open(out_png[:-4] + ".json", "w"), ensure_ascii=False)
