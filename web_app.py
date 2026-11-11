"""약 설명서 쉬운 말 변환 웹 서비스 (Gradio). 사용: python web_app.py  -> 공개 링크(share) 출력
학습용 시연이며 의사·약사의 복약 지도를 대신하지 않습니다."""
import io, html, gradio as gr
from PIL import Image, ImageDraw
import pillbag_core as P

P.load("ckpt/kobart-summarization_s0_filt.pt")
E = html.escape

def badge(c):
    if c["ok"]:
        return '<span style="background:#1a7f37;color:#fff;padding:2px 8px;border-radius:10px">✔ 숫자·금지 표현 보존</span>'
    r = "-" if c["recall"] is None else f'{c["recall"]}%'
    return ('<span style="background:#bf5700;color:#fff;padding:2px 8px;border-radius:10px">⚠ 확인 필요</span> '
            f'<small>숫자 보존 {r} · 새 숫자 {c["halluc"]} · 금지 누락 {c["neg_lost"]}</small>')

def sec_html(s):
    return (f'<div style="border-left:4px solid #0969da;padding:6px 12px;margin:10px 0">'
            f'<h4 style="margin:2px 0">{s["section"]}</h4>'
            f'<div style="font-size:1.1em;line-height:1.6">{E(s["rerank"])}</div>'
            f'<div style="margin:4px 0">{badge(s["chk"])}</div>'
            f'<details><summary style="cursor:pointer;color:#57606a">원문(허가정보) 보기</summary>'
            f'<div style="color:#57606a;font-size:.9em">{E(s["src"])}</div></details></div>')

def drug_html(item, corr, name, secs):
    kind = "전문의약품" if "전문" in str(item.get("ETC_OTC_CODE")) else "일반의약품"
    note = f'<div style="color:#bf5700">사진의 "{E(name)}"을(를) 가장 비슷한 허가 제품명으로 보정 (유사도 {corr})</div>' if corr else ""
    return (f'<div style="border:1px solid #d0d7de;border-radius:8px;padding:10px 14px;margin:12px 0">'
            f'<h3 style="margin:0">{E(str(item.get("ITEM_NAME")))} <small style="color:#57606a">({kind})</small></h3>{note}'
            + "".join(sec_html(s) for s in secs) + "</div>")

NOTICE = '<p style="color:#57606a">학습용 시연이며 의사·약사의 복약 지도를 대신하지 않습니다. "확인 필요" 표시가 있으면 반드시 원문이나 약사에게 확인하세요.</p>'

def run_img(img, progress=gr.Progress()):
    if img is None:
        return None, "<p>사진을 올려 주세요.</p>"
    buf = io.BytesIO(); img.convert("RGB").save(buf, "PNG")
    pb, size, warn = P.prep_image(buf.getvalue())
    progress(0.1, desc="글자 읽는 중")
    lines = P.ocr(pb, prep=False); cand = P.candidates(lines)
    im = Image.open(io.BytesIO(pb)).convert("RGB"); d = ImageDraw.Draw(im)
    for box, t, c in lines:
        hit = any(x[:2] in t.replace(" ", "") for x in cand)
        d.polygon([tuple(map(int, p)) for p in box], outline=(220, 0, 0) if hit else (170, 170, 170), width=6)
    out = [NOTICE]
    if warn: out.append(f'<p style="color:#bf5700;font-weight:bold">⚠ {E(warn)}</p>')
    out.append(f"<p><b>찾은 약 이름 후보:</b> {E(', '.join(cand)) or '없음'}</p>")
    miss = []
    for k, name in enumerate(cand[:12]):
        progress(0.2 + 0.8 * k / max(1, len(cand)), desc=f"{name} 설명 만드는 중")
        item, corr = P.lookup(name)
        if item is None: miss.append(name); continue
        out.append(drug_html(item, corr, name, P.explain_drug(item)))
    if miss: out.append(f"<small style='color:#57606a'>허가정보에서 못 찾은 후보: {E(', '.join(miss))}</small>")
    return im, "".join(out)

def run_txt(sec, src):
    src = (src or "").strip()
    if not src: return "<p>문장을 입력하세요.</p>"
    o = P.simplify(sec, src)
    return NOTICE + sec_html(dict(section=sec, src=src, rerank=o["rerank"], chk=P.check(src, o["rerank"])))

with gr.Blocks(title="약 설명서를 쉬운 말로") as demo:
    gr.Markdown("# 💊 약 설명서를 쉬운 말로\n약봉투·처방전 사진을 올리면 약 이름을 읽어 식약처 허가정보를 찾고, 환자가 읽기 쉬운 설명으로 바꿔 줍니다. "
                "설명마다 원문의 숫자(용량·횟수·나이)와 금지 표현이 지켜졌는지 자동으로 검사합니다.")
    with gr.Tab("약봉투 사진"):
        with gr.Row():
            with gr.Column(scale=2):
                img = gr.Image(type="pil", label="약봉투 / 처방전 사진 (개인정보는 가리고 올려 주세요)")
                gr.Examples(["demo/pharmacy_bag.png", "demo/sample_bag.png"], inputs=img, label="예시 사진")
                b1 = gr.Button("설명 만들기", variant="primary")
                vis = gr.Image(label="인식 결과 (빨간 상자: 약 이름)")
            with gr.Column(scale=3):
                res = gr.HTML()
        b1.click(run_img, img, [vis, res])
    with gr.Tab("문장 직접 입력"):
        sec = gr.Radio(["용법", "효능", "주의사항"], value="용법", label="항목")
        txt = gr.Textbox(lines=4, label="허가정보 문장", value="성인 1회 1정, 1일 3회 식후 30분에 경구 투여한다. 15세 미만 소아에게는 투여하지 않는다.")
        b2 = gr.Button("쉬운 말로 바꾸기", variant="primary"); res2 = gr.HTML()
        b2.click(run_txt, [sec, txt], res2)
    gr.Markdown("모델: KoBART(정답 품질 필터 재학습) + 사실 보존 재순위화 · 데이터: 식품의약품안전처 의약품 제품 허가정보, e약은요 · 고급딥러닝 Project 2")

demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7860, share=True)
