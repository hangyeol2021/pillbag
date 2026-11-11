"""시연용 '가상' 약봉투 이미지 생성 (실제 환자 정보 없음). 약 이름은 실제 허가된 제품명 예시."""
from PIL import Image, ImageDraw, ImageFont
import koreanize_matplotlib, glob, os
fp = [f for f in glob.glob(os.path.dirname(koreanize_matplotlib.__file__) + "/**/*.ttf", recursive=True)][0]
W, H = 900, 620
im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im)
F1, F2 = ImageFont.truetype(fp, 34), ImageFont.truetype(fp, 26)
d.rectangle([10, 10, W - 10, H - 10], outline="black", width=3)
d.text((30, 30), "약 봉 투  (시연용 가상 예시)", font=F1, fill="black")
d.text((30, 90), "환자명: ○○○     조제일: 2026.11.06", font=F2, fill="black")
y = 160
for line in ["1. 타이레놀정500밀리그램   1정 x 3회 x 3일", "2. 아모잘탄정5/50밀리그램   1정 x 1회 x 30일", "3. 무코스타정   1정 x 3회 x 3일"]:
    d.text((30, y), line, font=F2, fill="black"); y += 60
d.text((30, y + 20), "식후 30분에 복용하세요.", font=F2, fill="black")
os.makedirs("demo", exist_ok=True); im.save("demo/sample_bag.png"); print("saved demo/sample_bag.png")
