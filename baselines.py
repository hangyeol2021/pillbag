"""기준선 2가지.
1) 원문 그대로(copy): 아무것도 바꾸지 않음 → 지표의 하한/상한 감각.
2) 사전 치환(rule): Project 1에서 제안한 '의학용어 사전 치환'.
   규칙은 일반적인 의약품 안내문 표현 지식으로 미리 작성했고(테스트 데이터를 보고 고치지 않음),
   국립국어원 '쉽고 바른 공공언어' 목록 중 의료·보건 관련 항목을 덧붙였다.
"""
import re, json, sys, pandas as pd
from metrics import evaluate_all

RULES = [
    (r"경구\s*투여한다", "먹습니다"), (r"투여하지\s*않는다", "복용하지 마십시오"), (r"투여하지\s*말\s*것", "복용하지 마십시오"),
    (r"복용하지\s*않는다", "복용하지 마십시오"), (r"투여한다", "복용합니다"), (r"투여", "복용"), (r"경구", "먹는"),
    (r"소아", "어린이"), (r"유소아", "어린이"), (r"영아", "젖먹이"), (r"고령자", "노인"), (r"임부", "임신부"),
    (r"수유부", "젖을 먹이는 여성"), (r"이상반응", "부작용"), (r"과민증", "알레르기"), (r"금기", "사용 금지"),
    (r"병용", "함께 사용"), (r"연용", "계속 사용"), (r"증량", "양을 늘림"), (r"감량", "양을 줄임"),
    (r"1일", "하루"), (r"1회", "한 번에"), (r"도포한다", "바릅니다"), (r"도포", "바름"), (r"환부", "아픈 부위"),
    (r"소양감", "가려움"), (r"작열감", "타는 듯한 느낌"), (r"동통", "통증"), (r"현훈", "어지러움"), (r"구역", "메스꺼움"),
    (r"점안한다", "눈에 넣습니다"), (r"점안", "눈에 넣기"), (r"흡입", "들이마심"), (r"식후", "식사 후"), (r"식전", "식사 전"),
    (r"할\s*것\.", "하십시오."), (r"주의할\s*것", "주의하십시오"), (r"상담할\s*것", "상담하십시오"),
    (r"중지하고", "멈추고"), (r"한다\.", "합니다."), (r"않는다\.", "않습니다."), (r"이다\.", "입니다."),
    (r"바람직하다", "좋습니다"), (r"제네릭", "복제약"), (r"바이오시밀러", "동등 생물 의약품"),
]


def rule_simplify(t):
    t = str(t)
    for a, b in RULES:
        t = re.sub(a, b, t)
    return t


if __name__ == "__main__":
    te = pd.read_csv("data/split_test.csv")
    res = {}
    for name, f in [("copy", lambda x: str(x)), ("rule", rule_simplify)]:
        preds = te.source_expert.map(f).tolist()
        res[name] = evaluate_all(te.source_expert.tolist(), preds, te.target_easy.tolist())
        print(name, res[name], flush=True)
    res["reference(e약은요)"] = evaluate_all(te.source_expert.tolist(), te.target_easy.tolist(), te.target_easy.tolist())
    print("reference", res["reference(e약은요)"])
    import os; os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/baselines.json", "w"), ensure_ascii=False, indent=1)
