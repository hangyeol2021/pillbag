#!/bin/bash
# 성능 개선 실험: (4) 정답 품질 필터 재학습  +  (1,2,3,6) 후보 생성·NLI 점수
cd ~/med
python make_train_filt.py || exit 1
B="kobart-base-v2_s0 kobart-base-v2_s1 kobart-base-v2_s2 kobart-summarization_s0 kobart-summarization_s1 kobart-summarization_s2"
F="kobart-base-v2_s0_filt kobart-base-v2_s1_filt kobart-base-v2_s2_filt kobart-summarization_s0_filt kobart-summarization_s1_filt kobart-summarization_s2_filt"
laneA(){ for s in 0 1 2; do python -u train_filt.py --model gogamza/kobart-base-v2 --seed $s --tag _filt; done; }
laneB(){ for s in 0 1 2; do python -u train_filt.py --model gogamza/kobart-summarization --seed $s --tag _filt; done; }
laneG(){ python -u gen_cands.py $B; python -u nli_score.py $B; }
laneA > logs/imp_laneA.log 2>&1 &
laneB > logs/imp_laneB.log 2>&1 &
laneG > logs/imp_laneG.log 2>&1 &
wait
python -u gen_cands.py $F > logs/imp_genF.log 2>&1
python -u nli_score.py $F >> logs/imp_genF.log 2>&1
echo IMPDONE > logs/impdone.txt
