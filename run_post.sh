#!/bin/bash
cd ~/med
while [ ! -f logs/maindone.txt ]; do sleep 30; done
python -u leak_check.py > results/leak.txt 2>&1
python -u subset_eval.py > results/subset.txt 2>&1
python -u pick_examples.py > /dev/null 2>&1
python attn_viz.py ckpt/kobart-base-v2_s0.pt "[용법] 성인 1회 1정, 1일 3회 식후 30분에 경구 투여한다. 15세 미만 소아에게는 투여하지 않는다." results/attn_yongbeop.png > logs/attn.log 2>&1
python -u llm_baseline.py > logs/llm.log 2>&1
python -u subset_eval.py > results/subset.txt 2>&1
python -u pillbag_demo.py demo/sample_bag.png ckpt/kobart-base-v2_s0.pt > logs/demo.log 2>&1
echo POSTDONE > logs/postdone.txt
