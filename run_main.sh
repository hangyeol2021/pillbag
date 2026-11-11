#!/bin/bash
cd ~/med
python -u build_pairs.py && python -u prep.py && python -u baselines.py || exit 1
laneA(){ for s in 0 1 2; do python -u train.py --model gogamza/kobart-base-v2 --seed $s; done; }
laneB(){ for s in 0 1 2; do python -u train.py --model gogamza/kobart-summarization --seed $s; done; }
laneA > logs/laneA.log 2>&1 &
laneB > logs/laneB.log 2>&1 &
wait
python -u aggregate.py > results/summary.txt 2>&1
echo MAINDONE > logs/maindone.txt
