#!/bin/bash
cd ~/med
for s in 0 1 2; do python -u train.py --model gogamza/kobart-summarization --seed $s; done > logs/laneB.log 2>&1
echo DONE > logs/laneBdone.txt
