#!/bin/bash
mkdir -p log
mkdir -p models
ps -ef | grep 'python -m visdom.server' | grep -v grep | awk '{print $2}' | xargs kill
python -m visdom.server &
OMP_NUM_THREADS=1
python train.py --use_lstm 1 --train_dir ./models/lstm/
