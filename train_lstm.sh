#!/bin/bash
mkdir -p log
mkdir -p models
python -m visdom.server &
python train.py --use_lstm 1
