#!/bin/bash
mkdir -p log
mkdir -p models
ps -ef | grep 'python -m visdom.server' | grep -v grep | awk '{print $2}' | xargs kill
python -m visdom.server &
python ./train.py
