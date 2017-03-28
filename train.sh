#!/bin/bash
python -m visdom.server &
python train.py
