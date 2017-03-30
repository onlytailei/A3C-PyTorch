#!/bin/bash
steps=30000
if [ "$#" -ne 0 ]; then
    steps=$1
fi

python ./train.py --load_weight $steps --t_flag 0 --train_dir ./models/lstm/
