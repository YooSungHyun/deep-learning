#!/usr/bin/env bash

python3 main.py \
    --cuda \
    --epochs 50 \
    --model GRU \
    --emsize 512 \
    --nhid 512 \
    --dropout 0.5 \
    --vocab_path  \
    --data  \
    --lr 0.1 \
    --nlayers 4 \
    --batch_size 128 \
    --bptt 5