#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
datasets=("sst2") # "mind" "ag_news" "enron")

for data_name in "${datasets[@]}"; do
    python redalarm.py \
        --data_name "$data_name" \
        --watermark --steal \
        --trigger_min_max_freq 0.005 0.02 \
        --wtm_lr 5e-4 \
        --cls_lr 2e-3 \
        --wtm_epoch 5 \
        --seed 2022
done