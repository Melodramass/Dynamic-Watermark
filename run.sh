#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
python train_watermark.py \
    --data_name sst2 \
    --watermark --cls --steal \
    --trigger_min_max_freq 0.005 0.02 \
    --wtm_lambda 10 \
    --cls_lr 2e-3 \
    --steal_lr 5e-2 \
    --wtm_epoch 5 \
    --noise_prob 0.5 \
    --noise_var 0.01 \
    --batch_size 32 \
    --seed 4277 \
    --output_file "outcomes/sst2.json" 


