#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
output_file="outcomes/noise_layer.json" 
datasets=("sst2" "mind" "ag_news" "enron") 
lambdas=("15 7 15 16" "12 5 10 15" "11 2 5 16" "11 3 5 16" "9 3 2 13")
freqs=("0.001 0.002" "0.005 0.02" "0.02 0.05" "0.1 0.2" "0.2 0.5")
noise_vars=(0.01 0.01 0.05 0.01)

for j in $(seq 4 4); do
    all_lambda=(${lambdas[$j]})
    freq=${freqs[$j]}

    for i in $(seq 0 3); do
        data="${datasets[$i]}"
        lambda=${all_lambda[$i]}
        noise_var=${noise_vars[$i]}
        echo "Current data_name: $data, lambda: $lambda"

        python train_watermark.py \
            --data_name ${data} \
            --watermark \
            --trigger_min_max_freq ${freq} \
            --loss_ratio ${lambda} \
            --wtm_lr 5e-4 \
            --cls_lr 2e-3 \
            --wtm_epoch 5 \
            --noise_prob 0.5 \
            --noise_var ${noise_var} \
            --pca \
            --seed 2022 \
            --output_file ${output_file}
    done            
done
