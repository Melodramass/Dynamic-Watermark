# CUDA_VISIBLE_DEVICES=5
python embmaker.py \
 --data_name enron \
 --cls \
 --trigger_min_max_freq 0.005 0.02 \
 --wtm_lr 5e-4 \
 --cls_lr 2e-3 \
  --seed 2022
