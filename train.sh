cd NextIP
date=$(date "+%Y%m%d-%H%M%S")

CUDA_VISIBLE_DEVICES=3 python -u train.py > $date.log 2>&1 &