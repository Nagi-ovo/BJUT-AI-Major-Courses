#!/bin/bash
# Please type in the following command to run the model:

# 1. chmod +x run_all.sh
# 2. ./run_all.sh METR_LA（后面跟的参数是数据集）

# 获取第一个命令行参数作为数据集名称
DATASET=$1

models=("LR" "HA" "GWN" "GRU" "STGCN")

python run_all_models.py --enable-cuda --dataset $DATASET
python run_all_models.py --enable-cuda --dataset $DATASET --mode test

for model in "${models[@]}"; do
    python show_all_results.py --dataset $DATASET --model $model
done