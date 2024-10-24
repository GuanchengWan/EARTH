#!/bin/bash

# 参数数组
horizon_list=(1 2 4)
hid_list=(64 128)
learning_rate_list=(0.001 0.0005)
lambda_list=(1)

# 获取数据输入
DATA="us_hhs"
SIM_MAT="us_hhs-adj"
LOG="hhs"
MODEL="mamba_epi"

# GPU 数组
gpu_list=(3 6 7)  # 假设有三个可用的 GPU

# 固定的参数
epochs=400
lam=0.1
hid=128
lr=0.001

# 并行运行不同的horizon
for i in "${!horizon_list[@]}"
do
    gpu_id=${gpu_list[$i % ${#gpu_list[@]}]}  # 循环使用 GPU
    horizon=${horizon_list[$i]}
    model_option="--model ${MODEL} --epochs ${epochs} --epilambda ${lam} --gpu ${gpu_id}"
    rnn_option="--n_hidden ${hid}"
    option="--lr ${lr} --dataset ${DATA} --sim_mat ${SIM_MAT} --horizon ${horizon} "
    cmd="python -u ./train.py ${option} ${model_option} ${rnn_option} | tee log/mamba_epi/mamba_epi.${LOG}.hid-${hid}.h-${horizon}.lr-${lr}.lam-${lam}.out"
    
    echo $cmd
    eval $cmd &
done

# 等待所有后台任务完成
wait
