#!/bin/bash

# 
horizon_list=(1 2 4)
hid_list=(64 128)
learning_rate_list=(0.001 0.0005)
lambda_list=(1)

# 
DATA="us_hhs"
SIM_MAT="us_hhs-adj"
LOG="hhs"
MODEL="mamba_epi"

# 
epochs=400
lam=0.1
hid=128
lr=0.001

# 
for i in "${!horizon_list[@]}"
do
    gpu_id=${gpu_list[$i % ${#gpu_list[@]}]}  # 
    horizon=${horizon_list[$i]}
    model_option="--model ${MODEL} --epochs ${epochs} --epilambda ${lam} --gpu ${gpu_id}"
    rnn_option="--n_hidden ${hid}"
    option="--lr ${lr} --dataset ${DATA} --sim_mat ${SIM_MAT} --horizon ${horizon} "
    cmd="python -u ./train.py ${option} ${model_option} ${rnn_option} | tee log/mamba_epi/mamba_epi.${LOG}.hid-${hid}.h-${horizon}.lr-${lr}.lam-${lam}.out"
    
    echo $cmd
    eval $cmd &
done

# 
wait
