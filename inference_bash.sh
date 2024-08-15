#!/bin/bash
single_gpus=1
node_num=1
PORT=23456
gpus=`expr $single_gpus \* $node_num`
cpus=8

srun -p ai4earth --kill-on-bad-exit=1 --quotatype=auto --ntasks-per-node=$single_gpus --cpus-per-task=8 -N $node_num --gres=gpu:$single_gpus python -u inference.py \
--init_method 'tcp://127.0.0.1:'$PORT  \
--world_size $gpus    \
--per_cpus $cpus  \
--tensor_model_parallel_size 1  \
--cfgdir "./result/FengWu_TC_physics_addfengwupre/world_size4-FengWu_TC_physics_addfengwupre_4_8" \
--length 1 \
--predict_len 20 \
--batch_size 1 \
--valid_log_name valid_demo.log \
--metric_list "MAE_new"