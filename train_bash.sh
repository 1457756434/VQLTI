#!/bin/bash
single_gpus=2
node_num=1
PORT=12345
gpus=`expr $single_gpus \* $node_num`
cpus=8

srun -p ai4earth --kill-on-bad-exit=1 --quotatype=spot --ntasks-per-node=$single_gpus --cpus-per-task=4 -N $node_num --gres=gpu:$single_gpus  python -u train.py \
--init_method 'tcp://127.0.0.1:'$PORT  \
-c ./configs/VQTC/FengWu_TC_physics.yaml  \
--world_size $gpus \
--per_cpus $cpus  \
--tensor_model_parallel_size 1  \
--outdir './result' \
--desc 'FengWu_TC_physics' \
