#!/bin/bash
single_gpus=2
node_num=1
PORT=12345
gpus=`expr $single_gpus \* $node_num`
cpus=8

srun -p ai4earth --kill-on-bad-exit=1 --quotatype=spot --ntasks-per-node=$single_gpus --cpus-per-task=4 -N $node_num -x SH-IDC1-10-140-24-89 --gres=gpu:$single_gpus  python -u train.py \
--init_method 'tcp://127.0.0.1:'$PORT  \
-c ./configs/VQTC/FengWu_TC_physics.yaml  \
--world_size $gpus \
--per_cpus $cpus  \
--tensor_model_parallel_size 1  \
--outdir './result' \
--desc 'FengWu_TC_physics' \

# VQ_LSTM_TC   _128_mlp32_nofrefu_hidr_many2one
#VQ_VAE_TC     Q_img_token
# --resume_from_config