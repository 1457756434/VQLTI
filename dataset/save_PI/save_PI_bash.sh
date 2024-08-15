#reserved  --time=5-21:29:36  -x SH-IDC1-10-140-24-69,SH-IDC1-10-140-24-78    -o job2/%j.out --async
srun -p ai4earth --kill-on-bad-exit=1 --quotatype=reserved -o job2/%j.out --async -x SH-IDC1-10-140-24-14,SH-IDC1-10-140-24-11 python -u save_PI.py \
-c ./PI_config/PI_config_mutil_process.yaml  \