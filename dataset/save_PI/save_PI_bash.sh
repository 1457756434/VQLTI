srun -p ai4earth --kill-on-bad-exit=1 --quotatype=reserved -o job2/%j.out --async python -u save_PI.py \
-c ./PI_config/PI_config_mutil_process.yaml  \