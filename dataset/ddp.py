import torch
import os
import re
from megatron_utils import mpu

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print



def get_ip(ip_list):
    if "," in ip_list:
        ip_list = ip_list.split(',')[0]
    if "[" in ip_list:
        ipbefore_4, ip4 = ip_list.split('[')
        ip4 = re.findall(r"\d+", ip4)[0]
        ip1, ip2, ip3 = ipbefore_4.split('-')[-4:-1]
    else:
        ip1,ip2,ip3,ip4 = ip_list.split('-')[-4:]
    ip_addr = "tcp://" + ".".join([ip1, ip2, ip3, ip4]) + ":"
    return ip_addr
    

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.local_rank = int(os.environ['SLURM_LOCALID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        ip_addr = get_ip(os.environ['SLURM_STEP_NODELIST'])
        port = int(os.environ['SLURM_SRUN_COMM_PORT'])
        # args.init_method = ip_addr + str(port)
        args.init_method = ip_addr + args.init_method.split(":")[-1]
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True


    # addr = subprocess.getoutput(
    #         "scontrol show hostname {} | head -n1".format(os.environ["SLURM_NODELIST"])
    #     )
    # os.environ["MASTER_PORT"] = args.init_method.split(":")[-1]
    # os.environ["MASTER_ADDR"] = addr
    # os.environ["WORLD_SIZE"] = str(args.world_size)
    # os.environ["RANK"] = str(args.rank)

    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}, local_rank {}): {}'.format(
        args.rank, args.local_rank, args.init_method), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.init_method,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    mpu.initialize_model_parallel(args.tensor_model_parallel_size)
    setup_for_distributed(args.rank == 0)
    print(f'> initialized tensor model parallel with size '
            f'{mpu.get_tensor_model_parallel_world_size()}')
    return args.distributed, args.local_rank, args.rank