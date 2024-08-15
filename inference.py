import argparse
import os
import torch
from utils.builder import ConfigBuilder
import utils.misc as utils
import yaml
from utils.logger import get_logger
import copy
from utils.metrics import MetricsRecorder

import pandas as pd
import numpy as np
#----------------------------------------------------------------------------

def subprocess_fn(args):
    utils.setup_seed(args.seed * args.world_size + args.rank)

    logger = get_logger("valid", args.run_dir, utils.get_rank(), filename=args.valid_log_name)

    # args.logger = logger
    args.cfg_params["logger"] = logger

    # build config
    logger.info('Building config ...')
    builder = ConfigBuilder(**args.cfg_params)

    # build model
    logger.info('Building models ...')
    model = builder.get_model()

    if model.use_ceph:
        model_checkpoint = os.path.join(args.relative_checkpoint_dir, 'checkpoint_best.pth')
        
    else:
        model_checkpoint = os.path.join(args.run_dir, 'checkpoint_best.pth')

    logger.info(f"{model_checkpoint}")
    model.load_checkpoint(model_checkpoint)
    
    model_without_ddp = utils.DistributedParallel_Model(model, args.local_rank)
    
    for key in model_without_ddp.model:
        params = [p for p in model_without_ddp.model[key].parameters() if p.requires_grad]
        cnt_params = sum([p.numel() for p in params])
        logger.info("params {key}: {params}".format(key=key, params=cnt_params))
        # print("params {key}:".format(key=key), cnt_params)


    # build dataset
    logger.info('Building dataloaders ...')
    args.cfg_params['dataset']['valid']['cfgdir']= args.cfgdir
    args.cfg_params['dataset']['valid']['valid_log_name']= args.valid_log_name
    dataset_params = args.cfg_params['dataset']

    #test_dataloader = builder.get_dataloader(dataset_params=dataset_params, split = 'test', batch_size=args.batch_size)
    valid_dataloader = builder.get_dataloader(dataset_params=dataset_params, split = 'valid', batch_size=args.batch_size)

    logger.info('valid dataloaders build complete')
    logger.info('begin valid ...')
    # model_without_ddp.test(test_data_loader=test_dataloader, epoch=0)
    model_without_ddp.eval_metrics = MetricsRecorder(args.metric_list)
    
    print(args.cfg_params["dataset"]["valid"]["type"])


    
    model_without_ddp.test_final_inter(valid_dataloader, args.predict_len, args.cfgdir, args.valid_log_name)
    print("Model is")
    my_model = args.cfg_params["model"]["params"].get('sub_model', {})
    my_model = list(my_model.keys())[0]
    print(my_model)


    
def main(args):
    if args.world_size > 1:
        utils.init_distributed_mode(args)
    else:
        args.rank = 0
        args.local_rank = 0
        args.distributed = False
        args.gpu = 0
        torch.cuda.set_device(args.gpu)


    run_dir = args.cfgdir
    print(run_dir)
    args.cfg = os.path.join(args.cfgdir, 'training_options.yaml')
    
    with open(args.cfg, 'r') as cfg_file:
        cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
    
    cfg_params['dataloader']['num_workers'] = args.per_cpus
    cfg_params['dataset']['valid'] = copy.deepcopy(cfg_params['dataset']['test'])
    cfg_params['dataset']['valid']['length'] = args.length
    cfg_params['dataset']['valid']['pred_length'] = args.predict_len
    if "checkpoint_path" in cfg_params["model"]["params"]["extra_params"]:
        del cfg_params["model"]["params"]["extra_params"]["checkpoint_path"]

   
    dataset_vnames = cfg_params['dataset']['train'].get("vnames", None)
    if dataset_vnames is not None:
        constants_len = len(dataset_vnames.get('constants'))
    else:
        constants_len = 0
    cfg_params['model']['params']['constants_len'] = constants_len
    # cfg_params['model']['params'].pop('optimizer')
    # cfg_params['model']['params'].pop('lr_scheduler')

    if args.rank == 0:
        with open(os.path.join(run_dir, 'valid_options.yaml'), 'wt') as f:
            yaml.dump(vars(args), f, indent=2, sort_keys=False)
            yaml.dump(cfg_params, f, indent=2, sort_keys=False)

    args.cfg_params = cfg_params
    args.run_dir = run_dir
    if "relative_checkpoint_dir" in cfg_params:
        
        args.relative_checkpoint_dir = cfg_params['relative_checkpoint_dir']
        print(args.relative_checkpoint_dir)

    print('Launching processes...')
    subprocess_fn(args)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensor_model_parallel_size', type = int,     default = 1,                                            help = 'tensor_model_parallel_size')
    parser.add_argument('--seed',           type = int,     default = 0,                                            help = 'seed')
    parser.add_argument('--cuda',           type = int,     default = 0,                                            help = 'cuda id')
    parser.add_argument('--world_size',     type = int,     default = 1,                                            help = 'Number of progress')
    parser.add_argument('--per_cpus',       type = int,     default = 1,                                            help = 'Number of perCPUs to use')
    parser.add_argument('--batch_size',     type = int,     default = 32,                                           help = "batch size")
    parser.add_argument('--predict_len',    type = int,     default = 15,                                           help = "predict len")
    parser.add_argument('--length',         type = int,     default = 16,                                           help = "predict len")
    parser.add_argument('--metric_list',    nargs = '+',    default = ["MAE"],                                    help = 'metric list')
    # parser.add_argument('--world_size',     type = int,     default = -1,                                           help = 'number of distributed processes')
    parser.add_argument('--init_method',    type = str,     default = 'tcp://127.0.0.1:23456',                      help = 'multi process init method')
    parser.add_argument('--cfgdir',         type = str,     default = 'world_size4-FengWu_TC_physics_4_8',  help = 'Where to save the results')
    parser.add_argument('--valid_log_name', type = str,     default = "valid.log",                                  help = 'valid log name')

    args = parser.parse_args()

    main(args)


