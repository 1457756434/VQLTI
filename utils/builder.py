"""
Configuration builder.

Authors: Hongjie Fang.
"""
import os
# from torch.utils.data.distributed import DistributedSampler
from utils.distributedsample import DistributedSampler, WeightedRandomSampler
from utils.misc import get_rank, get_world_size, dictToObj, collate_fn, is_dist_avail_and_initialized
from timm.scheduler import create_scheduler
import numpy as np
from megatron_utils import mpu
from pytorch_lamb import Lamb


import torch
#from torch.utils.data import WeightedRandomSampler
#from torch.utils.data.distributed import DistributedSampler
# logging.setLoggerClass(ColoredLogger)
# logger = logging.getLogger(__name__)


class ConfigBuilder(object):
    """
    Configuration Builder.

    Features includes:
        
        - build model from configuration;
        
        - build optimizer from configuration;
        
        - build learning rate scheduler from configuration;
        
        - build dataset & dataloader from configuration;
        
        - build statistics directory from configuration;
        
        - build criterion from configuration;

        - build metrics from configuration;
        
        - fetch training parameters (e.g., max_epoch, multigpu) from configuration.

        - fetch inferencer parameters (e.g., inference image size, inference checkpoint path, inference min depth and max depth, etc.)
    """
    def __init__(self, **params):
        """
        Set the default configuration for the configuration builder.

        Parameters
        ----------
        
        params: the configuration parameters.
        """
        super(ConfigBuilder, self).__init__()
        self.model_params = params.get('model', {})
        # self.optimizer_params = params.get('optimizer', {})
        # self.lr_scheduler_params = params.get('lr_scheduler', {})
        self.dataset_params = params.get('dataset', {'data_dir': 'data'})
        self.dataloader_params = params.get('dataloader', {})
        self.trainer_params = params.get('trainer', {})

        self.logger = params.get('logger', None)
    
    def get_model(self, model_params = None):
        """
        Get the model from configuration.

        Parameters
        ----------
        
        model_params: dict, optional, default: None. If model_params is provided, then use the parameters specified in the model_params to build the model. Otherwise, the model parameters in the self.params will be used to build the model.
        
        Returns
        -------
        
        A model, which is usually a torch.nn.Module object.
        """
        
        from models.Iter_model import Iter_model
        
        if model_params is None:
            model_params = self.model_params
        type = model_params.get('type', 'VQGAN')
        params = model_params.get('params', {})
        if type == 'Iter_model':
            model = Iter_model(self.logger, **params)
        else:
            raise NotImplementedError('Invalid model type.')
        return model
    

    
    def get_dataset(self, dataset_params = None, split = 'train'):
        """
        Get the dataset from configuration.

        Parameters
        ----------
        
        dataset_params: dict, optional, default: None. If dataset_params is provided, then use the parameters specified in the dataset_params to build the dataset. Otherwise, the dataset parameters in the self.params will be used to build the dataset;
        
        split: str in ['train', 'test'], optional, default: 'train', the splitted dataset.

        Returns
        -------
        
        A torch.utils.data.Dataset item.
        """
        from dataset.GRIDSAT_dataset_full_crop_fengwu_pre import GRIDSAT_crop_dataset_fengwu_pre
        from dataset.GRIDSAT_dataset_full_crop_fengwu_pre_vae import GRIDSAT_crop_dataset_fengwu_pre_vae


        if dataset_params is None:
            dataset_params = self.dataset_params
        dataset_params = dataset_params.get(split, None)
        if dataset_params is None:
            return None
        if type(dataset_params) == dict:
            dataset_type = str.lower(dataset_params.get('type', 'gridsat_crop_fengwu_pre'))
            print(dataset_type)
            if dataset_type == 'gridsat_crop_fengwu_pre':
                dataset = GRIDSAT_crop_dataset_fengwu_pre(split = split, **dataset_params)
            elif dataset_type == 'gridsat_crop_fengwu_pre_vae':
                dataset = GRIDSAT_crop_dataset_fengwu_pre_vae(split = split, **dataset_params)
            else:
                raise NotImplementedError('Invalid dataset type: {}.'.format(dataset_type))
            # logger.info('Load {} dataset as {}ing set with {} samples.'.format(dataset_type, split, len(dataset)))
        else:
            raise AttributeError('Invalid dataset format.')
        return dataset
    
    def get_sampler(self, dataset, split = 'train', drop_last=False, sampler_mode="transformer_base"):
        if split == 'train':
            shuffle = True
        else:
            shuffle = False

        # rank = get_rank()
        # num_gpus = get_world_size()
        if is_dist_avail_and_initialized():
            rank = mpu.get_data_parallel_rank()
            num_gpus = mpu.get_data_parallel_world_size()
        else:
            rank = 0
            num_gpus = 1
        

        # sampler = DistributedSampler(dataset, rank=rank, shuffle=shuffle, num_replicas=num_gpus, seed=0, drop_last=drop_last)


        if split == 'train' and (sampler_mode=="transformer_base_class_2" or sampler_mode=="transformer_base_class"):
            #
            if sampler_mode=="transformer_base_class_2":
                class_sample_counts = [12470+8148+12259, 1372]
            else:
                class_sample_counts = [12470, 8148, 12259, 1372]
            weights = 1./ torch.tensor(class_sample_counts, dtype=torch.float)
            # 
            train_targets = dataset.get_classes_for_all_imgs()
            samples_weights = weights[train_targets]

            sampler = WeightedRandomSampler(dataset, weights=samples_weights, num_samples=len(samples_weights), replacement=True, rank=rank, num_replicas=num_gpus, drop_last=drop_last)
        
        else:
            sampler = DistributedSampler(dataset, rank=rank, shuffle=shuffle, num_replicas=num_gpus, seed=0, drop_last=drop_last)






        return sampler
   

    def get_dataloader(self, dataset_params = None, split = 'train', batch_size = None, dataloader_params = None, sampler_mode="transformer_base"):
        """
        Get the dataloader from configuration.

        Parameters
        ----------
        
        dataset_params: dict, optional, default: None. If dataset_params is provided, then use the parameters specified in the dataset_params to build the dataset. Otherwise, the dataset parameters in the self.params will be used to build the dataset;
        
        split: str in ['train', 'test'], optional, default: 'train', the splitted dataset;
        
        batch_size: int, optional, default: None. If batch_size is None, then the batch size parameter in the self.params will be used to represent the batch size (If still not specified, default: 4);
        
        dataloader_params: dict, optional, default: None. If dataloader_params is provided, then use the parameters specified in the dataloader_params to get the dataloader. Otherwise, the dataloader parameters in the self.params will be used to get the dataloader.

        Returns
        -------
        
        A torch.utils.data.DataLoader item.
        """

        if is_dist_avail_and_initialized() and mpu.get_tensor_model_parallel_rank() > 0:
            return None
        from torch.utils.data import DataLoader
        # from petrel_client.utils.data import DataLoader
        drop_last = False
        if batch_size is None:
            if split == 'train':
                batch_size = self.trainer_params.get('batch_size', 32)
            elif split == "test":
                batch_size = self.trainer_params.get('test_batch_size', 1)
            else:
                batch_size = self.trainer_params.get('valid_batch_size', 1)
        if split != "train":
            drop_last = True
        if dataloader_params is None:
            dataloader_params = self.dataloader_params
        dataset = self.get_dataset(dataset_params, split)
        if dataset is None:
            return None
        sampler = self.get_sampler(dataset, split, drop_last=drop_last, sampler_mode=sampler_mode)
        return DataLoader(
            dataset,
            batch_size = batch_size,
            sampler=sampler,
            drop_last=drop_last,
            **dataloader_params
        )

    def get_max_epoch(self, trainer_params = None):
        """
        Get the max epoch from configuration.

        Parameters
        ----------
        
        trainer_params: dict, optional, default: None. If trainer_params is provided, then use the parameters specified in the trainer_params to get the maximum epoch. Otherwise, the trainer parameters in the self.params will be used to get the maximum epoch.

        Returns
        -------
        
        An integer, which is the max epoch (default: 40).
        """
        if trainer_params is None:
            trainer_params = self.trainer_params
        return trainer_params.get('max_epoch', 40)
    
    def get_stats_dir(self, stats_params = None):
        """
        Get the statistics directory from configuration.

        Parameters
        ----------
        
        stats_params: dict, optional, default: None. If stats_params is provided, then use the parameters specified in the stats_params to get the statistics directory. Otherwise, the statistics parameters in the self.params will be used to get the statistics directory.

        Returns
        -------
        
        A string, the statistics directory.
        """
        if stats_params is None:
            stats_params = self.stats_params
        stats_dir = stats_params.get('stats_dir', 'stats')
        stats_exper = stats_params.get('stats_exper', 'default')
        stats_res_dir = os.path.join(stats_dir, stats_exper)
        if os.path.exists(stats_res_dir) == False:
            os.makedirs(stats_res_dir)
        return stats_res_dir
    
    def get_resume_lr(self, trainer_params = None):
        """
        Get the resume learning rate from configuration.

        Parameters
        ----------

        trainer_params: dict, optional, default: None. If trainer_params is provided, then use the parameters specified in the trainer_params to get the resume learning rate. Otherwise, the trainer parameters in the self.params will be used to get the resume learning rate.

        Returns
        -------

        A float value, the resume learning rate (default: 0.001).
        """
        if trainer_params is None:
            trainer_params = self.trainer_params
        return trainer_params.get('resume_lr', 0.001)

    
    def get_metrics(self, metrics_params = None):
        """
        Get the metrics settings from configuration.

        Parameters
        ----------

        metrics_params: dict, optional, default: None. If metrics_params is provided, then use the parameters specified in the metrics_params to get the metrics. Otherwise, the metrics parameters in the self.params will be used to get the metrics.
        
        Returns
        -------

        A MetricsRecorder object.
        """
        if metrics_params is None:
            metrics_params = self.metrics_params
        metrics_list = metrics_params.get('types', ['MSE', 'MaskedMSE', 'RMSE', 'MaskedRMSE', 'REL', 'MaskedREL', 'MAE', 'MaskedMAE', 'WRMSE', 'WACC'])
        from utils.metrics import MetricsRecorder
        metrics = MetricsRecorder(metrics_list = metrics_list, **metrics_params)
        return metrics
    
    def get_inference_checkpoint_path(self, inference_params = None):
        """
        Get the inference checkpoint path from inference configuration.

        Parameters
        ----------

        inference_params: dict, optional, default: None. If inference_params is provided, then use the parameters specified in the inference_params to get the inference checkpoint path. Otherwise, the inference parameters in the self.params will be used to get the inference checkpoint path.
        
        Returns
        -------

        str, the checkpoint path.
        """
        if inference_params is None:
            inference_params = self.inference_params
        return inference_params.get('checkpoint_path', os.path.join('checkpoint', 'checkpoint.tar'))
    
    def get_inference_cuda_id(self, inference_params = None):
        """
        Get the inference CUDA ID from inference configuration.

        Parameters
        ----------

        inference_params: dict, optional, default: None. If inference_params is provided, then use the parameters specified in the inference_params to get the inference CUDA ID. Otherwise, the inference parameters in the self.params will be used to get the inference CUDA ID.
        
        Returns
        -------

        int, the CUDA ID.
        """
        if inference_params is None:
            inference_params = self.inference_params
        return inference_params.get('cuda_id', 0)


def get_optimizer(model, optimizer_params = None, resume = False, resume_lr = None):
    """
    Get the optimizer from configuration.
    
    Parameters
    ----------
    
    model: a torch.nn.Module object, the model.
    
    optimizer_params: dict, optional, default: None. If optimizer_params is provided, then use the parameters specified in the optimizer_params to build the optimizer. Otherwise, the optimizer parameters in the self.params will be used to build the optimizer;
    
    resume: bool, optional, default: False, whether to resume training from an existing checkpoint;

    resume_lr: float, optional, default: None, the resume learning rate.
    
    Returns
    -------
    
    An optimizer for the given model.
    """
    from torch.optim import SGD, ASGD, Adagrad, Adamax, Adadelta, Adam, AdamW, RMSprop
    from utils.optim import Adan, gassopti, gasspidopti, Agassopti, Lion
    # from apex import optimizers
    type = optimizer_params.get('type', 'AdamW')
    params = optimizer_params.get('params', {})



    if resume:
        network_params = [{'params': model.parameters(), 'initial_lr': resume_lr}]
        params.update(lr = resume_lr)
    else:
        network_params = model.parameters()
    if type == 'SGD':
        optimizer = SGD(network_params, **params)
    elif type == 'ASGD':
        optimizer = ASGD(network_params, **params)
    elif type == 'Adagrad':
        optimizer = Adagrad(network_params, **params)
    elif type == 'Adamax':
        optimizer = Adamax(network_params, **params)
    elif type == 'Adadelta':
        optimizer = Adadelta(network_params, **params)
    elif type == 'Adam':
        optimizer = Adam(network_params, **params)
    elif type == 'AdamW':
        optimizer = AdamW(network_params, **params)
    elif type == 'RMSprop':
        optimizer = RMSprop(network_params, **params)
    elif type == 'Adan':
        optimizer = Adan(network_params, **params)
    elif type == 'gassopti':
        optimizer = gassopti(network_params, **params)
    elif type == 'gasspidopti':
        optimizer = gasspidopti(network_params, **params)
    elif type == 'Agassopti':
        optimizer = Agassopti(network_params, **params)
    elif type == 'lamb':
        optimizer = Lamb(network_params, **params)
    elif type == 'lion':
        optimizer = Lion(network_params, **params)
    # elif type == 'FusedAdam':
    #     optimizer = optimizers.FusedAdam(network_params, **params)
    else:
        raise NotImplementedError('Invalid optimizer type.')
    return optimizer

# def get_lr_scheduler(optimizer, lr_scheduler_params = None, resume = False, resume_epoch = None):
#     """
#     Get the learning rate scheduler from configuration.
    
#     Parameters
#     ----------
    
#     optimizer: an optimizer;
    
#     lr_scheduler_params: dict, optional, default: None. If lr_scheduler_params is provided, then use the parameters specified in the lr_scheduler_params to build the learning rate scheduler. Otherwise, the learning rate scheduler parameters in the self.params will be used to build the learning rate scheduler;

#     resume: bool, optional, default: False, whether to resume training from an existing checkpoint;

#     resume_epoch: int, optional, default: None, the epoch of the checkpoint.
    
#     Returns
#     -------

#     A learning rate scheduler for the given optimizer.
#     """
#     from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, CyclicLR, CosineAnnealingLR, LambdaLR, StepLR, OneCycleLR, ReduceLROnPlateau
#     type = lr_scheduler_params.get('type', '')
#     params = lr_scheduler_params.get('params', {})
#     if resume:
#         params.update(last_epoch = resume_epoch)
#     if type == 'MultiStepLR':
#         scheduler = MultiStepLR(optimizer, **params)
#     elif type == 'ExponentialLR':
#         scheduler = ExponentialLR(optimizer, **params)
#     elif type == 'CyclicLR':
#         scheduler = CyclicLR(optimizer, **params)
#     elif type == 'CosineAnnealingLR':
#         scheduler = CosineAnnealingLR(optimizer, **params)
#     elif type == 'LambdaLR':
#         scheduler = LambdaLR(optimizer, **params)
#     elif type == 'ReduceLROnPlateau':
#         scheduler = ReduceLROnPlateau(optimizer, **params)
#     elif type == 'StepLR':
#         scheduler = StepLR(optimizer, **params)
#     elif type == 'OneCycleLR':
#         scheduler = OneCycleLR(optimizer, **params)
#     elif type == '':
#         scheduler = None
#     else:
#         raise NotImplementedError('Invalid learning rate scheduler type.')
#     return scheduler


def get_lr_scheduler(optimizer, lr_scheduler_params = None, resume = False, resume_epoch = None):
    """
    Get the learning rate scheduler from configuration.
    
    Parameters
    ----------
    
    optimizer: an optimizer;
    
    lr_scheduler_params: dict, optional, default: None. If lr_scheduler_params is provided, then use the parameters specified in the lr_scheduler_params to build the learning rate scheduler. Otherwise, the learning rate scheduler parameters in the self.params will be used to build the learning rate scheduler;

    resume: bool, optional, default: False, whether to resume training from an existing checkpoint;

    resume_epoch: int, optional, default: None, the epoch of the checkpoint.
    
    Returns
    -------

    A learning rate scheduler for the given optimizer.
    """
    # type = lr_scheduler_params.get('type', '')
    # params = lr_scheduler_params.get('params', {})

    scheduler_args = dictToObj(lr_scheduler_params)
    # if resume:
    #     params.update(last_epoch = resume_epoch)
    from torch.optim.lr_scheduler import LambdaLR, ExponentialLR, LinearLR
    if scheduler_args.sched in ["cosine", "tanh", "step", "multistep", "plateau", "poly"]:
        scheduler, _ = create_scheduler(scheduler_args, optimizer)
    elif scheduler_args.sched == "exponential":
        begin_lr = optimizer.state_dict()['param_groups'][0]['lr']
        end_lr = scheduler_args.min_lr
        gamma = np.exp(np.log(end_lr/begin_lr)/scheduler_args.epochs)
        scheduler = ExponentialLR(optimizer=optimizer, gamma=gamma)
    elif scheduler_args.sched == "linear":
        begin_lr = optimizer.state_dict()['param_groups'][0]['lr']
        end_lr = scheduler_args.min_lr
        scheduler = LinearLR(optimizer=optimizer, start_factor=1.0, end_factor=(begin_lr-end_lr)/scheduler_args.epochs,total_iters=scheduler_args.epochs)


    return scheduler
