import torch
import torch.nn as nn



from networks.FengWu_TC import VQ_TC_Model_ERA5, FengWu_TC


# from networks.PVFlash import BidirectionalTransformer
from utils.builder import get_optimizer, get_lr_scheduler
from utils.metrics import MetricsRecorder
import utils.misc as utils
import time
import datetime
from pathlib import Path
import torch.cuda.amp as amp
from utils.checkpoint_ceph import checkpoint_ceph, data_ceph
import os
from collections import OrderedDict
from torch.functional import F

from megatron_utils import mpu
from utils.misc import is_dist_avail_and_initialized
from megatron_utils.tensor_parallel.data import broadcast_data,get_data_loader_length
from torch.distributions import Normal
import warnings

import gc

import pandas as pd
import numpy as np


import wandb
import random # for demo script
import os
import io
from petrel_client.client import Client

LOG_SIG_MAX = 5
LOG_SIG_MIN = -8

class EMA():
    def __init__(self, model, decay, add_to_train=False):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.add_to_train = add_to_train
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name].to(param.device)
                if self.add_to_train:
                    param.data.copy_ ((1.0 - self.decay) * param.data + self.decay * self.shadow[name].to(param.device))                 
                self.shadow[name] = new_average.clone()


    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name].to(param.device)
        self.backup = {}

class pretrain_checkpoint_ceph(object):
    def __init__(self, conf_path="~/petreloss.conf", checkpoint_dir="cephnew:s3://myBucket/my_checkpoint") -> None:
        self.client = Client(conf_path=conf_path)
        self.checkpoint_dir = checkpoint_dir
        

    def load_checkpoint(self, url):
        url = os.path.join(self.checkpoint_dir, url)
        # url = self.checkpoint_dir + "/" + url
        if not self.client.contains(url):
            return None
        with io.BytesIO(self.client.get(url, update_cache=True)) as f:
            checkpoint_data = torch.load(f, map_location=torch.device('cpu')) 
        return checkpoint_data

class basemodel(nn.Module):
    def __init__(self, logger, **params) -> None:
        super().__init__()
        self.model = {}
        self.sub_model_name = []
        self.params = params
        self.logger = logger
        self.save_best_param = self.params.get("save_best", "MSE")
        self.metric_best = None
        self.constants_len = self.params.get("constants_len", 0)
        self.extra_params = params.get("extra_params", {})
        self.loss_type = self.extra_params.get("loss_type", "LpLoss")
        self.L1_re_rate = self.extra_params.get("L1_re_rate", 0.00001)
        self.enabled_amp = self.extra_params.get("enabled_amp", False)
        self.output_step_length = params.get("output_step_length", 1)
        self.pre_len = params.get("pre_len", 4)
        
        self.Model_choose = params.get("Model_choose", ["FPN", "CCAtt", "AR"])
        self.is_hid_vq = params.get("is_hid_vq", False)
        self.inp_len = params.get("inp_len", 4)


        self.is_load_pretrain = params.get("is_load_pretrain", False)
        if self.is_load_pretrain:
            self.pretrain_path = params.get("pretrain_path", "FengWu_TC_physics/world_size4-FengWu_TC_physics/checkpoint_best.pth")

        self.is_save_vq_state = params.get("is_save_vq_state", False)
        self.vq_name = params.get("vq_name", "MSW20")

        self.begin_epoch = 0
        self.metric_best = 1000

        # self.gscaler = amp.GradScaler(init_scale=1024, growth_interval=2000)
        self.gscaler = amp.GradScaler(enabled=self.enabled_amp)

        self.checkpoint_ceph = checkpoint_ceph()
        # self.data_save_ceph = data_ceph()
        self.use_ceph = self.params.get('use_ceph', True)
        
        # self.whether_final_test = self.params.get("final_test", False)
        # self.predict_length = self.params.get("predict_length", 20)


        # load model
        # print(params)
        sub_model = params.get('sub_model', {})
        # print(sub_model)
        for key in sub_model:
            self.model_name = key
            
            if key =="VQ_TC_ERA5":
                self.model[key] = VQ_TC_Model_ERA5(**sub_model['VQ_TC_ERA5'])
        
            elif key =="FengWu_TC":
                self.model[key] = FengWu_TC(**sub_model['FengWu_TC'])
            else:
                raise NotImplementedError('Invalid model type.')
            
            if self.is_load_pretrain:
                self.model[key] = self.load_pretrain_checkpoint(self.model[key], checkpoint_path=self.pretrain_path)

            if self.loss_type == "Possloss" or self.loss_type == "RF_Loss" or self.loss_type == "NLgpossloss":
                output_dim = self.params['sub_model'][list(self.model.keys())[0]]["out_chans"]
                img_size = self.params['sub_model'][list(self.model.keys())[0]].get("img_size", [32, 64])
                self.max_logvar = self.model[key].max_logvar = torch.nn.Parameter((torch.ones((1, output_dim*img_size[-2]*img_size[-1]//2)).float() / 2))
                self.min_logvar = self.model[key].min_logvar = torch.nn.Parameter((-torch.ones((1, output_dim*img_size[-2]*img_size[-1]//2)).float() * 10))
            elif self.loss_type == "Ema_Weight_Loss":
                output_dim = self.params['sub_model'][list(self.model.keys())[0]]["out_chans"]
                img_size = self.params['sub_model'][list(self.model.keys())[0]].get("img_size", [32, 64])
                self.ema_weight_params = torch.zeros(output_dim, *img_size).float()
                self.beta = 0.99
                self.ema_step = 0

            if self.loss_type == "Possloss":
                self.weight_begin_index = self.extra_params.get("weight_begin_index", 3)
                self.weight_end_index = self.extra_params.get("weight_end_index", 17)
                self.weight_number = self.extra_params.get("weight_number", 1)
            elif self.loss_type == "NLgpossloss":
                self.log_sig_max = self.extra_params.get("log_sig_max", 10)
                self.log_sig_min = self.extra_params.get("log_sig_min", -20)

                
            
            self.sub_model_name.append(key)

        # load optimizer and lr_scheduler
        self.optimizer = {}
        self.lr_scheduler = {}
        self.lr_scheduler_by_step = {}

        optimizer = params.get('optimizer', {})
        lr_scheduler = params.get('lr_scheduler', {})
        # print(optimizer)
        # print(lr_scheduler)

        for key in self.sub_model_name:
            if (key != "HA") and (key != "direct_intensity") :
                if key in optimizer:
                    self.optimizer[key] = get_optimizer(self.model[key], optimizer[key])
                    op_params = optimizer[key].get('params', {})
                    self.lr_rate = op_params.get('lr', 0.0001)
                if key in lr_scheduler:
                    self.lr_scheduler_by_step[key] = lr_scheduler[key].get('by_step', False)
                    self.lr_scheduler[key] = get_lr_scheduler(self.optimizer[key], lr_scheduler[key])

        # load metrics
        eval_metrics_list = params.get('metrics_list', [])
        if len(eval_metrics_list) > 0:
            self.eval_metrics = MetricsRecorder(eval_metrics_list)
        else:
            self.eval_metrics = None

        for key in self.model:
            self.model[key].eval()


        
        self.replay_buff_params = self.extra_params.get("replay_buff", None)
        self.two_step_training = self.extra_params.get("two_step_training", False)
        self.use_noise = self.extra_params.get("use_noise", False)
        # if self.two_step_training:
        self.checkpoint_path = self.extra_params.get('checkpoint_path', None)
        if self.checkpoint_path is None:
            self.logger.info("finetune checkpoint path not exist")
        else:
            self.load_checkpoint(self.checkpoint_path, load_model=True, load_optimizer=False, load_scheduler=False, load_epoch=False, load_metric_best=False)
        
        
        
        if self.loss_type == "LpLoss":
            self.loss = self.LpLoss
        elif self.loss_type == "Possloss":
            self.loss = self.Possloss
        elif self.loss_type == "NLgpossloss":
            self.loss = self.NLgpossloss
        elif self.loss_type == "MAELoss":
            self.loss = self.MAELoss
        elif self.loss_type == "MSELoss":
            self.loss = self.MSELoss
        elif self.loss_type == "StdLoss":
            self.loss = self.StdLoss
        elif self.loss_type == "signLoss":
            self.loss = self.signLoss
        elif self.loss_type == "Weight_diff_Loss":
            self.loss = self.Weight_diff_Loss
        elif self.loss_type == "Ema_Weight_Loss":
            self.loss = self.Ema_Weight_Loss
        elif self.loss_type == "RF_Loss":
            self.loss = self.RF_Loss
        elif self.loss_type == "CE_Loss":
            self.loss = self.CE_Loss
        elif self.loss_type == "weighted_mse_loss":
            self.loss = self.weighted_mse_loss
        elif self.loss_type == "weighted_mae_loss":
            self.loss = self.weighted_mae_loss       
        elif self.loss_type == "diff2_mae_loss":
            self.loss = self.diff2_mae_loss
        self.wandb_name = params.get("wandb_name", "TC_Pre")
        #self.wandb_init()
        self.is_add_RI_Loss = params.get('is_add_RI_Loss', False)
        if self.is_add_RI_Loss:
            self.addition_Loss = self.RI_Loss
            self.RI_threshold = params.get('RI_threshold', 30.0)


        self.is_diff = params.get("is_diff", False)


        
        self.ema_config = params.get("ema_config", {})
        self.is_ema = self.ema_config.get('is_ema', False)
        if self.is_ema:
            self.ema_decay = self.ema_config.get('ema_decay', 0.999)
            self.ema_add_to_train = self.ema_config.get('ema_add_to_train', False)
            self.ema = EMA(self.model[list(self.model.keys())[0]], decay=self.ema_decay, add_to_train=self.ema_add_to_train)
            self.ema.register()

        self.cmp_mode = self.params.get("cmp_mode", "Many_to_Many")


    def load_pretrain_checkpoint(self, mymodel, checkpoint_path, load_model=True, ):
        checkpoint_ceph = pretrain_checkpoint_ceph()
        checkpoint_dict = checkpoint_ceph.load_checkpoint(checkpoint_path)
        if checkpoint_dict is None:
            self.logger.info("checkpoint is not exist")
            return
        
        checkpoint_model = checkpoint_dict['model']
        print(f"load epoch {checkpoint_dict['epoch']} {checkpoint_path}")
        if load_model:
            for key in checkpoint_model:
                new_state_dict = OrderedDict()
                for k, v in checkpoint_model[key].items():
                    if "module" == k[:6]:
                        name = k[7:]
                    else:
                        name = k
                    
                    new_state_dict[name] = v

                mymodel.load_state_dict(new_state_dict, strict=False)
        return mymodel



    def Pre_load_checkpoint(self, checkpoint_path, use_ceph=True, load_model=True):
        if utils.get_world_size() > 1 and mpu.get_tensor_model_parallel_world_size() > 1:
            path1, path2 = checkpoint_path.split('.')
            checkpoint_path = f"{path1}_{mpu.get_tensor_model_parallel_rank()}{path2}"
        
        if use_ceph:
            print(checkpoint_path)
            checkpoint_dict = self.Pre_checkpoint_ceph.load_checkpoint(checkpoint_path)
  
            if checkpoint_dict is None:
                return
        elif os.path.exists(checkpoint_path):
          
            checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            #checkpoint_dict = torch.load(checkpoint_path, map_location=self.device)
        else:
            return
        checkpoint_model = checkpoint_dict['model']
        if load_model:
            for key in checkpoint_model:
                new_state_dict = OrderedDict()
                for k, v in checkpoint_model[key].items():
                    if "module" == k[:6]:
                        name = k[7:]
                    else:
                        name = k
                    new_state_dict[name] = v

                self.class_pre_model[key].load_state_dict(new_state_dict, strict=False)
                # self.model[key].load_state_dict(checkpoint_model[key])
    


    def to(self, device):
        self.device = device
        # import pdb
        # pdb.set_trace()
        for key in self.model:
            self.model[key].to(device)
        

        for key in self.optimizer:
            for state in self.optimizer[key].state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        if self.loss_type == "Ema_Weight_Loss":
            self.ema_weight_params = self.ema_weight_params.to(device)
        # if hasattr(self, 'max_logvar') and self.max_logvar is not None:
        #     self.max_logvar = self.max_logvar.to(device)
        #     # self.max_logvar.requires_grad=True
        # if hasattr(self, 'min_logvar') and self.min_logvar is not None:
        #     self.min_logvar = self.min_logvar.to(device)
            # self.min_logvar.requires_grad=True
        

    def data_preprocess(self, data):
        return None, None

    def Ema_Weight_Loss(self, pred, target, **kwargs):

        # if not hasattr(self, 'ema_weight_params'):
        #     self.ema_weight_params = torch.ones((target.shape[1])).float().to(self.device)
        self.ema_step += 1
        # loss = (torch.abs(pred - target)).permute(1, 0, 2, 3).reshape(target.shape[1], -1).mean(dim=-1)
        loss = (torch.abs(pred - target)).mean(dim=0)
        self.ema_weight_params = self.beta * self.ema_weight_params + (1 - self.beta) * loss.detach()
        
        if is_dist_avail_and_initialized():
            utils.dist.barrier()
            utils.dist.all_reduce(self.ema_weight_params)
            self.ema_weight_params = self.ema_weight_params / utils.get_world_size()

        # loss = loss / (self.ema_weight_params * (0.9 + torch.rand_like(self.ema_weight_params, device=self.device) * 0.2))

        # target = (target - self.diff_mean) / self.diff_std
        # num_examples = pred.size()[0]
        # dim = pred.size()[1]

        # diff_norms = torch.norm(pred.reshape(num_examples, dim, -1) - target.reshape(num_examples, dim, -1), 2, 2)
        # y_norms = torch.norm(target.reshape(num_examples, dim, -1), 2, 2)
        return torch.mean(loss)
        # return torch.mean(diff_norms/y_norms)

    
    def Weight_diff_Loss(self, pred, target, **kwargs):
        
        target = (target - self.diff_mean) / self.diff_std
        # num_examples = pred.size()[0]
        # dim = pred.size()[1]

        # diff_norms = torch.norm(pred.reshape(num_examples, dim, -1) - target.reshape(num_examples, dim, -1), 2, 2)
        # y_norms = torch.norm(target.reshape(num_examples, dim, -1), 2, 2)
        return torch.mean((pred-target)**2)
        # return torch.mean(diff_norms/y_norms)

    def MAELoss(self, pred, target, **kwargs):
        return torch.abs(pred-target).mean()

    def MSELoss(self, pred, target, **kwargs):
        loss = nn.MSELoss()
        output = loss(pred, target)
        return output

    def StdLoss(self, pred, target, **kwargs):
        num_examples = pred.size()[0]

        # diff_norms = torch.norm(pred - target, 2, [-2, -1])

        # res = diff_norms * self.datastd.unsqueeze(0) / 100

        # return(torch.mean(res))
        diff_norms = torch.abs(pred - target) * (self.datastd ** 0.5).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        # diff_norms = (pred - target) ** 2 * (100*torch.sin(self.datastd/self.datastd.max()*torch.pi/2) + 1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        # diff_norms = (pred - target)**2 * (1 + torch.log(self.datastd + 1)).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        return torch.mean(diff_norms)



    def signLoss(self, pred, target, **kwargs):
        eps=1e-8
        diff_norms = torch.abs(pred-target)
        res = diff_norms / (diff_norms.detach() + eps)
        return torch.mean(res)


    def LpLoss(self, pred, target, **kwargs):
        num_examples = pred.size()[0]

        diff_norms = torch.norm(pred.reshape(num_examples,-1) - target.reshape(num_examples,-1), 2, 1)
        y_norms = torch.norm(target.reshape(num_examples,-1), 2, 1)

        return torch.mean(diff_norms/y_norms)


    def Possloss(self, pred, target, **kwargs):
        
        inc_var_loss = kwargs.get("inc_var_loss", True)
        
        num_examples = pred.size()[0]

        mean, log_var = pred.chunk(2, dim = 1)
        # log_var = torch.tanh(log_var)

        # mean = mean.reshape(num_examples, -1)
        log_var = log_var.reshape(num_examples, -1)
        # target = target.reshape(num_examples, -1)





        log_var = self.max_logvar - F.softplus(self.max_logvar - log_var)
        log_var = self.min_logvar + F.softplus(log_var - self.min_logvar)

        log_var = log_var.reshape(*(target.shape))
        weight = torch.ones_like(target, device=target.device)
        weight[:, self.weight_begin_index:self.weight_end_index, :, :] = self.weight_number

        inv_var = torch.exp(-log_var)
        if inc_var_loss:
            # Average over batch and dim, sum over ensembles.
            mse_loss = torch.mean(torch.pow(mean - target, 2) * inv_var * weight)
            var_loss = torch.mean(log_var * weight)

            # mse_loss = torch.mean(torch.mean(torch.pow(mean - target, 2) * inv_var, dim=-1), dim=-1)
            # var_loss = torch.mean(torch.mean(log_var, dim=-1), dim=-1)
            total_loss = mse_loss + var_loss
        else:
            mse_loss = torch.mean(torch.pow(mean - target, 2), dim=(1, 2))
            total_loss = mse_loss
            
        total_loss += 0.01 * torch.mean(self.max_logvar) - 0.01 * torch.mean(self.min_logvar)
        return total_loss


    def NLgpossloss(self, pred, target, **kwargs):
        
        inc_var_loss = kwargs.get("inc_var_loss", True)
        
        num_examples = pred.size()[0]

        mean, log_var = pred.chunk(2, dim = 1)
        # log_var = torch.tanh(log_var)

        mean = mean.reshape(num_examples, -1)
        log_var = log_var.reshape(num_examples, -1)
        target = target.reshape(num_examples, -1)



        log_var = self.max_logvar - F.softplus(self.max_logvar - log_var)
        log_var = self.min_logvar + F.softplus(log_var - self.min_logvar)

        log_var = torch.clamp(log_var, min=self.log_sig_min, max=self.log_sig_max)

        inv_var = torch.exp(-log_var)
        if inc_var_loss:
            # Average over batch and dim, sum over ensembles.
            mse_loss = torch.mean(torch.mean(torch.pow(mean - target, 2) * inv_var, dim=-1), dim=-1)
            var_loss = torch.mean(torch.mean(log_var, dim=-1), dim=-1)
            total_loss = mse_loss + var_loss
        else:
            mse_loss = torch.mean(torch.pow(mean - target, 2), dim=(1, 2))
            total_loss = mse_loss
            
        total_loss += 0.01 * torch.mean(self.max_logvar) - 0.01 * torch.mean(self.min_logvar)
        return total_loss

    def RF_Loss(self, pred, target, **kwargs):
        num_lat = pred.shape[2]
        
        predict_mean, log_var = pred.chunk(2, dim = 1)
       


        log_var = self.max_logvar.reshape(*(target.shape[1:])) - F.softplus(self.max_logvar.reshape(*(target.shape[1:])) - log_var)
        log_var = self.min_logvar.reshape(*(target.shape[1:])) + F.softplus(log_var - self.min_logvar.reshape(*(target.shape[1:])))


        predict_var = torch.exp(log_var)
        predict_std = torch.sqrt(predict_var)
        normal_func = Normal(predict_mean, predict_std)
        # pred_sample = normal_func.rsample()
        log_prob = normal_func.log_prob(target)


        lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
        s = torch.sum(torch.cos(3.1416 / 180. * (90. - lat_t * 180. / float(num_lat - 1))))
        weight = torch.reshape(num_lat * torch.cos(3.1416 / 180. * (90. - lat_t * 180. / float(num_lat - 1))) / s, (1, 1, -1, 1))
        reward = 0 - (weight * (predict_mean.detach() - target.detach()) ** 2) ** 0.5 * self.datastd.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

        loss = torch.mean(reward * log_prob) + 0.01 * torch.mean(self.max_logvar) - 0.01 * torch.mean(self.min_logvar)
        return loss

    def CE_Loss(self, pred, target, **kwargs):
        ce_loss = nn.CrossEntropyLoss()
        target = target.long()        
        return ce_loss(pred, target)
    
    def softplus(self, x):
        return torch.log(1+torch.exp(x))
    
    def RI_Loss(self, input, pred, target, std, RI_threshold, **kwargs):
        input = input[:, :, 0]*std[0]
        pred = pred[:, :, 0]*std[0]
        target = target[:, :, 0]*std[0]
        v_inp = input
        v_pred = pred
        v_target = target
        v_seq_true = torch.cat((v_inp, v_target), dim=-1)
        v_seq_true_crop = v_seq_true[:, (-v_pred.shape[-1]-4):-4]
        RI_diff = (v_pred - v_seq_true_crop - RI_threshold)/std[0]
        RI_diff = self.softplus(RI_diff)*5.0 + 1.0
        ad_loss = torch.mean(RI_diff)
        #print("diff: {}".format((v_pred - v_seq_true_crop - RI_threshold)))
        #print("ad_loss: {}".format(ad_loss.item()))
        return ad_loss

    def weighted_mse_loss(self, pred, targets, weights=None):
        loss = (pred - targets) ** 2
        if weights is not None:
            loss *= weights.expand_as(loss)
        loss = torch.mean(loss)
        return loss

    def weighted_mae_loss(self, pred, targets, weights=None):
        loss = torch.abs(pred - targets)
        if weights is not None:
            loss *= weights.expand_as(loss)
        loss = torch.mean(loss)
        return loss

    def diff2_mae_loss(self, net, pred, target):
        
     
        loss = torch.abs(pred-target).mean()
        L1_reg = 0
        for param in net.parameters():
            if param.requires_grad:
                L1_reg += torch.sum(torch.abs(param))
        loss += self.L1_re_rate * L1_reg  # lambda=0.0001

        return loss

  
    def train_one_step(self, batch_data, step, data_std):
        input, target = self.data_preprocess(batch_data)
        if len(self.model) == 1:
            predict = self.model[list(self.model.keys())[0]](input)
        else:
            raise NotImplementedError('Invalid model type.')
        
        loss = self.loss(predict, target)
        if len(self.optimizer) == 1:
            self.optimizer[list(self.optimizer.keys())[0]].zero_grad()
            loss.backward()
            self.optimizer[list(self.optimizer.keys())[0]].zero_grad()
        else:
            raise NotImplementedError('Invalid model type.')
        
        return loss

    def multi_step_predict(self, batch_data, clim_time_mean_daily, data_std, index, batch_len):
        pass


    def test_one_step(self, batch_data, step):
        input, target = self.data_preprocess(batch_data)
        if len(self.model) == 1:
            predict = self.model[list(self.model.keys())[0]](input)

        data_dict = {}
        data_dict['gt'] = target
        data_dict['pred'] = predict
        if MetricsRecorder is not None:
            loss = self.eval_metrics(data_dict)
        else:
            raise NotImplementedError('No Metric Exist.')
        return loss


    def train_one_epoch(self, train_data_loader, epoch, max_epoches):

        # if epoch > self.begin_epoch:
        #     for key in self.lr_scheduler:
        #         if not self.lr_scheduler_by_step[key]:
        #             self.lr_scheduler[key].step(epoch, torch.tensor(self.metric_logger.meters[self.save_best_param].global_avg))
        # else:
        #     for key in self.lr_scheduler:
        #         if not self.lr_scheduler_by_step[key]:
        #             self.lr_scheduler[key].step(epoch, torch.tensor(self.metric_best))
        for key in self.lr_scheduler:
            if not self.lr_scheduler_by_step[key]:
                self.lr_scheduler[key].step(epoch)


        # test_logger = {}


        end_time = time.time()           
        for key in self.optimizer:              # only train model which has optimizer
            self.model[key].train()

        metric_logger = utils.MetricLogger(delimiter="  ", sync=True)
        iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
        data_time = utils.SmoothedValue(fmt='{avg:.3f}')

        max_step = get_data_loader_length(train_data_loader)
        # if is_dist_avail_and_initialized() and mpu.get_tensor_model_parallel_world_size() > 1:
        #     if train_data_loader is not None:
        #         max_step = torch.tensor(len(train_data_loader))
        #     else:
        #         max_step = None
        #     max_step_output = broadcast_data(['max_step'], {'max_step': max_step}, torch.int64)
        #     max_step = max_step_output['max_step'].item()
        # else:
        #     max_step = len(train_data_loader)



        # max_step = len(train_data_loader)

        header = 'Epoch [{epoch}/{max_epoches}][{step}/{max_step}]'

        if train_data_loader is None:
            data_loader = range(max_step)
        else:
            data_loader = train_data_loader
        wandb_loss = 0
        wandb_loss_num = 0
        for step, batch in enumerate(data_loader):
            if isinstance(batch, int):
                batch = None
            for key in self.lr_scheduler:
                if self.lr_scheduler_by_step[key]:
                    self.lr_scheduler[key].step(epoch*max_step+step)
        
            # record data read time
            data_time.update(time.time() - end_time)
   
            loss = self.train_one_step(batch, step, self.datastd)
            wandb_loss +=loss[self.loss_type]
            wandb_loss_num += 1
            # record loss and time
            metric_logger.update(**loss)
            iter_time.update(time.time() - end_time)
            end_time = time.time()

            # output to logger
            if (step+1) % 100 == 0 or step+1 == max_step:
                eta_seconds = iter_time.global_avg * (max_step - step - 1 + max_step * (max_epoches-epoch-1))
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                self.logger.info(
                    metric_logger.delimiter.join(
                        [header,
                        "lr: {lr}",
                        "eta: {eta}",
                        "time: {time}",
                        "data: {data}",
                        "memory: {memory:.0f}",
                        "{meters}"
                        ]
                    ).format(
                        epoch=epoch+1, max_epoches=max_epoches, step=step+1, max_step=max_step,
                        lr=self.optimizer[list(self.optimizer.keys())[0]].param_groups[0]["lr"],
                        eta=eta_string,
                        time=str(iter_time),
                        data=str(data_time),
                        memory=torch.cuda.memory_reserved() / (1024. * 1024),
                        meters=str(metric_logger)
                    ))
                # begin_time1 = time.time()
                # print("logger output time:", begin_time1-end_time)

       # wandb.log({"train_step_loss":(wandb_loss / wandb_loss_num)})
            # if step+1 == max_step:
            #     raise NotImplementedError('No Metric Exist.')

    def load_checkpoint(self, checkpoint_path, load_model=True, load_optimizer=True, load_scheduler=True, load_epoch=True, load_metric_best=True):
        if utils.get_world_size() > 1 and mpu.get_tensor_model_parallel_world_size() > 1:
            path1, path2 = checkpoint_path.split('.')
            checkpoint_path = f"{path1}_{mpu.get_tensor_model_parallel_rank()}{path2}"
        
               
       
        if self.use_ceph:
            checkpoint_dict = self.checkpoint_ceph.load_checkpoint(checkpoint_path)
            if checkpoint_dict is None:
                self.logger.info("checkpoint is not exist")
                return
        elif os.path.exists(checkpoint_path):
            checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        else:
            self.logger.info("checkpoint is not exist")
            return
        checkpoint_model = checkpoint_dict['model']
        checkpoint_optimizer = checkpoint_dict['optimizer']
        checkpoint_lr_scheduler = checkpoint_dict['lr_scheduler']
        if load_model:
            for key in checkpoint_model:
                new_state_dict = OrderedDict()
                for k, v in checkpoint_model[key].items():
                    if "module" == k[:6]:
                        name = k[7:]
                    else:
                        name = k
                    # if name == "net.pos_embed":
                    #     H, W = 32, 64
                    #     _, L_v, C = v.shape
                    #     _, L_n, C = self.model[key].state_dict()[name].shape
                    #     n = int((L_v // H // W) ** 0.5)
                    #     N = int((L_n // H // W) ** 0.5)
                    #     if N != n:
                    #         v_data = v.reshape(1, n*H, n*W, C).permute(0, 3, 1, 2)
                    #         v_data = F.interpolate(v_data, (N*H, N*W), mode='bilinear', align_corners=True).permute(0, 2, 3, 1).reshape(1, -1, C)
                    #     else:
                    #         v_data = v
                    # else:
                    #     v_data = v
                    new_state_dict[name] = v

                self.model[key].load_state_dict(new_state_dict, strict=False)
                # self.model[key].load_state_dict(checkpoint_model[key])
        if load_optimizer:
            for key in checkpoint_optimizer:
                self.optimizer[key].load_state_dict(checkpoint_optimizer[key])
        if load_scheduler:
            for key in checkpoint_lr_scheduler:
                self.lr_scheduler[key].load_state_dict(checkpoint_lr_scheduler[key])
        if load_epoch:
            self.begin_epoch = checkpoint_dict['epoch']
        if load_metric_best and 'metric_best' in checkpoint_dict:
            self.metric_best = checkpoint_dict['metric_best']
        if 'amp_scaler' in checkpoint_dict:
            self.gscaler.load_state_dict(checkpoint_dict['amp_scaler'])

        # if 'max_logvar' in checkpoint_dict:
        #     self.max_logvar = checkpoint_dict['max_logvar']
        # if 'min_logvar' in checkpoint_dict:
        #     self.min_logvar = checkpoint_dict['min_logvar']


        self.logger.info("last epoch:{epoch}, metric best:{metric_best}".format(epoch=checkpoint_dict['epoch'], metric_best=checkpoint_dict['metric_best'] if 'metric_best' in checkpoint_dict else None))


    def save_checkpoint(self, epoch, checkpoint_savedir, save_type='save_best'): 
        checkpoint_savedir = Path(checkpoint_savedir)
        # checkpoint_path = checkpoint_savedir / '{}'.format('checkpoint_best.pth' \
        #                     if save_type == 'save_best' else 'checkpoint_latest.pth')

    
        # print(save_type, checkpoint_path)

        if (utils.get_world_size() > 1 and mpu.get_tensor_model_parallel_world_size() == 1) or utils.get_world_size() == 1:
            if save_type == "save_best":
                checkpoint_path = checkpoint_savedir / '{}'.format('checkpoint_best.pth')
            else:
                checkpoint_path = checkpoint_savedir / '{}'.format('checkpoint_latest.pth')
        else:
            if save_type == "save_best":
                checkpoint_path = checkpoint_savedir / f'checkpoint_best_{mpu.get_tensor_model_parallel_rank()}.pth'
            else:
                checkpoint_path = checkpoint_savedir / f'checkpoint_latest_{mpu.get_tensor_model_parallel_rank()}.pth'


        if utils.get_world_size() > 1 and mpu.get_data_parallel_rank() == 0:
            if self.use_ceph:
                self.checkpoint_ceph.save_checkpoint(
                    checkpoint_path,
                    {
                    'epoch':            epoch+1,
                    'model':            {key: self.model[key].module.state_dict() for key in self.model},
                    'optimizer':        {key: self.optimizer[key].state_dict() for key in self.optimizer},
                    'lr_scheduler':     {key: self.lr_scheduler[key].state_dict() for key in self.lr_scheduler},
                    'metric_best':      self.metric_best,
                    'amp_scaler':       self.gscaler.state_dict(),
                    # "max_logvar":       self.max_logvar if hasattr(self, 'max_logvar') else None,
                    # "min_logvar":       self.min_logvar if hasattr(self, 'min_logvar') else None,
                    }
                )
            else:
                torch.save(
                    {
                    'epoch':            epoch+1,
                    'model':            {key: self.model[key].module.state_dict() for key in self.model},
                    'optimizer':        {key: self.optimizer[key].state_dict() for key in self.optimizer},
                    'lr_scheduler':     {key: self.lr_scheduler[key].state_dict() for key in self.lr_scheduler},
                    'metric_best':      self.metric_best,
                    'amp_scaler':       self.gscaler.state_dict(),
                    # "max_logvar":       self.max_logvar if hasattr(self, 'max_logvar') else None,
                    # "min_logvar":       self.min_logvar if hasattr(self, 'min_logvar') else None,
                    }, checkpoint_path
                )
        elif utils.get_world_size() == 1:
            if self.use_ceph:
                self.checkpoint_ceph.save_checkpoint(
                    checkpoint_path,
                    {
                    'epoch':            epoch+1,
                    'model':            {key: self.model[key].state_dict() for key in self.model},
                    'optimizer':        {key: self.optimizer[key].state_dict() for key in self.optimizer},
                    'lr_scheduler':     {key: self.lr_scheduler[key].state_dict() for key in self.lr_scheduler},
                    'metric_best':      self.metric_best,
                    'amp_scaler':       self.gscaler.state_dict(),
                    # "max_logvar":       self.max_logvar if hasattr(self, 'max_logvar') else None,
                    # "min_logvar":       self.min_logvar if hasattr(self, 'min_logvar') else None,
                    }
                )
            else:
                torch.save(
                    {
                    'epoch':            epoch+1,
                    'model':            {key: self.model[key].state_dict() for key in self.model},
                    'optimizer':        {key: self.optimizer[key].state_dict() for key in self.optimizer},
                    'lr_scheduler':     {key: self.lr_scheduler[key].state_dict() for key in self.lr_scheduler},
                    'metric_best':      self.metric_best,
                    'amp_scaler':       self.gscaler.state_dict(),
                    # "max_logvar":       self.max_logvar if hasattr(self, 'max_logvar') else None,
                    # "min_logvar":       self.min_logvar if hasattr(self, 'min_logvar') else None,
                    }, checkpoint_path
                )


    def whether_save_best(self, metric_logger):
        metric_now = metric_logger.meters[self.save_best_param].global_avg
        if self.metric_best is None:
            self.metric_best = metric_now
            return True
        if metric_now < self.metric_best:
            self.metric_best = metric_now
            return True
        return False



    def trainer(self, train_data_loader, test_data_loader, max_epoches, checkpoint_savedir=None, save_ceph=False, resume=False):
       
        if train_data_loader is not None:
            data_std = train_data_loader.dataset.get_meanstd()[1]
            if type(data_std) == torch.Tensor:
                data_std = train_data_loader.dataset.get_meanstd()[1].float()
            else:
                data_std = torch.Tensor(train_data_loader.dataset.get_meanstd()[1]).float()
        else:
            data_std = None
        # if is_dist_avail_and_initialized() and mpu.get_tensor_model_parallel_world_size() > 1:
        #     data_std_dict = broadcast_data(['data_std'], {'data_std': data_std}, torch.float32)
        #     data_std = data_std_dict['data_std']
        # self.datastd = data_std.to(self.device)

        self.datastd = data_std.to(self.device)
        if self.loss_type == "Weight_diff_Loss":
            diff_mean, diff_std = train_data_loader.dataset.get_diffmeanstd()
            self.diff_mean = torch.Tensor(diff_mean).float().to(self.device)
            self.diff_std = torch.Tensor(diff_std).float().to(self.device)
        
        # if self.use_noise:
        #     self.noise_weight = torch.Tensor(train_data_loader.dataset.get_noise_weight()).float().to(self.device)
            

        # if type(train_data_loader.dataset.get_meanstd()[1]) == torch.Tensor:
        #     self.datastd = torch.Tensor(train_data_loader.dataset.get_meanstd()[1].float()).to(self.device)
        # else:
        #     self.datastd = torch.Tensor(train_data_loader.dataset.get_meanstd()[1]).float().to(self.device)
        # metric_logger = self.test(test_data_loader, 0)
        if (self.two_step_training and self.checkpoint_path is not None) and (self.begin_epoch == 0):
            metric_logger = self.test(test_data_loader, 0)
        for epoch in range(self.begin_epoch, max_epoches):
            if train_data_loader is not None:
                train_data_loader.sampler.set_epoch(epoch)


            self.train_one_epoch(train_data_loader, epoch, max_epoches)
            # # update lr_scheduler
            # begin_time = time.time()
            if utils.get_world_size() > 1:
                for key in self.model:
                    utils.check_ddp_consistency(self.model[key])
            # torch.save(self.save_data1, f"data/{epoch}_1.pth")
            # torch.save(self.save_data2, f"data/{epoch}_2.pth")
            
            # begin_time1 = time.time()
            # print("lrscheduler time:", begin_time1 - begin_time)

            if self.is_ema:
                self.ema.apply_shadow()

            # test model
            metric_logger = self.test(test_data_loader, epoch)

            # begin_time2 = time.time()
            # print("test time:", begin_time2 - begin_time1)

            
            # save model
            if checkpoint_savedir is not None:
                if self.whether_save_best(metric_logger):
                    self.save_checkpoint(epoch, checkpoint_savedir, save_type='save_best')
                if (epoch + 1) % 1 == 0:
                    self.save_checkpoint(epoch, checkpoint_savedir, save_type='save_latest')
            gc.collect()
            if is_dist_avail_and_initialized():
                torch.distributed.barrier()
            # end_time = time.time()
            # print("save model time", end_time - begin_time2)
            if self.is_ema:
                self.ema.restore()            

    @torch.no_grad()
    def test(self, test_data_loader, epoch):

        metric_logger = utils.MetricLogger(delimiter="  ", sync=True)
        # set model to eval
        for key in self.model:
            self.model[key].eval()

        # if is_dist_avail_and_initialized() and mpu.get_tensor_model_parallel_world_size() > 1:
        #     if test_data_loader is not None:
        #         max_step = torch.tensor(len(test_data_loader))
        #     else:
        #         max_step = None
        #     max_step_output = broadcast_data(['max_step'], {'max_step': max_step}, torch.int64)
        #     max_step = max_step_output['max_step'].item()
        # else:
        #     max_step = len(test_data_loader)

        max_step = get_data_loader_length(test_data_loader)

        if test_data_loader is None:
            data_loader = range(max_step)
        else:
            data_loader = test_data_loader

        wandb_loss = 0
        wandb_loss_num = 0
        # max_step = len(iter(test_data_loader))
        for step, batch in enumerate(data_loader):
            if isinstance(batch, int):
                batch = None
        # for step in range(max_step):
        #     if is_dist_avail_and_initialized() and mpu.get_tensor_model_parallel_rank() > 0:
        #         batch = None
        #     else:
        #         batch = next(iter(test_data_loader))

            loss = self.test_one_step(batch, step)
            wandb_loss += loss[self.loss_type]
            wandb_loss_num += 1
            metric_logger.update(**loss)
        #wandb.log({"test_step_loss": wandb_loss/wandb_loss_num})
        self.logger.info('  '.join(
                [f'Epoch [{epoch + 1}](val stats)',
                 "{meters}"]).format(
                    meters=str(metric_logger)
                 ))


        return metric_logger

    @torch.no_grad()
    def test_final(self, valid_data_loader, predict_length):
        if self.loss_type == "Possloss":
            print(self.max_logvar.max(), self.max_logvar.min())
            print(self.min_logvar.max(), self.min_logvar.min())
            print(torch.mean(self.max_logvar, dim=-1))
            print(torch.mean(self.min_logvar, dim=-1))        

        if self.loss_type == "Weight_diff_Loss":
            diff_mean, diff_std = valid_data_loader.dataset.get_diffmeanstd()
            self.diff_mean = torch.Tensor(diff_mean).float().to(self.device)
            self.diff_std = torch.Tensor(diff_std).float().to(self.device)


        self.valid_data_loader = valid_data_loader
        metric_logger = []
        for i in range(predict_length):
            metric_logger.append(utils.MetricLogger(delimiter="  "))
        # set model to eval
        for key in self.model:
            self.model[key].eval()
        # print(self.max_logvar)
        # print(self.min_logvar)
        # if valid_data_loader is not None:
        #     data_mean, data_std = valid_data_loader.dataset.get_meanstd()

        if torch.distributed.is_initialized() and mpu.get_tensor_model_parallel_world_size() > 1:
            if valid_data_loader is not None:
                data_mean, data_std = valid_data_loader.dataset.get_meanstd()
            else:
                data_std = None
            data_std_output = broadcast_data(['data_std'], {'data_std': data_std}, torch.float32)
            data_std = data_std_output['data_std']
        else:
            data_mean, data_std = valid_data_loader.dataset.get_meanstd()

        
            
        # clim_time_mean_daily = valid_data_loader.dataset.get_clim_daily()
        clim_time_mean_daily = None
        data_std = data_std.to(self.device)
        if utils.get_world_size() > 1:
            rank = mpu.get_data_parallel_rank()
            world_size = mpu.get_data_parallel_world_size()
        else:
            rank = 0
            world_size = 1
        

        if torch.distributed.is_initialized() and mpu.get_tensor_model_parallel_world_size() > 1:
            if valid_data_loader is not None:
                data_set_total_size = valid_data_loader.sampler.total_size
            else:
                data_set_total_size = None
            data_set_total_size_output = broadcast_data(['data_set_total_size'], {'data_set_total_size': data_set_total_size}, torch.int64)
            data_set_total_size = data_set_total_size_output['data_set_total_size']
        else:
            data_set_total_size = valid_data_loader.sampler.total_size


        base_index = rank * (data_set_total_size // world_size)
        # warnings.warn(f"baseindex {data_set_total_size}")
        total_step = get_data_loader_length(valid_data_loader)

        if valid_data_loader is not None:
            data_loader = valid_data_loader
        else:
            data_loader = total_step

        for step, batch in enumerate(data_loader):
            if isinstance(batch, int):
                batch = None
            # batch_len = batch[0].shape[0]
            # index += batch_len

            losses = self.multi_step_predict(batch, clim_time_mean_daily, data_std, step, predict_length, base_index)
            #losses = self.inter_predict(batch, clim_time_mean_daily, data_std, step, predict_length, base_index)

            for i in range(len(losses)):
                metric_logger[i].update(**losses[i])
            # index += batch_len

            self.logger.info("#"*80)
            self.logger.info(step)
            if step % 10 == 0 or step == total_step-1:
                for i in range(predict_length):
                    self.logger.info('  '.join(
                            [f'final valid {i}th step predict (val stats)',
                            "{meters}"]).format(
                                meters=str(metric_logger[i])
                            ))
        if is_dist_avail_and_initialized():
            torch.distributed.barrier()
        return None
  

    @torch.no_grad()
    def test_final_inter(self, valid_data_loader, predict_length, cfgdir, valid_log_name):
        self.cfgdir = cfgdir
        self.valid_log_name = valid_log_name
        if self.loss_type == "Possloss":
            print(self.max_logvar.max(), self.max_logvar.min())
            print(self.min_logvar.max(), self.min_logvar.min())
            print(torch.mean(self.max_logvar, dim=-1))
            print(torch.mean(self.min_logvar, dim=-1))        

        if self.loss_type == "Weight_diff_Loss":
            diff_mean, diff_std = valid_data_loader.dataset.get_diffmeanstd()
            self.diff_mean = torch.Tensor(diff_mean).float().to(self.device)
            self.diff_std = torch.Tensor(diff_std).float().to(self.device)


        self.valid_data_loader = valid_data_loader
        metric_logger = []
        for i in range(predict_length):
            metric_logger.append(utils.MetricLogger(delimiter="  "))
        # set model to eval
        for key in self.model:
            self.model[key].eval()
        # print(self.max_logvar)
        # print(self.min_logvar)
        # if valid_data_loader is not None:
        #     data_mean, data_std = valid_data_loader.dataset.get_meanstd()

        if torch.distributed.is_initialized() and mpu.get_tensor_model_parallel_world_size() > 1:
            if valid_data_loader is not None:
                data_mean, data_std = valid_data_loader.dataset.get_meanstd()
            else:
                data_std = None
            data_std_output = broadcast_data(['data_std'], {'data_std': data_std}, torch.float32)
            data_std = data_std_output['data_std']
        else:
            data_mean, data_std = valid_data_loader.dataset.get_meanstd()

        
            
        # clim_time_mean_daily = valid_data_loader.dataset.get_clim_daily()
        clim_time_mean_daily = None
        data_std = data_std.to(self.device)
        if utils.get_world_size() > 1:
            rank = mpu.get_data_parallel_rank()
            world_size = mpu.get_data_parallel_world_size()
        else:
            rank = 0
            world_size = 1
        

        if torch.distributed.is_initialized() and mpu.get_tensor_model_parallel_world_size() > 1:
            if valid_data_loader is not None:
                data_set_total_size = valid_data_loader.sampler.total_size
            else:
                data_set_total_size = None
            data_set_total_size_output = broadcast_data(['data_set_total_size'], {'data_set_total_size': data_set_total_size}, torch.int64)
            data_set_total_size = data_set_total_size_output['data_set_total_size']
        else:
            data_set_total_size = valid_data_loader.sampler.total_size


        base_index = rank * (data_set_total_size // world_size)
        # warnings.warn(f"baseindex {data_set_total_size}")
        total_step = get_data_loader_length(valid_data_loader)

        if valid_data_loader is not None:
            data_loader = valid_data_loader
        else:
            data_loader = total_step





        for step, batch in enumerate(data_loader):
            if isinstance(batch, int):
                batch = None
            # batch_len = batch[0].shape[0]
            # index += batch_len

            #losses = self.multi_step_predict(batch, clim_time_mean_daily, data_std, step, predict_length, base_index)
            losses = self.inter_predict(batch, clim_time_mean_daily, data_mean, data_std, step, predict_length, base_index)

            for i in range(len(losses)):
                metric_logger[i].update(**losses[i])
            # index += batch_len

            self.logger.info("#"*80)
            self.logger.info(step)
            if step % 10 == 0 or step == total_step-1:
                for i in range(predict_length):
                    self.logger.info('  '.join(
                            [f'final valid {i}th step predict (val stats)',
                            "{meters}"]).format(
                                meters=str(metric_logger[i])
                            ))
        if is_dist_avail_and_initialized():
            torch.distributed.barrier()
        return None


