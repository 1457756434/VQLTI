import torch
from models.model import basemodel
import torch.cuda.amp as amp
import csv
import time
import wandb
import pandas as pd
import os
import torch.nn.functional as F
import numpy as np

import json
def to_numpy(tensor):
    return tensor.detach().cpu().numpy().astype(np.float32) if tensor.requires_grad else tensor.cpu().numpy().astype(np.float32)
class Iter_model(basemodel):
    def __init__(self, logger, **params) -> None:
        super().__init__(logger, **params)
        
        self.use_MKE = self.extra_params.get("use_MKE", False)
        self.MKE_rate = self.extra_params.get("MKE_rate", 0.3)
        self.logger_print_time = False

        self.data_begin_time = time.time()
        sub_model = self.params.get('sub_model', {})
        for key in sub_model:
            self.model_type = key

        self.vq_state = None
        if self.is_save_vq_state:
            self.vq_num = 0
            self.vq_state = None

  
    def save_vq_state(self, vq_state, name):
        if self.vq_state == None:
            self.vq_state = vq_state
            self.vq_num  += 1
        else:
            self.vq_state = torch.cat((self.vq_state , vq_state), dim=0)
            self.vq_num  += 1
        if self.vq_num==50:
            np_vq = self.vq_state.cpu().numpy()
            save_path = "save_vq_demo"
            np.save(f"{save_path}/{name}.npy", np_vq)

    def save_vq_state_dic(self, wind, vq_state, name):
        vq_state = vq_state.cpu().numpy()
        
        wind =  wind.cpu().numpy()
        save_path = "save_vq_demo"
        if self.vq_state == None:
        
            self.vq_state = {str(wind):vq_state.tolist()}

            with open(f"{save_path}/{name}.json", 'w') as file:
                json.dump(self.vq_state, file)
        else:
            
            with open(f"{save_path}/{name}.json", 'r') as file:
                self.vq_state = json.load(file)
            if str(wind) in self.vq_state.keys():
                vq_state_wind_np = np.array(self.vq_state[str(wind)])
                
                self.vq_state[str(wind)] = np.concatenate((vq_state_wind_np, vq_state.tolist()), axis=0).tolist()
            else:
                self.vq_state[str(wind)] = vq_state.tolist()
            
            with open(f"{save_path}/{name}.json", 'w') as file:
                json.dump(self.vq_state, file)

    def data_preprocess(self, data):
        #print(len(data))
        # begin_time = time.time()
       
        inp_lable=0
        label=0
        inp_last_tc_day_info = None
        weight = None
        add_data = None
        
        if len(data)==3:
            
            new_era5_data = data[0].float().to(self.device, non_blocking=True)
            inp_lable = data[-2].float().to(self.device, non_blocking=True)
            label = data[-1].float().to(self.device, non_blocking=True)


        elif len(data)==4:
            
            new_era5_data = data[0].float().to(self.device, non_blocking=True)
            inp_lable = data[-3].float().to(self.device, non_blocking=True)
            label = data[-2].float().to(self.device, non_blocking=True)
            weight = data[-1].float().to(self.device, non_blocking=True)
        elif len(data)==6:
           
            new_era5_data =         data[0].float().to(self.device, non_blocking=True)
            inp_lable =             data[1].float().to(self.device, non_blocking=True)
            label  =                data[2].float().to(self.device, non_blocking=True) 
            inp_last_tc_day_info =  data[3]
            weight =                data[4].float().to(self.device, non_blocking=True) 
            add_data =              data[5].float().to(self.device, non_blocking=True) 
        else:
            raise ValueError("data size error")
     
        return new_era5_data, inp_lable, label, inp_last_tc_day_info, weight, add_data






    def train_one_step(self, batch_data, step, data_std):
        if self.logger_print_time:
            self.logger.info(f"others time:{time.time() - self.data_begin_time}")
        begin_time = time.time()
        
        new_era5_data, inp_lable, label, inp_last_tc_day_info, weight, add_data = self.data_preprocess(batch_data)
        
   
        if self.logger_print_time:
            self.logger.info(f"data preprocess time:{time.time() - begin_time}")
        begin_time = time.time()

        self.optimizer[list(self.model.keys())[0]].zero_grad()

        with amp.autocast(enabled=self.enabled_amp):
            if (self.model_type == "VQ_TC_ERA5"):
                predict, recons_loss, vq_loss, quantized_hidden_state = self.model[list(self.model.keys())[0]](new_era5_data, inp_lable, label)
            elif (self.model_type == "FengWu_TC"):
                if self.cmp_mode=="Many_to_Many":
                    input_len = self.inp_len
                    fengwu_pre = new_era5_data[:, input_len:]
                    new_era5_data = new_era5_data[:, :input_len]
                
                    pre_lable = None
                    if self.is_hid_vq:
                     
                        pre_lable = inp_lable[:, input_len:]
                        inp_lable = inp_lable[:, :input_len]
                    pre_len = fengwu_pre.shape[1]
                    if (self.model_type == "FengWu_TC"):
                        predict, hid_vq_loss,quantized_hidden_state = self.model[list(self.model.keys())[0]](new_era5_data, inp_lable, fengwu_pre, pre_len, add_data, 
                                                                                  pre_lable, is_train_mode=True)
                
             
            if self.logger_print_time:
                self.logger.info(f"model time:{time.time() - begin_time}")
            begin_time = time.time()
            # if self.loss_type == "Weight_diff_Loss":
            #     target1 = tar_step1 - inp[:,self.constants_len:]
            # else:
            #     target1 = tar_step1
            
            if (self.loss_type == "weighted_mse_loss") or (self.loss_type == "weighted_mae_loss"):
       
                step_one_loss = self.loss(predict, label, weights=weight)
            elif (self.model_type == "VQ_TC_ERA5"):
                step_one_loss = recons_loss + vq_loss
            elif (self.loss_type == "diff2_mae_loss"):
                step_one_loss = self.loss(self.model[list(self.model.keys())[0]] ,predict, label)
            
            else:
                step_one_loss = self.loss(predict, label)
            if self.is_hid_vq:
                step_one_loss = step_one_loss + hid_vq_loss


            if self.is_add_RI_Loss:
                addition_loss = self.addition_Loss(inp_lable, predict, label, data_std, self.RI_threshold)
                step_one_loss = step_one_loss * addition_loss.detach()
        # import pdb
        # pdb.set_trace()
        

        self.gscaler.scale(step_one_loss).backward()
        



        step_two_loss = None
        with amp.autocast(enabled=self.enabled_amp):
        
            loss = step_one_loss


        # loss.backward()
        # self.optimizer[list(self.model.keys())[0]].step()
        if step_two_loss is not None:
            self.gscaler.scale(step_two_loss).backward()
        # torch.nn.utils.clip_grad_value_(self.model[list(self.model.keys())[0]].parameters(), 0.5)
        
        if self.logger_print_time:
            self.logger.info(f"loss backward time:{time.time() - begin_time}")
        begin_time = time.time()
        self.gscaler.step(self.optimizer[list(self.model.keys())[0]])
        self.gscaler.update()
        
        if self.is_ema:
            self.ema.update()
        # after_loss_scale = self.gscaler.get_scale()
        # loss.backward()
        # self.optimizer['swinunet'].step()
        if self.logger_print_time:
            self.logger.info(f"optimizer time:{time.time() - begin_time}")
        self.data_begin_time = time.time()



        if (self.model_type == "VQ_TC_ERA5"):
            return {self.loss_type: loss.item(), "loss": step_one_loss.item(), "recons_loss":recons_loss.item(), "vq_loss":vq_loss.item()}
        elif self.is_hid_vq:
            return {self.loss_type: loss.item(), "loss": step_one_loss.item(), "hid_vq_loss":hid_vq_loss.item()}
        else:
            return {self.loss_type: loss.item(), "loss": step_one_loss.item()}

    def test_one_step(self, batch_data, step):
        new_era5_data, inp_lable, label, inp_last_tc_day_info, weight, add_data = self.data_preprocess(batch_data)

        if (self.model_type == "VQ_TC_ERA5"):
            predict, recons_loss, vq_loss, quantized_hidden_state = self.model[list(self.model.keys())[0]](new_era5_data, inp_lable, label)

        elif (self.model_type == "FengWu_TC"):
            if self.cmp_mode=="Many_to_Many":
                input_len = self.inp_len
                fengwu_pre = new_era5_data[:, input_len:]
                new_era5_data = new_era5_data[:, :input_len]
                pre_len = fengwu_pre.shape[1]
                
                pre_lable = None
                if self.is_hid_vq:

                    pre_lable = inp_lable[:, input_len:]
                    inp_lable = inp_lable[:, :input_len]

                if (self.model_type == "FengWu_TC"):
                    predict, hid_vq_loss,quantized_hidden_state = self.model[list(self.model.keys())[0]](new_era5_data, inp_lable, fengwu_pre, pre_len, add_data)
               
        else:
            predict = self.model[list(self.model.keys())[0]](new_era5_data)


        if (self.loss_type == "weighted_mse_loss") or (self.loss_type == "weighted_mae_loss"):
            step_one_loss = self.loss(predict, label, weights=weight)
        elif (self.model_type == "VQ_TC_ERA5"):
            step_one_loss = recons_loss + vq_loss
        elif (self.loss_type == "diff2_mae_loss"):
            step_one_loss = self.loss( self.model[list(self.model.keys())[0]],predict, label)
        
        else:
            step_one_loss = self.loss(predict, label)
       
       
        if (self.loss_type == "Possloss") or (self.loss == "NLgpossloss"):
            predict, log_var = predict.chunk(2, dim = 1)
        
        loss = step_one_loss
        
        data_dict = {}
        data_dict['gt'] = label
        data_dict['pred'] = predict
        metrics_loss = self.eval_metrics.evaluate_batch(data_dict)

        if (self.model_type == "VQ_TC_ERA5"):
            metrics_loss.update({self.loss_type: loss.item(), "loss": step_one_loss.item(), "recons_loss":recons_loss.item(), "vq_loss":vq_loss.item()})
        else:
            metrics_loss.update({self.loss_type: loss.item(), "loss": step_one_loss.item()})
        
        return metrics_loss


    def inter_predict(self, batch_data, clim_time_mean_daily, data_mean, data_std, step, predict_length, base_index):
       
        metrics_losses = []

        new_era5_data, inp_lable, label, inp_last_tc_day_info, weight, add_data = self.data_preprocess(batch_data)
       
        if (self.model_type == "VQ_TC_ERA5"):
            predict, recons_loss, vq_loss, quantized_hidden_state = self.model[list(self.model.keys())[0]](new_era5_data, inp_lable, label)
            if self.is_save_vq_state:
                
                self.save_vq_state(quantized_hidden_state, self.vq_name)
        elif (self.model_type == "FengWu_TC"):
            if self.cmp_mode=="Many_to_Many":
                input_len = self.inp_len
                fengwu_pre = new_era5_data[:, input_len:]
                new_era5_data = new_era5_data[:, :input_len]
                pre_len = fengwu_pre.shape[1]
              
                pre_lable = None
                if self.is_hid_vq:
               
                    pre_lable = inp_lable[:, input_len:]
                    inp_lable = inp_lable[:, :input_len]
                if (self.model_type == "FengWu_TC"):
                    
                    predict, hid_vq_loss,quantized_hidden_state = self.model[list(self.model.keys())[0]](new_era5_data, inp_lable, fengwu_pre, pre_len, add_data)
              
                
                if self.is_save_vq_state:
                    MSW_0 = inp_lable[:, 0, 0] * data_std[0] + data_mean[0]
    
                    quantized_hidden_state = torch.unsqueeze(quantized_hidden_state[0], dim=0)
                    self.save_vq_state_dic(MSW_0, quantized_hidden_state, self.vq_name)

        else:
            predict = self.model[list(self.model.keys())[0]](new_era5_data)

    
        if self.is_diff:
            label[:, 0, :] = label[:, 0, :] + inp_lable[:, -1, :]
            for i in range(1, label.shape[1]):
                label[:, i, :] = label[:, i, :] + label[:, i-1, :]

            predict[:, 0, :] = predict[:, 0, :] + inp_lable[:, -1, :]
            for i in range(1, predict.shape[1]):
                predict[:, i, :] = predict[:, i, :] + predict[:, i-1, :]
        if (self.model_type == "VQ_TC_ERA5"):
            data_dict = {}
            data_dict['gt'] = label * data_std
            data_dict['pred'] = predict * data_std
            data_dict['clim_mean'] = None
            data_dict['std'] = data_std
            metrics_losses.append(self.eval_metrics.evaluate_batch(data_dict))
        else:
            for i in range(predict_length):
                
                data_dict = {}
                data_dict['gt'] = label[:, i, :2] * data_std
                data_dict['pred'] = predict[:, i, :2] * data_std
                data_dict['clim_mean'] = None
                data_dict['std'] = data_std
                metrics_losses.append(self.eval_metrics.evaluate_batch(data_dict))



        if (self.model_type == "VQ_TC_ERA5"):
            pass
     
        else:
            all_year = inp_last_tc_day_info["ALL_Year"]
            basin = self.valid_log_name.split('.')[0]
            path = os.path.join(self.cfgdir, f"Pre_{basin}_{all_year[0][0].item()}_{all_year[-1][0].item()}.csv")

            csv_path = path
            df = pd.read_csv(csv_path)
            for j in range(predict.shape[0]):

                for i in range(predict.shape[1]):
                    wind = predict[j, i, 0] * data_std[0] + data_mean[0]
                    
                    sid = inp_last_tc_day_info["SID"][j]
                    iso = inp_last_tc_day_info["ISO_TIME"][j]
                    h = (i+1)*self.output_step_length*6
                    wind_name_pre = f"USA_WIND_PRE_{h}h"
                    df.loc[(df["ISO_TIME"]==iso)&(df["SID"]==sid), (wind_name_pre)] = wind.item()

                    if predict.shape[2]==2:
                        mslp = predict[j, i, 1] * data_std[1] + data_mean[1]
                        mslp_name_pre = f"USA_PRES_PRE_{h}h"
                        df.loc[(df["ISO_TIME"]==iso)&(df["SID"]==sid), (mslp_name_pre)] = mslp.item()
                    elif predict.shape[2]==4:
                        mslp = predict[j, i, 1] * data_std[1] + data_mean[1]
                        mslp_name_pre = f"USA_PRES_PRE_{h}h"
                        df.loc[(df["ISO_TIME"]==iso)&(df["SID"]==sid), (mslp_name_pre)] = mslp.item()

                        lat = predict[j, i, 2] * 90
                        lat_pre = f"USA_LAT_PRE_{h}h"
                        df.loc[(df["ISO_TIME"]==iso)&(df["SID"]==sid), (lat_pre)] = lat.item()
                        lon = predict[j, i, 3] * 180
                        lon_pre = f"USA_LON_PRE_{h}h"
                        df.loc[(df["ISO_TIME"]==iso)&(df["SID"]==sid), (lon_pre)] = lon.item()
                
            df.to_csv(csv_path, sep=',', index=False, header=True)


        return metrics_losses