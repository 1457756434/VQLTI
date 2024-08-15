# original data
import math
import xarray as xr
import more_itertools
from torch.utils.data import Dataset
from petrel_client.client import Client
import csv
import numpy as np
import io
import time

import json
import pandas as pd
import os
import copy
import queue
import torch
import torchvision.transforms as transforms

from datetime import datetime
from datetime import timedelta


import onnx
import onnxruntime as ort

import metpy.calc as mpcalc

from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d
import torchvision.transforms.functional as TF
import random


Years = {
    'train': range(2011, 2018),
    'valid': range(2019, 2021),
    'test': range(2018, 2019),
    'all': range(2011, 2021)
}

multi_level_vnames = [
    "z", "t", "q", "r", "u", "v", "vo", "pv",
]
single_level_vnames = [
    "t2m", "u10", "v10", "tcc", "tp", "tisr",
]
long2shortname_dict = {"geopotential": "z", "temperature": "t", "specific_humidity": "q", "relative_humidity": "r", "u_component_of_wind": "u", "v_component_of_wind": "v", "vorticity": "vo", "potential_vorticity": "pv", \
    "2m_temperature": "t2m", "10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10", "total_cloud_cover": "tcc", "total_precipitation": "tp", "toa_incident_solar_radiation": "tisr"}
constants = [
    "lsm", "slt", "orography"
]
height_level = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, \
    500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
# height_level = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

multi_level_dict_param = {"z":height_level, "t": height_level, "q": height_level, "r": height_level}

from typing import Sequence

class MyRotateTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

class GRIDSAT_crop_dataset_fengwu_pre(Dataset):
    def __init__(self, data_dir='cephnew:s3://tropical_cyclone_data/GRIDSAT/npy', split='train', **kwargs) -> None:
        super().__init__()
        #print("init begin")
        self.data_dir = data_dir
        self.split = split
        self.data_dir = data_dir
        self.client = Client(conf_path="~/petreloss.conf")

        self.IBTrACS_url = kwargs.get('IBTrACS_url', 'dataset/ibtracs.ALL.list.v04r00.csv')

        self.save_meanstd_dir = kwargs.get('save_meanstd_dir', 'dataset/new_meanstd')

        self.valid_log_name = kwargs.get('valid_log_name', "all_basin")
        self.cfgdir = kwargs.get('cfgdir', "TC_Pre_base_FengWu")

        self.img_data_nan_rate_threshold = kwargs.get('img_data_nan_rate_threshold', 0.01)

        self.train_begin_year = kwargs.get('train_begin_year', 2011)
        self.train_end_year = kwargs.get('train_end_year', 2017)
        self.valid_begin_year = kwargs.get('valid_begin_year', 2019)
        self.valid_end_year = kwargs.get('valid_end_year', 2020)
        self.test_begin_year = kwargs.get('test_begin_year', 2018)
        self.test_end_year = kwargs.get('test_end_year', 2018)
        Years = {
            'train': range(self.train_begin_year, self.train_end_year+1),
            'valid': range(self.valid_begin_year, self.valid_end_year+1),
            'test' : range(self.test_begin_year, self.test_end_year + 1),
            'all': range(2011, 2021)
        }

        self.all_year = Years[split]


        self.input_length = kwargs.get('input_length', 4)
        self.output_length = kwargs.get('output_length', 4)


        self.is_use_singel_msw = kwargs.get("is_use_singel_msw", False)
        if self.is_use_singel_msw:
            self.msw_mslp_choose = kwargs.get("msw_mslp_choose", [20])
            self.mswname = kwargs.get("mswname", "MSW20")


        self.is_fengwu_pre = kwargs.get('is_fengwu_pre', False)
        self.is_use_fengwu = kwargs.get("is_use_fengwu", False)
     
        self.is_use_fengwu_crop = kwargs.get("is_use_fengwu_crop", False)
        self.fengwu_crop_base_dir = kwargs.get("fengwu_crop_base_dir", "cephnew:s3://tropical_cyclone_data/Fengwu_v1_pre_crop")
        self.is_cmp_origin_field_intensity = kwargs.get("is_cmp_origin_field_intensity", False)



        self.output_step_length = kwargs.get('output_step_length', 1)

        
        self.ERA5_image_size = kwargs.get('ERA5_image_size', 40)
        
        self.window_size = self.input_length + self.output_length

        self.resolution = kwargs.get('resolution', 0.25)
        
        self.resolution = 1 / self.resolution
       
        self.radius = kwargs.get('radius', 5)
        self.radius_np = int(self.radius * self.resolution)
        Years_dict = kwargs.get('years', Years)
        self.is_map_inp_intensity = kwargs.get('is_map_inp_intensity', False)

        self.is_save_npy = kwargs.get('is_save_npy', False)
        self.is_load_npy = kwargs.get('is_load_npy', True)
        self.is_use_lifetime_num = kwargs.get("is_use_lifetime_num", False)
        self.is_diff = kwargs.get("is_diff", False)

        self.inp_type = kwargs.get('inp_type', ["ERA5", "Seq"])

        self.set_ERA5_zero = kwargs.get('set_ERA5_zero', False)
        self.set_Seq_zero = kwargs.get('set_Seq_zero', False)

        self.set_PI_zero = kwargs.get('set_PI_zero', False)
        self.set_fengwu_zero = kwargs.get('set_fengwu_zero', False)

        vnames_type = kwargs.get("vnames", {})
        self.constants_types = vnames_type.get('constants', [])
        self.ERA5_vnames_dic = self.get_ERA5_dic()
     
        self.single_level_vnames = vnames_type.get('single_level_vnames', ['u10', 'v10', 't2m', 'msl'])
        self.multi_level_vnames = vnames_type.get('multi_level_vnames', ['z', 'q', 'u', 'v', 't'])
        self.height_level_list = vnames_type.get('hight_level_list', [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])

        self.height_level_indexes = [height_level.index(j) for j in self.height_level_list]


        self.label_vnames = vnames_type.get('label_vnames', ["USA_WIND", "USA_PRES"])
        self.TCIR_vnames = vnames_type.get('TCIR_vnames', ["Vmax", "MSLP"])
        self.is_pre_latlon = vnames_type.get('is_pre_latlon', False)
        if self.is_pre_latlon:
            self.latlon_scale = vnames_type.get('latlon_scale', [90, 180])
        

        
        self.train_label_Basin = vnames_type.get('train_label_Basin', ["EP", "NA", "NI", "SA", "SI", "SP", "WP"])
        self.test_label_Basin = vnames_type.get('test_label_Basin', ["EP", "NA", "NI", "SA", "SI", "SP", "WP"])



        if len(self.constants_types) > 0:
            self.constants_data = self.get_constants_data(self.constants_types)
        else:
            self.constants_data = None
        
        
        
        self.intensity_mean, self.intensity_std = self.get_intensity_meanstd()
        self.era5_mean, self.era5_std = self.get_era5_crop_meanstd()
   

        self.data_element_num = len(self.single_level_vnames) + len(self.multi_level_vnames) * len(self.height_level_list)

     
        years = Years_dict[split]

        self.is_train_fengwu_crop = kwargs.get('is_train_fengwu_crop', False)
        if (split=="train") and self.is_train_fengwu_crop:
            self.train_fengwu_crop_year_list = kwargs.get('train_fengwu_crop_year_list', [2021])
            from itertools import chain
            years = chain(years, self.train_fengwu_crop_year_list)

        if split == "valid":
            self.input_day_list, self.label_day_list, self.input_intensity_final, self.label_intensity_final, self.input_latlon_final, self.label_latlon_final = self.init_file_list_save_pre(years)
        else:
            self.input_day_list, self.label_day_list, self.input_intensity_final, self.label_intensity_final, self.input_latlon_final, self.label_latlon_final = self.init_file_list(years)

        self.era5_inp_url = []
       
        
        self.is_hid_vq = kwargs.get('is_hid_vq', False)
        self.is_add_data = kwargs.get('is_add_data', False)
        if self.is_add_data:
            self.add_data_inp_url = [] 
            self.add_config = kwargs.get('add_config', \
                #"cephnew:s3://tropical_cyclone_data/npy_fengwu_era5_PI"                   "cephnew:s3://tropical_cyclone_data/Fengwu_pretrain_v2_pre_crop_PI"
                {"add_data_path":"cephnew:s3://tropical_cyclone_data/npy_fengwu_era5_PI"
                 }
                 )
            self.add_data_path = self.add_config["add_data_path"]

        for i in range(len(self.input_day_list)):
            input_window = self.input_day_list[i]
            output_window = self.label_day_list[i]
            urls = []
        
            if self.is_add_data:
                urls_add_data = []
            for day in input_window:
                url = self.url_to_era5(day)
                
                urls.append(url)
    
                end_inp_day = day
            if self.is_fengwu_pre:
                
                for i,day in enumerate(output_window):
                  
                    url = self.url_to_era5(day)
                    if self.is_add_data:
                        if self.is_use_fengwu:
                            lead_time = (i+1)*6
                            url_add_data = self.url_to_add_data_fengwuPre(lead_time, end_inp_day)
                        else:
                            url_add_data = self.url_to_add_data(day)
                        urls_add_data.append(url_add_data)
                    urls.append(url)
                    
            self.era5_inp_url.append(urls)
          
            if self.is_add_data:
                self.add_data_inp_url.append(urls_add_data)
        self.era5_inp_url = np.array(self.era5_inp_url)
       
        if self.is_add_data:
            self.add_data_inp_url = np.array(self.add_data_inp_url)
        self.is_rand_rotation = kwargs.get('is_rand_rotation', False)
        

        self.ERA5_transform =transforms.Compose([
            transforms.CenterCrop(self.ERA5_image_size),
            ])




        self.len_file = len(self.input_day_list)
        print("dataset length:{}".format(self.len_file))
        



        if self.is_use_fengwu_crop:
            self.is_use_fengwu = True
   

        if self.is_use_fengwu:
            
            self.vo_region = [89, -89, 0, 360]
            self.radius_mslp = 4.5
            self.cmp_wind = True
            self.cmp_thickness =True
            self.cmp_lsm = True
            
            

            self.init_resulation = 0.25
            self.is_wait_genesis = False
            self.empty_time_threshold = 1
            with io.BytesIO(self.client.get("cephnew:s3://myBucket/dxdy/msl.nc", update_cache=True)) as f:
                single_levels_data = xr.open_dataset(f)
            to_get_dxy = single_levels_data.msl.loc[:, self.vo_region[0]:self.vo_region[1], self.vo_region[2]:self.vo_region[3]]
            self.dx, self.dy = mpcalc.lat_lon_grid_deltas(to_get_dxy.longitude, to_get_dxy.latitude)
            self.all_vname_index = self.get_all_vname_index()
            self.fengwu_mean_std_mutil_json, self.fengwu_mean_std_single_json, self.fengwu_mean, self.fengwu_std  = self.get_fengwu_meanstd_json()




        
    def get_fengwu_meanstd_json(self):
        with open(f'{self.save_meanstd_dir}/mean_std.json') as user_file:
            multi_level_mean_std = json.load(user_file)
        with open(f'{self.save_meanstd_dir}/mean_std_single.json') as user_file:
            single_level_mean_std = json.load(user_file)

        single_level_vnames = ['u10', 'v10', 't2m', 'msl']
        multi_level_vnames = ['z', 'q', 'u', 'v', 't']
        height_level_list = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        height_level = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, \
        500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
        height_level_indexes = [height_level.index(j) for j in height_level_list]


        mean_std = {}
        multi_level_mean_std['mean'].update(single_level_mean_std['mean'])
        multi_level_mean_std['std'].update(single_level_mean_std['std'])
        mean_std['mean'] = multi_level_mean_std['mean']
        mean_std['std'] = multi_level_mean_std['std']
        for vname in single_level_vnames:
            mean_std['mean'][vname] = np.array(mean_std['mean'][vname])[::-1][:,np.newaxis,np.newaxis]
            mean_std['std'][vname] = np.array(mean_std['std'][vname])[::-1][:,np.newaxis,np.newaxis]
        for vname in multi_level_vnames:
            mean_std['mean'][vname] = np.array(mean_std['mean'][vname])[::-1][:,np.newaxis,np.newaxis]
            mean_std['std'][vname] = np.array(mean_std['std'][vname])[::-1][:,np.newaxis,np.newaxis]




        return_data_mean = []
        return_data_std = []
        
        for vname in single_level_vnames:
            return_data_mean.append(mean_std['mean'][vname])
            return_data_std.append(mean_std['std'][vname])
        for vname in self.multi_level_vnames:
            return_data_mean.append(mean_std['mean'][vname][height_level_indexes])
            return_data_std.append(mean_std['std'][vname][height_level_indexes])

        return multi_level_mean_std, single_level_mean_std, \
            np.concatenate(return_data_mean, axis=0)[:, 0, 0], np.concatenate(return_data_std, axis=0)[:, 0, 0]



    def get_all_vname_index(self):

        single_level_vnames = ['u10', 'v10', 't2m', 'msl']
        multi_level_vnames = ['z', 'q', 'u', 'v', 't']

        height_level_list = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        index = 0
        all_vname_index = {}
        for vname in single_level_vnames:
            all_vname_index[vname] = index
            index += 1 
        for vname in multi_level_vnames:
            for height in height_level_list:
                all_vname_index[f"{vname}_{height}"] = index
                index += 1
        return all_vname_index


    def get_sid_isotime(self, ir_list, ir_nan_data):
        for i in range(len(ir_nan_data)):
            tc_name = ir_nan_data["SID"].iloc[i]
            tc_ISO_TIME = ir_nan_data["ISO_TIME"].iloc[i]
            ir_list.append(str(tc_name)+"_"+str(tc_ISO_TIME))
        return ir_list

    

    def get_ERA5_dic(self):
        ERA5_vnames_dic = {}
        if self.data_dir=='cephnew:s3://tropical_cyclone_data/GRIDSAT/npy':
            single_level_vnames = ["u10", "v10", "t2m", "sp", "msl"]
            multi_level_vnames = ["z", "t", "q", "r", "u", "v"]
            height_level_list = [50, 150, 200, 300, 350, 500, 550, 700, 750, 850, 950]
        else:
            single_level_vnames = ['u10', 'v10', 't2m', 'msl']
            multi_level_vnames = ['z', 'q', 'u', 'v', 't']
            height_level_list = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        index = 0
        for vname in single_level_vnames:
            ERA5_vnames_dic[vname] = index
            index += 1
        for vname in multi_level_vnames:
            ERA5_vnames_dic[vname] = {}
            for height in height_level_list:
                ERA5_vnames_dic[vname][height] = index
                index += 1
        return ERA5_vnames_dic

    def init_file_list(self, years):
        IBTrACS_url = self.IBTrACS_url
        
        csv_data = pd.read_csv(IBTrACS_url, encoding="utf-8", low_memory=False, )
        input_final = []
        label_final = []
        input_intensity_final = []
        label_intensity_final = []
        input_latlon_final = []
        label_latlon_final = []
        lifetime_num_dic = {}
        all_tc_len = 0
        if self.split == "train" or self.split == "test":
            label_Basin = self.train_label_Basin
        else:
            label_Basin = self.test_label_Basin

        for year in years:
            
            year_data = csv_data.loc[(csv_data["SEASON"]==str(year))]
            
            all_year_tc = []

            #print(len(year_data))
            for i in range(len(year_data)):
                tc_name = year_data["SID"].iloc[i]
                tc_Basin = year_data["BASIN"].iloc[i]
                
                if pd.isna(tc_Basin):
                    tc_Basin="NA"
                if (tc_name not in all_year_tc) and (tc_Basin in label_Basin):
                    all_year_tc.append(tc_name)
                # if (tc_Basin not in label_Basin):
                #     print(tc_name, tc_Basin)
      
            print(len(all_year_tc))
            all_tc_len = all_tc_len + len(all_year_tc)

            # import pdb
            # pdb.set_trace()
            for tc in all_year_tc:
                if self.is_use_lifetime_num:
                    
                    lifetime_num = -1
                tc_data = year_data.loc[(year_data["SID"]==tc)]
                len_tc = len(tc_data)
            
                if len_tc>=self.window_size:
                    time_need = []
                    label_need = []
                    for i in range(len_tc):
                        iso_time = tc_data["ISO_TIME"].iloc[i]
                        if iso_time[11:] in ["00:00:00", "06:00:00", "12:00:00", "18:00:00"]:
                            if self.is_use_lifetime_num:
                                lifetime_num = lifetime_num + 1
                            day_data = tc_data.loc[(tc_data["ISO_TIME"]==iso_time)]
                            day_label = []
                            flag = True
                            
                            for vname in self.label_vnames:
                                data_vname = np.array(day_data[vname])[0] 
                                #print(data_vname)
                                if data_vname != ' ':
                                    data_vname = float(data_vname)
                                    day_label.append(data_vname)
                                else:
                                    flag = False
                                    break
                            if flag:
                                #print(day_label)
                                iso_time_path = os.path.join(str(year), tc, iso_time)
                                time_need.append(iso_time_path)
                                label_need.append(day_label)
                                if self.is_use_lifetime_num:
                                    lifetime_num_dic[str(iso_time_path)] = lifetime_num
                    #print(time_need)
                    input_start = 0
                    input_end   = 0
                    label_start = 0
                    label_end   = 0
                    for j in range(len_tc-self.window_size):
                        
                        input_start = j
                        input_end   = input_start + self.input_length
                        label_start = input_end
                        label_end   = label_start + self.output_length

                        input_day = time_need[input_start:input_end]
                        label_day = time_need[label_start:label_end]

                        window_day = input_day + label_day
                        if len(window_day)!= self.window_size:
                            break
                        if self.check_window(window_day):
                            input_final.append(input_day)
                            label_final.append(label_day)
                            input_intensity = []
                            input_latlon = []
                            label_latlon = []
                            for time_url in input_day:
                                in_time = time_url.split('/')[-1]
                                inp_data = tc_data.loc[(tc_data["ISO_TIME"]==in_time)]
                                day_label = []
                                for vname in self.label_vnames:
                                    data_vname = np.array(inp_data[vname])[0]
                                    data_vname = float(data_vname)
                                    day_label.append(data_vname)
                                if self.is_use_lifetime_num:
                                    day_label.append(lifetime_num_dic[str(time_url)])
                                input_intensity.append(day_label)
                                

                                lat = float(np.array(inp_data["LAT"])[0])
                                lon = float(np.array(inp_data["LON"])[0])
                                input_latlon.append([lat, lon])

                            label_intensity = []
                            
                            for lab_time in label_day:
                                lab_time = lab_time.split('/')[-1]
                                lab_data = tc_data.loc[(tc_data["ISO_TIME"]==lab_time)]
                                day_label = []
                                for vname in self.label_vnames:
                                    data_vname = np.array(lab_data[vname])[0]
                                    data_vname = float(data_vname)
                                    day_label.append(data_vname)
                                lat = float(np.array(lab_data["LAT"])[0])
                                lon = float(np.array(lab_data["LON"])[0])
                                label_latlon.append([lat, lon])
                                
                                label_intensity.append(day_label)
                            input_latlon_final.append(input_latlon)
                            label_latlon_final.append(label_latlon)
                            input_intensity_final.append(input_intensity)
                            label_intensity_final.append(label_intensity)
                else:
                    print(f"{year}{tc} too short")
                    continue

           
            # print(time_latlon)
        
        print(f"TC all num: {all_tc_len}")
        return input_final, label_final, input_intensity_final, label_intensity_final, input_latlon_final, label_latlon_final


    
    def init_file_list_save_pre(self, years):
        IBTrACS_url = self.IBTrACS_url
        
        csv_data = pd.read_csv(IBTrACS_url, encoding="utf-8", low_memory=False, )
        headers = ["year", "SID", "ISO_TIME", "USA_ATCF_ID", "USA_STATUS",]
        #"USA_WIND", "USA_WIND_PRE", "USA_PRES", "USA_PRES_PRE",
        for num in range(self.output_length):
            h = (num+1) * 6
            if self.is_pre_latlon:
                headers = headers + [f"USA_WIND_{h}h"] + [f"USA_WIND_PRE_{h}h"] + [f"USA_PRES_{h}h"] + [f"USA_PRES_PRE_{h}h"]+ \
                                    [f"USA_LAT_{h}h"] +  [f"USA_LAT_PRE_{h}h"] +  [f"USA_LON_{h}h"] +  [f"USA_LON_PRE_{h}h"]
            else:
                headers = headers + [f"USA_WIND_{h}h"] + [f"USA_WIND_PRE_{h}h"] + [f"USA_PRES_{h}h"] + [f"USA_PRES_PRE_{h}h"]
        basin = self.valid_log_name.split('.')[0]
        path = os.path.join(self.cfgdir, f"Pre_{basin}_{self.valid_begin_year}_{self.valid_end_year}.csv")

        with open(path,'w',encoding='utf8',newline='') as f :
            writer = csv.writer(f)
            writer.writerow(headers)
        input_final = []
        label_final = []
        input_intensity_final = []
        label_intensity_final = []
        input_latlon_final = []
        label_latlon_final = []
        lifetime_num_dic = {}
        if self.split == "train" or self.split == "test":
            label_Basin = self.train_label_Basin
        else:
            label_Basin = self.test_label_Basin

        for year in years:
            
            year_data = csv_data.loc[(csv_data["SEASON"]==str(year))]
            
            all_year_tc = []

            #print(len(year_data))
            for i in range(len(year_data)):
                tc_name = year_data["SID"].iloc[i]
                tc_Basin = year_data["BASIN"].iloc[i]
                
                if pd.isna(tc_Basin):
                    tc_Basin="NA"
                if (tc_name not in all_year_tc) and (tc_Basin in label_Basin):
                    all_year_tc.append(tc_name)
                # if (tc_Basin not in label_Basin):
                #     print(tc_name, tc_Basin)
            
            print(len(all_year_tc))

            # import pdb
            # pdb.set_trace()
            
            for tc in all_year_tc:
                if self.is_use_lifetime_num:
                    
                    lifetime_num = -1
                tc_data = year_data.loc[(year_data["SID"]==tc)]
                len_tc = len(tc_data)
            
                if len_tc>=self.window_size:
                    time_need = []
                    label_need = []
                    for i in range(len_tc):
                        iso_time = tc_data["ISO_TIME"].iloc[i]
                        if iso_time[11:] in ["00:00:00", "06:00:00", "12:00:00", "18:00:00"]:
                            if self.is_use_lifetime_num:
                                lifetime_num = lifetime_num + 1                            
                            day_data = tc_data.loc[(tc_data["ISO_TIME"]==iso_time)]
                            day_label = []
                            flag = True
                            
                            for vname in self.label_vnames:
                                data_vname = np.array(day_data[vname])[0] 
                                #print(data_vname)
                                if data_vname != ' ':
                                    data_vname = float(data_vname)
                                    day_label.append(data_vname)
                                else:
                                    flag = False
                                    break
                            if flag:
                                #print(day_label)
                                iso_time_path = os.path.join(str(year), tc, iso_time)
                                time_need.append(iso_time_path)
                                label_need.append(day_label)
                                if self.is_use_lifetime_num:
                                    lifetime_num_dic[str(iso_time_path)] = lifetime_num                                
                    #print(time_need)
                    input_start = 0
                    input_end   = 0
                    label_start = 0
                    label_end   = 0
                    for j in range(len_tc-self.window_size):
                        
                        input_start = j
                        input_end   = input_start + self.input_length
                        label_start = input_end
                        label_end   = label_start + self.output_length

                        input_day = time_need[input_start:input_end]
                        label_day = time_need[label_start:label_end]

                        window_day = input_day + label_day
                        if len(window_day)!= self.window_size:
                            break
                        if self.is_use_singel_msw:
                            init_time = input_day[0].split('/')[-1]
                            init_data = tc_data.loc[(tc_data["ISO_TIME"]==init_time)]
                            init_data_vname = np.array(init_data[self.label_vnames[0]])[0]
                            
                            if (int(init_data_vname) - int(self.msw_mslp_choose[0]) == 0):
                                pass
                            else:
                                continue
                        if self.check_window(window_day):
                            input_final.append(input_day)
                            label_final.append(label_day)
                            input_intensity = []
                            input_latlon = []
                            label_latlon = []
                            for time_url in input_day:
                                in_time = time_url.split('/')[-1]
                                inp_data = tc_data.loc[(tc_data["ISO_TIME"]==in_time)]
                                day_label = []
                                for vname in self.label_vnames:
                                    data_vname = np.array(inp_data[vname])[0]
                                    data_vname = float(data_vname)
                                    day_label.append(data_vname)
                                if self.is_use_lifetime_num:
                                    day_label.append(lifetime_num_dic[str(time_url)])                                    
                                input_intensity.append(day_label)

                                lat = float(np.array(inp_data["LAT"])[0])
                                lon = float(np.array(inp_data["LON"])[0])
                                input_latlon.append([lat, lon])

                            label_intensity = []
                            
                            for lab_time in label_day:
                                lab_time = lab_time.split('/')[-1]
                                lab_data = tc_data.loc[(tc_data["ISO_TIME"]==lab_time)]
                                day_label = []
                                for vname in self.label_vnames:
                                    data_vname = np.array(lab_data[vname])[0]
                                    data_vname = float(data_vname)
                                    day_label.append(data_vname)
                                
                                
                                label_intensity.append(day_label)
                                lat = float(np.array(lab_data["LAT"])[0])
                                lon = float(np.array(lab_data["LON"])[0])
                                label_latlon.append([lat, lon])

                            last_day = input_day[-1].split('/')[-1]
                            last_inp_day = tc_data.loc[(tc_data["ISO_TIME"]==last_day)]
            
                            rows = {"year":year, "SID":tc, "ISO_TIME":last_day, "USA_ATCF_ID":np.array(last_inp_day["USA_ATCF_ID"])[0], "USA_STATUS":np.array(last_inp_day["USA_STATUS"])[0],}
                            #"USA_WIND", "USA_WIND_PRE", "USA_PRES", "USA_PRES_PRE",
                            for num in range(self.output_length):
                                h = (num+1) * 6
                                num_iso = label_day[num].split('/')[-1]
                                num_day = tc_data.loc[(tc_data["ISO_TIME"]==num_iso)]
                                rows[f"USA_WIND_{h}h"] = np.array(num_day["USA_WIND"])[0]
                                rows[f"USA_WIND_PRE_{h}h"] = " "
                                rows[f"USA_PRES_{h}h"] = np.array(num_day["USA_PRES"])[0]
                                rows[f"USA_PRES_PRE_{h}h"] = " "
                                if self.is_pre_latlon:
                                    rows[f"USA_LAT_{h}h"] = np.array(num_day["LAT"])[0]
                                    rows[f"USA_LAT_PRE_{h}h"] = " "
                                    rows[f"USA_LON_{h}h"] = np.array(num_day["LON"])[0]
                                    rows[f"USA_LON_PRE_{h}h"] = " "


                            basin = self.valid_log_name.split('.')[0]
                            path = os.path.join(self.cfgdir, f"Pre_{basin}_{self.valid_begin_year}_{self.valid_end_year}.csv")
            
                            with open(path,'a',encoding='utf8',newline='') as f :
                                writer = csv.DictWriter(f, fieldnames=rows.keys())
                                writer.writerow(rows)


                            input_latlon_final.append(input_latlon)
                            label_latlon_final.append(label_latlon)
                            input_intensity_final.append(input_intensity)
                            label_intensity_final.append(label_intensity)
                
                else:
                    print(f"{year}{tc} too short")
                    continue


            # print(time_latlon)
        return input_final, label_final, input_intensity_final, label_intensity_final, input_latlon_final, label_latlon_final


      
    def is_order(self, day1, day2):
        day1 = day1.split('/')[-1]
        day2 = day2.split('/')[-1]
        day1 = datetime(int(day1[0:4]), int(day1[5:7]), int(day1[8:10]), int(day1[11:13]), 0, 0)
        day2 = datetime(int(day2[0:4]), int(day2[5:7]), int(day2[8:10]), int(day2[11:13]), 0, 0)
        hour = day2 - day1
        
        if hour.__str__() == "6:00:00":
            return True
        else:
            return False
        
    def check_window(self, day_list):
        for i in range(len(day_list)-1):
            if self.is_order(day_list[i], day_list[i+1]):
                continue
            else:
                return False
        return True
                

    def get_era5_crop_meanstd(self):
        
        with open(f'{self.save_meanstd_dir}/ERA5_single_TC_mean_std.json',mode='r') as f:
            era5_crop_single_mean_std = json.load(f)
        with open(f'{self.save_meanstd_dir}/ERA5_TC_mean_std.json',mode='r') as f:
            era5_crop_mutil_mean_std = json.load(f)
        era5_crop_mean = []
        era5_crop_std = []
        
        for vname in self.single_level_vnames:
            vname = str(vname)
            era5_crop_mean.append(np.array([era5_crop_single_mean_std["mean"][vname]]))
            era5_crop_std.append(np.array([era5_crop_single_mean_std["std"][vname]]))
        for vname in self.multi_level_vnames:
            for height in self.height_level_list:
                vname = str(vname)
                height = str(height)
                
                era5_crop_mean.append(np.array([era5_crop_mutil_mean_std["mean"][vname][height]]))
                era5_crop_std.append(np.array([era5_crop_mutil_mean_std["std"][vname][height]]))

        return torch.from_numpy(np.concatenate(era5_crop_mean, axis=0)), torch.from_numpy(np.concatenate(era5_crop_std, axis=0))
    


    

    def get_intensity_meanstd(self):
        intensity_mean = np.array([49.040946869297855, 988.9592162921715])
        intensity_std = np.array([28.33988190313529,  21.844814128245602])
        return torch.from_numpy(np.array(intensity_mean)), torch.from_numpy(np.array(intensity_std))



    def get_meanstd(self):
        return self.intensity_mean, self.intensity_std
 
    def url_to_era5(self, url):
        
        aim = url.split('/')[-1]
        # #print(aim)
        year = aim[0:4]
        month = aim[5:7]
        day = aim[8:10]
        hour = aim[11:13]
        y_m_d = year + '-' + month + '-' + day
        h_m_s = hour + ':' + "00" + ':' + "00"
        era5_url = []
        era5_url_base = os.path.join(url, h_m_s)
        # era5_url_base = f"{self.data_dir}/{era5_url_base}"
        for vname in self.single_level_vnames:
            url_vname = f"{era5_url_base}-{vname}.npy"
            url_vname = f"{self.data_dir}/{url_vname}"
            era5_url.append(url_vname)
        for vname in self.multi_level_vnames:
            for height in self.height_level_list:
                url_vname = f"{era5_url_base}-{vname}-{height}.0.npy"
                url_vname = f"{self.data_dir}/{url_vname}"
                era5_url.append(url_vname)
        return era5_url
    

    def url_to_add_data(self, url):
        
        
        add_data_url = f"{self.add_data_path}/{url}/PI_data.npy"
        
        return add_data_url

    def url_to_add_data_fengwuPre(self, lead_time, url):
        
        
        add_data_url = f"{self.add_data_path}/{url}/{lead_time}h/PI_data.npy"
        
        return add_data_url

    def get_ceph(self, url):
        #print(url)
        with io.BytesIO(self.client.get(url)) as f:
            try:
                data = np.load(f, allow_pickle=True)
            except Exception as err:
                raise ValueError(f"{url}")
        return data


    def test_pic(self, name, data, cmap='viridis'):
        import matplotlib.pyplot as plt
        plt.imshow(data, cmap=cmap)
        plt.axis('off')
        plt.savefig(name, transparent=True)  



    def load_era5_full(self, index, inp_latlon=None):
        era5_inp_urls = self.era5_inp_url[index]
        #print(era5_inp_urls)
        era5_inp_data = []
        year = era5_inp_urls[self.input_length-1, 0].split("/")[-4]
        if self.is_use_fengwu or ((self.split=="train") and self.is_train_fengwu_crop and (int(year) in self.train_fengwu_crop_year_list)):
            len_era5_inp_urls = self.input_length
        else:
            len_era5_inp_urls = era5_inp_urls.shape[0]
        for i in range(len_era5_inp_urls):
            era5_inp_v = []
            head = era5_inp_urls[i, 0].split('//')[0]
            tail = era5_inp_urls[i, 0].split('//')[-1].split('/')[:-1]
            tail.append("ERA5_data.npy")
            head = head + '//'
            full_url = tail
            full_url = os.path.join(*full_url)
            full_url = head + full_url
            #print(full_url)
            full_data = self.get_ceph(full_url)
            index_list = []

            for vname in self.single_level_vnames:
                index_list.append(self.ERA5_vnames_dic[vname])
                
            for vname in self.multi_level_vnames:
                for height in self.height_level_list:
                    
                    index_list.append(self.ERA5_vnames_dic[vname][height])

            era5_inp_v = full_data[index_list, :, :]
            #print(f"ERA5 index_list: {index_list}")
            era5_inp_data.append(era5_inp_v)
        
        if self.is_use_fengwu:
            
            last_time = era5_inp_urls[self.input_length-1, 0]
            last_time = last_time.split("/")[-2]
          
            lat0 = inp_latlon[0]
            lon0 = inp_latlon[1]
            # print(last_time1, last_time2)
            # print(lat0)
            # print(lon0)
            if self.is_use_fengwu_crop:
                year = era5_inp_urls[self.input_length-1, 0].split("/")[-4]
                sid = era5_inp_urls[self.input_length-1, 0].split("/")[-3]
                time = era5_inp_urls[self.input_length-1, 0].split("/")[-2]
                base_url = os.path.join(self.fengwu_crop_base_dir, year, sid, time)
                fengwu_pre = self.fengwu_pre_crop(base_url=base_url,last_time=last_time)
            else:
                fengwu_pre = self.fengwu_pre(last_time=last_time, lat0=lat0, lon0=lon0)
        elif ((self.split=="train") and self.is_train_fengwu_crop and (int(year) in self.train_fengwu_crop_year_list)):
            last_time = era5_inp_urls[self.input_length-1, 0]
            last_time = last_time.split("/")[-2]
            year = era5_inp_urls[self.input_length-1, 0].split("/")[-4]
            sid = era5_inp_urls[self.input_length-1, 0].split("/")[-3]
            time = era5_inp_urls[self.input_length-1, 0].split("/")[-2]
            base_url = os.path.join(self.fengwu_crop_base_dir, year, sid, time)
            fengwu_pre = self.fengwu_pre_crop(base_url=base_url,last_time=last_time)
        era5_inp_data = np.array(era5_inp_data)
        era5_inp_data = torch.from_numpy(era5_inp_data)
        era5_inp_data = self.ERA5_transform(era5_inp_data)
        era5_inp_data = era5_inp_data.numpy()
        # print(era5_inp_data.shape)
        # print(fengwu_pre.shape)
        if self.is_use_fengwu or ((self.split=="train") and self.is_train_fengwu_crop and (int(year) in self.train_fengwu_crop_year_list)):
            era5_inp_data = np.concatenate((era5_inp_data, fengwu_pre), axis=0)
      
        return era5_inp_data




    def load_era5(self, index):
        era5_inp_urls = self.era5_inp_url[index]
        # print(era5_inp_urls)
        era5_inp_data = []
        for i in range(era5_inp_urls.shape[0]):
            era5_inp_v = []
            for j in range(era5_inp_urls.shape[1]):
                #time0 = time.time()
                data = self.get_ceph(era5_inp_urls[i, j])
                
                era5_inp_v.append(data)
            era5_inp_data.append(era5_inp_v)
        era5_inp_data = np.array(era5_inp_data)
       
        return era5_inp_data
    

    def load_add_data(self, index):
        urls = self.add_data_inp_url[index]
        urls = np.array(urls)
        #print(era5_inp_urls[0,0])
        datas = []
       
        for i in range(urls.shape[0]):
            url = urls[i]
            data = self.get_ceph(url)
            #print(f"GRIDAST index_list: {index_list}")
            datas.append(data)
        datas = np.array(datas)
        

        return datas


    def get_ERA5(self, index, inp_latlon, inp_lable): 
    
        era5_inp = self.load_era5_full(index=index, inp_latlon=inp_latlon)
        #era5_inp = self.load_era5(index=index)
        new_era5_data = era5_inp
        
        new_era5_data = np.array(new_era5_data)
        new_era5_data = torch.from_numpy(new_era5_data)
        new_era5_data = (new_era5_data - self.era5_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) / self.era5_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        new_era5_data = self.ERA5_transform(new_era5_data)



        new_era5_data = self.ERA5_transform(new_era5_data)

        
        if self.is_map_inp_intensity:
            map_era5 = torch.zeros((new_era5_data.shape[0], inp_lable.shape[1], new_era5_data.shape[2], new_era5_data.shape[3]))
            for t in range(inp_lable.shape[0]):
                for v in range(inp_lable.shape[1]):
                    map_era5[t, v, :, :] = inp_lable[t, v]
            new_era5_data = torch.concat((new_era5_data, map_era5), dim=1)
            


        return new_era5_data

    def get_add_data(self, index):
        add_data = self.load_add_data(index=index,)
        add_data = torch.from_numpy(add_data)
        # print(f"add_data:{add_data.shape}")
        add_data = torch.permute(add_data, (0,3,1,2))
        add_data = self.ERA5_transform(add_data)
        return add_data



    def fengwu_get_target(self, time, lead_time):
        #base_dir = "cephnew:s3://era5_np"
        # print(f"time: {time}")
        time = time.replace(" ", "T")
        base_dir = f"cephnew:s3://tropical_cyclone_data/Fengwu_v1_pre/nwp_initial_fileds/analysis_MIR/np721x1440/{time}/{lead_time}"
        single_level_vnames = ['u10', 'v10', 't2m', 'msl']
        multi_level_vnames = ['z', 'q', 'u', 'v', 't']

        height_level_list = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


        era5_inp_v = []
        for vname in single_level_vnames:
            url_vname = f"{vname}.npy"
            url_vname = f"{base_dir}/{url_vname}"
            
            data = self.get_ceph(url_vname)
            era5_inp_v.append(data)
        for vname in multi_level_vnames:
            for height in height_level_list:
                url_vname = f"{vname}-{height}.0.npy"
                url_vname = f"{base_dir}/{url_vname}"
                
                data = self.get_ceph(url_vname)
                era5_inp_v.append(data)

        era5_inp_data = np.array(era5_inp_v)
        #print(f"era5_inp_data: {era5_inp_data.shape}")
        #era5_inp_data = (era5_inp_data - self.fengwu_mean[:, np.newaxis, np.newaxis]) / self.fengwu_std[:, np.newaxis, np.newaxis]
        return era5_inp_data



    def fengwu_track_data_process(self, pre_data,):
        datasets = []
        
        #level:[1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 
        #       450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]


        for i in range(pre_data.shape[0]):
            msl = pre_data[i][self.all_vname_index["msl"]]
            # msl = msl * msl_std + msl_mean
            u_850 = pre_data[i][self.all_vname_index["u_850"]]
            
            v_850 = pre_data[i][self.all_vname_index["v_850"]]
            
            z_500 = pre_data[i][self.all_vname_index["z_500"]]
           
            tc_data = {}
            tc_data["msl"] = msl
            tc_data["u_850"] = u_850
            tc_data["v_850"] = v_850
            tc_data["z_500"] = z_500
            if self.cmp_wind:
                
                u10 = pre_data[i][self.all_vname_index["u10"]]
                
                v10 = pre_data[i][self.all_vname_index["v10"]]
                
                tc_data["u10"] = u10
                tc_data["v10"] = v10
            if self.cmp_thickness:
                
                z_850 = pre_data[i][self.all_vname_index["z_850"]]
                
                z_200 = pre_data[i][self.all_vname_index["z_200"]]
              
                # tc_data.append(z_850)
                # tc_data.append(z_200)
                tc_data["z_850"] = z_850
                tc_data["z_200"] = z_200
                    
            datasets.append(tc_data)
        return datasets


    def era5_process(self, data, lat, lon):
        """
        lat: 90~-90
        lon: 0~180 ~ -180 ~0
        to lat 0~180  lon 0~360

        """
        if lon<0:
            lon = 360+lon
        lon_np = round(lon * self.resolution)
        lat = 90 - lat
        lat_np = round(lat * self.resolution)


        R = round(self.radius * self.resolution)
        new_map = np.zeros((data.shape[0], 2*R+data.shape[1], 2*R+data.shape[2]), dtype=np.float32)

        new_map[:, 0:R, R:(R+data.shape[2])] = data[:, (data.shape[1]-R):data.shape[1], :]
        new_map[:, (R+data.shape[1]):, R:(R+data.shape[2])] = data[:, 0:R, :]

        new_map[:, R:(R+data.shape[1]), 0:R] = data[:, :, (data.shape[2]-R):data.shape[2]]
        new_map[:, R:(R+data.shape[1]), (R+data.shape[2]):] = data[:, :, 0:R]

        new_map[:, 0:R, 0:R] = data[:, (data.shape[1]-R):data.shape[1], (data.shape[2]-R):data.shape[2]]
        new_map[:, 0:R, (R+data.shape[2]):] = data[:, (data.shape[1]-R):data.shape[1], 0:R]
        new_map[:, (R+data.shape[1]):, 0:R] = data[:, 0:R, (data.shape[2]-R):data.shape[2]]
        new_map[:, (R+data.shape[1]):, (R+data.shape[2]):] = data[:, 0:R, 0:R]

        new_map[:, R:(R+data.shape[1]), R:(R+data.shape[2])] = data

        new_lat_np = lat_np + R
        new_lon_np = lon_np + R
        
        box_data = new_map[:, (new_lat_np-R):(new_lat_np+R), (new_lon_np-R):(new_lon_np+R)]
        
        return box_data

    
    def fengwu_track_know_true_loc(self, pre_data, lats, lons):
        crop_pre_data = []
        for i in range(pre_data.shape[0]):
            lat = lats[i]
            lon = lons[i]
            crop_data = self.era5_process(pre_data[i], lat=lat, lon=lon)
            crop_pre_data.append(crop_data)
        
        crop_pre_data = np.array(crop_pre_data)
        return crop_pre_data


    def fengwu_pre(self, last_time, lat0, lon0):
        

   
        pre_data = []
        #print(f"input: {input.shape}")
        for i in range(self.output_length):
            lead_time = (i+1)*6
            output_norm = self.fengwu_get_target(time=last_time, lead_time=lead_time).astype(np.float32)
            pre_data.append(output_norm)
            #output = (output[0, :69] * data_std) + data_mean
            # print(output_norm.shape)
        pre_data = np.array(pre_data)

        crop_fengwu = self.fengwu_track_know_true_loc(pre_data, lat0, lon0)

        return crop_fengwu
    def fengwu_pre_crop(self, base_url, last_time):
        

        pre_data = []
        #print(f"input: {input.shape}")
        for i in range(self.output_length):
            lead_time = (i+1)*6
            
            full_url = os.path.join(base_url, f"{lead_time}h/ERA5_data.npy")

            full_data = self.get_ceph(full_url)
            index_list = []

            for vname in self.single_level_vnames:
                index_list.append(self.ERA5_vnames_dic[vname])
                
            for vname in self.multi_level_vnames:
                for height in self.height_level_list:
                    
                    index_list.append(self.ERA5_vnames_dic[vname][height])

            era5_inp_v = full_data[index_list, :, :]

            pre_data.append(era5_inp_v)
            #output = (output[0, :69] * data_std) + data_mean
            # print(output_norm.shape)
        pre_data = np.array(pre_data)
        pre_data = torch.from_numpy(pre_data)
       
        # print(pre_data.shape)
        
        crop_fengwu = self.ERA5_transform(pre_data)
        crop_fengwu = crop_fengwu.numpy()
 
        return crop_fengwu
    def check_lon(self, lons):
        len_lon = len(lons)
        for i in range(len_lon):
            if lons[i]>=180:
                lons[i] = lons[i] - 360
        return lons
    def __getitem__(self, index):
       

        label = self.label_intensity_final[index]
        inp_lable = self.input_intensity_final[index]
        inp_latlon = self.input_latlon_final[index]
        label_latlon = self.label_latlon_final[index]

        label = np.array(label)
        inp_lable = np.array(inp_lable)
        inp_latlon = np.array(inp_latlon)
        label_latlon = np.array(label_latlon)
        
 

        label = torch.from_numpy(label) 
        inp_lable = torch.from_numpy(inp_lable)

        inp_latlon = torch.from_numpy(inp_latlon) 
        label_latlon = torch.from_numpy(label_latlon)


        label = (label - self.intensity_mean.unsqueeze(0)) / self.intensity_std.unsqueeze(0)
    
        if self.is_use_lifetime_num:
            inp_lable[:, :-1] = (inp_lable[:, :-1] - self.intensity_mean.unsqueeze(0)) / self.intensity_std.unsqueeze(0)
        else:
            inp_lable = (inp_lable - self.intensity_mean.unsqueeze(0)) / self.intensity_std.unsqueeze(0)
        if self.is_diff:
            
            for i in range(label.shape[0]-1, 0, -1):
                label[i] = label[i] - label[i-1]
            label[0] = label[0] - inp_lable[-1]

        label = label[(self.output_step_length-1)::self.output_step_length, :]

       
        new_era5_data = torch.tensor(0)
  
        if "ERA5" in self.inp_type:
            
            inp_latlon_era5 = self.label_latlon_final[index]
            lat = []
            lon = []
            for i in range(len(inp_latlon_era5)):
                lat.append(inp_latlon_era5[i][0])
                lon.append(inp_latlon_era5[i][1])
            inp_latlon_era5 = [lat, lon]
            new_era5_data = self.get_ERA5(index, inp_latlon_era5, inp_lable)

            if self.set_ERA5_zero:
                new_era5_data = new_era5_data * torch.tensor(0)
            if self.is_cmp_origin_field_intensity:
                new_era5_data = new_era5_data * self.era5_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) + self.era5_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                
        if self.set_Seq_zero:
            inp_lable = inp_lable * torch.tensor(0)

        if self.is_pre_latlon:
            inp_latlon[:, 0] = inp_latlon[:, 0]/self.latlon_scale[0]
            inp_latlon[:, 1] = self.check_lon(inp_latlon[:, 1])
            inp_latlon[:, 1] = inp_latlon[:, 1]/self.latlon_scale[1]
            label_latlon[:, 0] = label_latlon[:, 0]/self.latlon_scale[0]
            label_latlon[:, 1] = self.check_lon(label_latlon[:, 1])
            label_latlon[:, 1] = label_latlon[:, 1]/self.latlon_scale[1]
            

            inp_lable = torch.concat((inp_lable, inp_latlon), dim=-1)
            label = torch.concat((label, label_latlon), dim=-1)

        add_data = torch.tensor(0.0)
        if self.is_add_data:
            add_data = self.get_add_data(index)
            add_data = torch.where(torch.isnan(add_data), torch.full_like(add_data, 0), add_data)
            add_data[:,1] = torch.where(add_data[:,1]==0, 1100, add_data[:,1])

        if self.set_PI_zero:
            add_data = add_data * torch.tensor(0)
        if self.set_fengwu_zero:
            new_era5_data[self.input_length:] = new_era5_data[self.input_length:] * torch.tensor(0)
        
        inp_last_tc_day_info = torch.tensor(0.0)
        if self.split=="valid":
            input_day = self.input_day_list[index][-1]
            year = input_day.split('/')[0]
            sid = input_day.split('/')[1]
            iso_time = input_day.split('/')[2]
            inp_last_tc_day_info = {"ALL_Year":self.all_year,"Year":year, "SID":sid, "ISO_TIME":iso_time}
            
        
        weight = torch.tensor(0.0)
     
            
        if self.is_hid_vq:
            inp_lable = torch.concat((inp_lable, label), dim=0)
        
        return new_era5_data, inp_lable, label, inp_last_tc_day_info, weight, add_data
    
    def __len__(self):    
        return self.len_file


