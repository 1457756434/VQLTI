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

from joblib import Parallel, delayed

import metpy.calc as mpcalc

from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d
import torchvision.transforms.functional as TF
import random

import argparse
import yaml

from tcpyPI import pi

# from s3_client import s3_client


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


class GRIDSAT_crop_dataset_fengwu_pre(Dataset):
    def __init__(self, data_dir="cephnew:s3://tropical_cyclone_data/GRIDSAT/npy_fengwu_era5", split='train', **kwargs) -> None:
        super().__init__()
        #print("init begin")
        self.data_dir = data_dir
        self.save_csv_name = kwargs.get('save_csv_name', "FengWu_Pre_Lat_Lon")
        self.base_save_dir = kwargs.get('base_save_dir', "cephnew:s3://tropical_cyclone_data/GRIDSAT/npy_fengwu_era5")
        self.IBTrACS_url = kwargs.get('IBTrACS_url', 'dataset/ibtracs.ALL.list.v04r00.csv')
        self.save_meanstd_dir = kwargs.get('save_meanstd_dir', 'dataset/new_meanstd')

        self.valid_log_name = kwargs.get('valid_log_name', "all_basin")
        self.cfgdir = kwargs.get('cfgdir', "TC_Pre_base_FengWu")

        self.img_data_nan_rate_threshold = kwargs.get('img_data_nan_rate_threshold', 0.01)

        self.train_begin_year = kwargs.get('train_begin_year', 1980)
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
       
        self.output_length = kwargs.get('forecast_time', 4)

        self.is_fengwu_pre = kwargs.get('is_fengwu_pre', False)



        self.output_step_length = kwargs.get('output_step_length', 1)

        self.ERA5_image_size = kwargs.get('ERA5_image_size', 40)
        
       

        self.resolution = kwargs.get('resolution', 0.25)
      
        self.resolution = 1 / self.resolution
     
        self.radius = kwargs.get('radius', 10)
        self.radius_np = int(self.radius * self.resolution)

        self.time_interval = kwargs.get('time_interval', 6)
        if self.time_interval==3:
            self.daytime_need_list = ["00:00:00", "03:00:00", "06:00:00", "09:00:00", "12:00:00", "15:00:00", "18:00:00", "21:00:00"]
        else:
            self.daytime_need_list = ["00:00:00", "06:00:00", "12:00:00", "18:00:00"]

        Years_dict = kwargs.get('years', Years)
        self.is_map_inp_intensity = kwargs.get('is_map_inp_intensity', False)

        self.is_save_npy = kwargs.get('is_save_npy', False)
        self.is_load_npy = kwargs.get('is_load_npy', True)
        self.is_use_lifetime_num = kwargs.get("is_use_lifetime_num", False)

        self.inp_type = kwargs.get('inp_type', ["ERA5", "Seq"])
      
        self.set_ERA5_zero = kwargs.get('set_ERA5_zero', False)
        self.set_Seq_zero = kwargs.get('set_Seq_zero', False)

        vnames_type = kwargs.get("vnames", {})
        self.constants_types = vnames_type.get('constants', [])
       
   
     
        self.single_level_vnames = vnames_type.get('single_level_vnames', ['u10', 'v10', 't2m', 'msl'])
        self.multi_level_vnames = vnames_type.get('multi_level_vnames', ['z', 'q', 'u', 'v', 't'])
        self.height_level_list = vnames_type.get('hight_level_list', [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])

        self.height_level_indexes = [height_level.index(j) for j in self.height_level_list]


        self.label_vnames = vnames_type.get('label_vnames', ["USA_WIND", "USA_PRES"])
  
       
        self.train_label_Basin = vnames_type.get('train_label_Basin', ["EP", "NA", "NI", "SA", "SI", "SP", "WP"])
        self.test_label_Basin = vnames_type.get('test_label_Basin', ["EP", "NA", "NI", "SA", "SI", "SP", "WP"])

    

        self.split = split
        
        self.client = Client(conf_path="~/petreloss.conf")
        

        if len(self.constants_types) > 0:
            self.constants_data = self.get_constants_data(self.constants_types)
        else:
            self.constants_data = None
        


        self.data_element_num = len(self.single_level_vnames) + len(self.multi_level_vnames) * len(self.height_level_list)



        years = Years_dict[split]
        is_FengWu_Pre = kwargs.get('is_FengWu_Pre', False)
        if is_FengWu_Pre:
            self.dir_save_final, self.latlon_final, self.input_day_list = self.init_file_list_fengwuPre(years)
        else:
            self.dir_save_final, self.latlon_final, self.input_day_list = self.init_file_list(years)


        self.era5_inp_url = []
    


        for save_url in self.dir_save_final:
            
            
            if is_FengWu_Pre:
                url = self.saveurl_to_inpurl_fengwuPre(save_url)
            else:
                url = self.saveurl_to_inpurl(save_url)
                
            self.era5_inp_url.append(url)
            
        self.len_file = len(self.input_day_list)
        print("dataset length:{}".format(self.len_file))
        

        
        self.is_diff = kwargs.get("is_diff", False)
        
        self.is_use_fengwu = kwargs.get("is_use_fengwu", True)
        self.use_track_algorithm = kwargs.get("use_track_algorithm", True)
        if self.is_use_fengwu:
            
            self.vo_region = [89, -89, 0, 360]
            self.radius_mslp = 4.5
            self.cmp_wind = True
            self.cmp_thickness =True
            self.cmp_lsm = True

            

            self.init_resulation = 0.25
            self.is_wait_genesis = False
            self.empty_time_threshold = self.output_length
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



    

    def init_file_list(self, years):
        IBTrACS_url = self.IBTrACS_url
        csv_data = pd.read_csv(IBTrACS_url, encoding="utf-8", low_memory=False, )
        latlon_final = []
        dir_save_final = []
        day_list_final = []
        
        all_tc_len = 0

        base_save_dir = self.base_save_dir

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
                
                tc_data = year_data.loc[(year_data["SID"]==tc)]
                len_tc = len(tc_data)
       
                
                time_need = []
                label_need = []
                for i in range(len_tc):
                    iso_time = tc_data["ISO_TIME"].iloc[i]
                    
                    if iso_time[11:] in self.daytime_need_list:
                        
                        day_data = tc_data.loc[(tc_data["ISO_TIME"]==iso_time)]
                        day_label = []
                        iso_time_path = os.path.join(str(year), tc, iso_time)
                        time_need.append(iso_time_path)
                        label_need.append(day_label)
                        lat = float(np.array(day_data["LAT"])[0])
                        lon = float(np.array(day_data["LON"])[0])
                        dir_save = os.path.join(base_save_dir, str(year), tc, iso_time, "PI_data.npy")
                        day_list_final.append(iso_time)
                        dir_save_final.append(dir_save)
                        latlon_final.append([lat, lon])
    
                    
            # print(time_latlon)
        
        print(f"TC all num: {all_tc_len}")
        return dir_save_final, latlon_final, day_list_final


    def init_file_list_fengwuPre(self, years):
        IBTrACS_url = self.IBTrACS_url
        csv_data = pd.read_csv(IBTrACS_url, encoding="utf-8", low_memory=False, )
        latlon_final = []
        dir_save_final = []
        day_list_final = []
        
        all_tc_len = 0

        base_save_dir = self.base_save_dir

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
                
                tc_data = year_data.loc[(year_data["SID"]==tc)]
                len_tc = len(tc_data)
    
                
                time_need = []
                label_need = []
                for i in range(len_tc):
                    iso_time = tc_data["ISO_TIME"].iloc[i]
                    
                    if iso_time[11:] in self.daytime_need_list:
                        
                        day_data = tc_data.loc[(tc_data["ISO_TIME"]==iso_time)]
                        day_label = []
                        iso_time_path = os.path.join(str(year), tc, iso_time)
                        time_need.append(iso_time_path)
                        label_need.append(day_label)
                        lat = float(np.array(day_data["LAT"])[0])
                        lon = float(np.array(day_data["LON"])[0])
                        for i in range(self.output_length):
                            lead_time = (i+1)*6
                            base_url = os.path.join(base_save_dir, str(year), tc, iso_time)
                            dir_save = os.path.join(base_url, f"{lead_time}h/PI_data.npy")
                            dir_save_final.append(dir_save)
                            day_list_final.append(iso_time)
                            latlon_final.append([lat, lon])
    
                    
            # print(time_latlon)
        
        print(f"TC all num: {all_tc_len}")
        return dir_save_final, latlon_final, day_list_final

    def saveurl_to_inpurl(self, url):
        "cephnew:s3://tropical_cyclone_data/npy_fengwu_era5_PI/2021/2021282N16165/2021-10-10 18:00:00/PI_data.npy"
        url_tail = url.split('/')[-4:-1]
        inpurl = os.path.join(self.data_dir, *url_tail, "ERA5_data.npy")
        return inpurl
    def saveurl_to_inpurl_fengwuPre(self, url):
        "cephnew:s3://tropical_cyclone_data/Fengwu_pretrain_v2_pre_crop_PI/2021/2020317S04092/2020-11-11 12:00:00/6h/PI_data.npy"
        url_tail = url.split('/')[-5:-1]
        inpurl = os.path.join(self.data_dir, *url_tail, "ERA5_data.npy")
        return inpurl
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
        era5_url = f"{self.data_dir}/{era5_url_base}/ERA5_data.npy"
        return era5_url
    


    def get_ceph(self, url):
        with io.BytesIO(self.client.get(url)) as f:
            try:
                data = np.load(f, allow_pickle=True)
            except Exception as err:
                raise ValueError(f"{url}")
        return data


    def get_GRIDSAT_npy(self, url):
        base_url = f"cephnew:s3://tropical_cyclone_data/HURSAT_B1"
        input_save_url = f"{base_url}/npy"
        url_list = url.split("//")[-1].split("/")[2:]
        url_list[-1] = url_list[-1].replace(".nc", ".npy")
        input_save_url = os.path.join(input_save_url, *url_list)
        # print("Load GRIDSAT_npy")
        # print(input_save_url)
        with io.BytesIO(self.client.get(input_save_url)) as f:
            try:
                data = np.load(f, allow_pickle=True)
            except Exception as err:
                raise ValueError(f"{input_save_url}")
        #print(data.shape)
        return data



    def is_need_process_url(self, url, is_print=False):
        exists = self.client.contains(url)
        if exists:
            if is_print:
                print(f"{url} exist")
            return False
        else:
            return True
    def save_crop_data(self, data, url, is_print=False):
        
        input_save_url = url
        if is_print:
            print(input_save_url)
        
        with io.BytesIO() as f:
            np.save(f, data)
            f.seek(0)
            self.client.put(input_save_url, f)
    def save_data(self, data, url, is_print=False):
        
        input_save_url = url
        if is_print:
            print(input_save_url)
        
        with io.BytesIO() as f:
            np.save(f, data)
            f.seek(0)
            self.client.put(input_save_url, f)




    def get_ERA5(self, index,): 
       
        save_url = self.dir_save_final[index]
        url_era5 = self.era5_inp_url[index] 
        # print(save_url)
        # print(url_era5)


        if self.is_need_process_url(url_era5):
            return None
        else:
            pre_flag = True
      
        if self.is_need_process_url(save_url):
            print(f"need {save_url}")
            pre_flag = True
        if pre_flag:
            data_true = self.get_ceph(url_era5)
            

            C,H,W = data_true.shape[-3], data_true.shape[-2], data_true.shape[-1]
        
            data = data_true.reshape(C,-1).transpose(1,0)
            # plt.imshow(data_true[3])
            # plt.show()

            sst1 = data[:, 2] - 273.15
            msl1 = data[:, 3]/100
            # p1 = np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
            p1 = np.array([[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]]*(H*W))
            t1 = data[:, -13:] - 273.15
            q1 = data[:, 3+13:3+13+13] * 1e3

            # print(f"p1:{p1.shape}")
            # print(f"t1:{t1.shape}")
            # print(f"q1:{q1.shape}")

            p1 = p1[:, ::-1]
            t1 = t1[:, ::-1]
            q1 = q1[:, ::-1]

            # print(f"t1:{t1}")
            # print(f"q1:{q1}")
            #pesq_score = Parallel(n_jobs=7)(delayed(self.pesq_loss)(c[0].cpu().numpy(),c[1].cpu().numpy(),n[0].cpu().numpy(),n[1].cpu().numpy()) for c, n in zip(clean, noisy))
            result = []
            for i in range(H*W):
                pre = pi(sst1[i], msl1[i], p1[i], t1[i], q1[i], 0.9,0,1,0.8,0)
                result.append(pre)
            # result = Parallel(n_jobs=8)(delayed(pi)(sst1[i], msl1[i], p1[i], t1[i], q1[i], 0.9,0,1,0.8,0) for i in range(H*W))
            result = np.array(result)
            result = np.reshape(result, (H,W,result.shape[-1]))
            
            self.save_data(data=result, url=save_url, is_print=True)
        return None


    def get_lds_kernel_window(self, kernel, ks, sigma):
        assert kernel in ['gaussian', 'triang', 'laplace']
        half_ks = (ks - 1) // 2
        if kernel == 'gaussian':
            base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
            kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
        elif kernel == 'triang':
            kernel_window = triang(ks)
        else:
            laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
            kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

        return kernel_window


    def _prepare_weights(self, labels, reweight, min_target=0, max_target=200, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        assert reweight in {'none', 'inverse', 'sqrt_inv'}
        assert reweight != 'none' if lds else True, \
            "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

        labels = labels - min_target
        max_target = max_target - min_target

        value_dict = {x: 0 for x in range(max_target)}
        # labels = self.label_intensity_final
        for label in labels:
            value_dict[min(max_target - 1, int(label))] += 1
        if reweight == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif reweight == 'inverse':
            value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
        num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
        if not len(num_per_label) or reweight == 'none':
            return None
        print(f"Using re-weighting: [{reweight.upper()}]")

        if lds:
            lds_kernel_window = self.get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
            num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]
        return weights

    def fengwu_get_target(self, time, lead_time):
        #base_dir = "cephnew:s3://era5_np"
        # print(f"time: {time}")
        time = time.replace(" ", "T")
        base_dir = f"{self.data_dir}/nwp_initial_fileds/analysis_MIR/np721x1440/{time}/{lead_time}"
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
        era5_inp_data = era5_inp_data * self.fengwu_std[:, np.newaxis, np.newaxis] + self.fengwu_mean[:, np.newaxis, np.newaxis]
        return era5_inp_data



    def fengwu_track_data_process(self, pre_data,):
        datasets = []

        

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

    
    def fengwu_track(self, pre_data, lat0, lon0):
        from track_algorithm_new import centre_location
     
        track_need_data = self.fengwu_track_data_process(pre_data=pre_data)


        if self.cmp_wind:
            fengwu_TC, MSL_location, wind_speed_max = centre_location(lat0, lon0, track_need_data, radius_mslp=self.radius_mslp, dx=self.dx, dy=self.dy, \
                                                                    region=self.vo_region, cmp_wind=self.cmp_wind, cmp_thickness=self.cmp_thickness, cmp_lsm=self.cmp_lsm,\
                                                                    is_wait_genesis=self.is_wait_genesis, empty_time_threshold=self.empty_time_threshold, init_resulation=self.init_resulation)
        else:
            fengwu_TC, MSL_location = centre_location(lat0, lon0, track_need_data, radius_mslp=self.radius_mslp, dx=self.dx, dy=self.dy, region=self.vo_region, \
                                                    cmp_wind=self.cmp_wind, cmp_thickness=self.cmp_thickness, cmp_lsm=self.cmp_lsm, \
                                                    is_wait_genesis=self.is_wait_genesis, empty_time_threshold=self.empty_time_threshold, init_resulation=self.init_resulation)
        crop_pre_data = []
        
        for i in range(pre_data.shape[0]):
            lat = fengwu_TC[i][0]
            lon = fengwu_TC[i][1]
            
            crop_data = self.era5_process(pre_data[i], lat=lat, lon=lon)
            crop_pre_data.append(crop_data)
        
        crop_pre_data = np.array(crop_pre_data)
        
        
        return crop_pre_data, fengwu_TC


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
            output = self.fengwu_get_target(time=last_time, lead_time=lead_time).astype(np.float32)
            pre_data.append(output)

            #output = (output[0, :69] * data_std) + data_mean
            # print(output_norm.shape)
        pre_data = np.array(pre_data)
   
        
        crop_fengwu, fengwu_TC = self.fengwu_track(pre_data, lat0, lon0)
    
        return crop_fengwu, fengwu_TC

    def __getitem__(self, index):
       

    

        if "ERA5" in self.inp_type:

            
            self.get_ERA5(index, )
        
    
    def __len__(self):    
        return self.len_file




def singel_process_code(data_set, now_range):
    for i in now_range:
        data_set.__getitem__(i)
        
        #print(i)
    #print("***************")


from multiprocessing import Process
import multiprocessing
def save_cropdata(years, 
                data_dir="cephnew:s3://tropical_cyclone_data/Fengwu_v1_pre",
                base_save_dir="cephnew:s3://tropical_cyclone_data/Fengwu_v1_pre_crop_PI", 
                radius=10, time_interval=6, save_csv_name="FengWu_Pre_Lat_Lon", forecast_time=4, Process_nums=1,
                is_FengWu_Pre=False,): 
    data_set = GRIDSAT_crop_dataset_fengwu_pre(
                                            save_meanstd_dir='dataset/npy_fengwu_era5_meanstd_140' ,\
                                            split='train', years=years, inp_type=["IR", "ERA5", "Seq"], is_map_inp_intensity=False,\
                                            is_fengwu_pre=True, is_use_fengwu=True, use_track_algorithm = True, \
                                            data_dir=data_dir, base_save_dir=base_save_dir, radius=radius, forecast_time=forecast_time,\
                                            time_interval=time_interval, save_csv_name=save_csv_name,
                                            is_FengWu_Pre=is_FengWu_Pre,)
    n = Process_nums
    step = int(len(data_set)/n)
    range_split = [ list(range(i,i+step,1)) for i in range(0,len(data_set),step)]
    process_list = []
    for i in range(n):
        now_range = range_split[i]
        p = Process(target=singel_process_code, args=(data_set, now_range))
        p.start()
        process_list.append(p)
            
    for i in process_list:
        p.join()
    
          
   



def muti_save_npy(end_year =2018, begin_year = 2018, \
                  data_dir="cephnew:s3://tropical_cyclone_data/GRIDSAT/npy_fengwu_era5", base_save_dir="cephnew:s3://tropical_cyclone_data/GRIDSAT/npy_fengwu_era5", 
                  radius=10, time_interval=6, save_csv_name="FengWu_Pre_Lat_Lon", forecast_time=4, Process_nums=1,
                  is_FengWu_Pre=False):
    
    
    # lock = multiprocessing.Lock()
  
    years = {
        'train': range(begin_year, end_year+1),
    }
    save_cropdata(years=years, data_dir=data_dir, base_save_dir=base_save_dir, 
                radius=radius, time_interval=time_interval, save_csv_name=save_csv_name,
                forecast_time=forecast_time, Process_nums=Process_nums, 
                is_FengWu_Pre=is_FengWu_Pre)
    




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--world_size',     type = int,     default = -1,                                           help = 'number of distributed processes')
    parser.add_argument('--cfg', '-c',    type = str,     default = "TC_Pre_base_FengWu/dataset/save_PI/PI_config/PI_config_mutil_process.yaml",  help = 'path to the configuration file')

    args = parser.parse_args()
    with open(args.cfg, 'r') as cfg_file:
        cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)



    data_dir = cfg_params["data_dir"]
    base_save_dir = cfg_params["base_save_dir"]
    radius = cfg_params["radius"]
    time_interval = cfg_params["time_interval"]
    end_year = cfg_params["end_year"] 
    begin_year = cfg_params["begin_year"] 
    Process_nums = cfg_params["Process_nums"] 
    is_FengWu_Pre = cfg_params["is_FengWu_Pre"]
    forecast_time = cfg_params["forecast_time"]
    muti_save_npy(end_year=end_year, begin_year = begin_year, data_dir=data_dir, 
                  base_save_dir=base_save_dir, radius=radius, time_interval=time_interval,
                 Process_nums=Process_nums, is_FengWu_Pre=is_FengWu_Pre, forecast_time=forecast_time)







#srun -p ai4earth --kill-on-bad-exit=1 -x SH-IDC1-10-140-24-11 --time=1:31:5 python -u 
#srun -p ai4earth --kill-on-bad-exit=1 -o job2/%j.out  --async python -u 




#srun -p ai4earth --kill-on-bad-exit=1 -x SH-IDC1-10-140-24-11 python -u 


