import xarray as xr
import numpy as np
from skimage.feature import peak_local_max

import pandas as pd
import metpy.calc as mpcalc
import metpy
from metpy.units import units

import xarray as xr

import io
import os
from petrel_client.client import Client


def dis_point(point, center):
    dis = (point[0] - center[0])**2 + (point[1] - center[1])**2
    return dis



def lon_to_360(lon):
    if lon <0:
        lon = 360 + lon
    return lon


def get_lsm():
    path = "lsm/2020-01-01.nc"
    data = xr.open_dataset(path,)
    lstm = data["lsm"]
    return lstm



def is_TC(TC_lat, TC_lon, vortivity, thickness=None, wind10_pre=None, lsm=None, cmp_thickness=False, cmp_lsm=False, cmp_wind=False, region=[45, -45, 100, 300], resulation=0.25 ,):
    radius_vo = 3
    radius_th = 3
    radius_wind = 3
    print(TC_lat, TC_lon)
    radius_vo_npy = round(radius_vo/resulation)
    radius_th_npy = round(radius_th/resulation)
    TC_lat_npy = round(TC_lat/resulation) - round((90 - region[0])/resulation)
    TC_lon_npy = round(TC_lon/resulation) - round(region[2]/resulation)

    print(round((90 - region[0])/resulation))
    print(TC_lat_npy, TC_lon_npy)


    print(TC_lat_npy-radius_vo_npy, TC_lat_npy+radius_vo_npy)
    print(TC_lon_npy-radius_vo_npy, TC_lon_npy+radius_vo_npy)
    vortivity = vortivity[TC_lat_npy-radius_vo_npy:TC_lat_npy+radius_vo_npy, TC_lon_npy-radius_vo_npy:TC_lon_npy+radius_vo_npy]
    vortivity = vortivity.magnitude * 1e5

    th_loc = [1]
    if cmp_thickness and ((TC_lat<60) or ((TC_lat>120))):
        th_loc = []
      
        print(thickness.shape)
        thickness = thickness[round(TC_lat/resulation)-radius_th_npy:round(TC_lat/resulation)+radius_th_npy, round(TC_lon/resulation)-radius_th_npy:round(TC_lon/resulation)+radius_th_npy]
        th_loc = peak_local_max(thickness, min_distance=1)
        if (len(th_loc)!=0):
            th_max_list = []
            for coordinate in th_loc:
                value = thickness[coordinate[0]][coordinate[1]]
                th_max_list.append(value)
            th_max = np.max(th_max_list)
            
        else:
            print("no thickness")
            return False
    
    if cmp_wind:
        speed = max_wind(wind10_pre, 90.0 - TC_lat, TC_lon, radius_wind=radius_wind, resulation=resulation)
        
        print(speed)
    
        if cmp_lsm:
            is_land = lsm.sel(latitude=90.0 - TC_lat, longitude=TC_lon, method="nearest").data
            # print(is_land)
            if is_land.any()>0:
                
                if speed<8:
                    
                    return False
    vo_loc = []
    if len(vortivity) != 0:
        
        vo_loc_po = peak_local_max(vortivity, min_distance=1)
        vo_loc_ne = peak_local_max(-vortivity, min_distance=1)
        vo_loc = np.concatenate((vo_loc_po,vo_loc_ne), axis=0)
       
        print(vo_loc.shape)

    #th_loc = peak_local_max(thickness, min_distance=20)
    if (len(vo_loc) != 0) and (len(th_loc)!=0):
        vo_max_list = []
       
        for coordinate in vo_loc:
            value = abs(vortivity[coordinate[0]][coordinate[1]])
            vo_max_list.append(value)
        vo_max = np.max(vo_max_list)
      
        print(vo_max)
    else:
        return False
    
    # th_max_list = []
    # for coordinate in th_loc:
    #     th_max_list = th_max_list.append(thickness.loc[coordinate[1], coordinate[0]].data)
    # th_max = np.max(th_max_list)

    # if (vo_max>=5*1e-5) and (len(th_max)!=0):
    if (abs(vo_max)>=5):
        return True
    else:
        return False


def max_wind(wind, TC0_lat, TC0_lon, radius_wind=2, resulation=0.25):
    TC0_lon = TC0_lon
    TC0_lat = 90.0 - TC0_lat
    
    print(TC0_lat, TC0_lon)
    radius_wind_npy = round(radius_wind/resulation)
    TC0_lon_npy = round(TC0_lon/resulation)
    TC0_lat_npy = round(TC0_lat/resulation)
    wind_data = wind[(TC0_lat_npy-radius_wind_npy):(TC0_lat_npy+radius_wind_npy), (TC0_lon_npy-radius_wind_npy):(TC0_lon_npy+radius_wind_npy)]
    # import pdb
    # pdb.set_trace()
    wind_speed = np.max(wind_data)
    return wind_speed

def centre_location(TC0_lat, TC0_lon,  data, dx, dy, region, radius_mslp=7, resulation=0.25, cmp_wind=False, cmp_lsm=False, cmp_thickness=False, \
                    is_wait_genesis=False, empty_time_threshold=1, init_resulation=0.25):
    """
    time : time[star, end]
    TC0_lon:[0  ~  180 -180 ~ 0]
    TC0_lat:[90  ~  -90]
    data : [time, msl, u_850, v_850, u10, v10, z850, z200,]
    """
  
    print(TC0_lat, TC0_lon)
    TC0_lon = lon_to_360(TC0_lon)
    TC0_lat = 90.0 - TC0_lat

    print(TC0_lat, TC0_lon)
    
    
 
    # print(region_npy)
    if cmp_lsm:
        lsm = get_lsm()
    else:
        lsm = None


    TC_location = []
    MSL_location = []
    thickness = None
    wind10_pre = None
    if cmp_wind:
        wind_speed_max = []
        radius_wind = 10
        #radius_wind_npy = round(radius_wind/resulation)
    time_len = len(data)

    # print(time_len)
    
    
    
    # 
    TC0_lon_npy = round(TC0_lon/init_resulation)
    TC0_lat_npy = round(TC0_lat/init_resulation)

    empty_time = 0

    if is_wait_genesis:
        genesis_flag = 0

    for i in range(0, time_len):
        if i ==0:
            region_npy = [round((90 - region[0])/init_resulation), round((90 - region[1])/init_resulation)+1, round(region[2]/init_resulation), round(region[3]/init_resulation)+1]
            radius_mslp_npy = round(radius_mslp/init_resulation)
            resulation_use = init_resulation
        else:
            region_npy = [round((90 - region[0])/resulation), round((90 - region[1])/resulation)+1, round(region[2]/resulation), round(region[3]/resulation)+1]
            radius_mslp_npy = round(radius_mslp/resulation)
            resulation_use = resulation
        print("==============================================================================")
        pre_data = data[i]
        if cmp_wind:
            u10_pre = pre_data["u10"]
            v10_pre = pre_data["v10"]
            wind10_pre = np.sqrt(u10_pre**2+v10_pre**2)
        if cmp_thickness:
            z850_pre = pre_data["z_850"]
            z200_pre = pre_data["z_200"]
            thickness =  z200_pre - z850_pre
        mslp_pre = pre_data["msl"]
     

     
        # print(mslp_pre.shape)
        u_wind_850 = pre_data["u_850"] * units("m/s")
        v_wind_850 = pre_data["v_850"] * units("m/s")
        
        
        ######################################################################################################
        u_wind_850 = u_wind_850[region_npy[0]:region_npy[1], region_npy[2]:region_npy[3]]
        v_wind_850 = v_wind_850[region_npy[0]:region_npy[1], region_npy[2]:region_npy[3]]

        ######################################################################################################
        
        mslp_pre_data = mslp_pre[(TC0_lat_npy-radius_mslp_npy):(TC0_lat_npy+radius_mslp_npy), (TC0_lon_npy-radius_mslp_npy):(TC0_lon_npy+radius_mslp_npy)]


        
        #vortivity
        if i>0:
            vortivity_850 = metpy.calc.vorticity( u_wind_850 , v_wind_850, dx=dx, dy=dy)
    
        print(mslp_pre_data.shape)
        coordinates = peak_local_max(-mslp_pre_data, min_distance=5)
        if len(coordinates)==0:
            print("Can't find TC")
            #
            # TC_location.append(TC_location[-1])
            # MSL_location.append(MSL_location[-1])
            #if cmp_wind:
            # wind_speed_max.append(wind_speed_max[-1])
            if is_wait_genesis and (genesis_flag==0):

                
                if cmp_wind:
                    speed = max_wind(wind10_pre, 90.0 - TC0_lat, TC0_lon, radius_wind=radius_wind, resulation=resulation_use)
                 
                    wind_speed_max.append(speed)
                
                TC_location.append([90.0 - TC0_lat, TC0_lon])
                MSL_location.append(mslp_pre[TC0_lat_npy, TC0_lon_npy])
                continue
            empty_time +=1

            if empty_time >= empty_time_threshold:
                print("no more TCs")
                break
        else:
          
            warm_point = []
            for point in coordinates:
                point_lat_npy = TC0_lat_npy-radius_mslp_npy+point[0]
                point_lat = point_lat_npy*resulation_use
                point_lon_npy = TC0_lon_npy-radius_mslp_npy+point[1]
                point_lon = point_lon_npy*resulation_use
                print(90.0 - point_lat, point_lon)
                if i>0:
                    if is_TC(point_lat, point_lon, vortivity_850, wind10_pre=wind10_pre, lsm=lsm, thickness=thickness, \
                            cmp_thickness=cmp_thickness, cmp_lsm=cmp_lsm, cmp_wind=cmp_wind, region=region, resulation=resulation_use):
                        warm_point.append(point)
                else:
                    warm_point.append(point)
            if len(warm_point) != 0:
                empty_time = 0

                best_point = coordinates[0]
                for j in range(len(coordinates)):
                    point = coordinates[j]
                   
                    point_lat_npy = TC0_lat_npy-radius_mslp_npy+point[0]
                    point_lat = point_lat_npy*resulation_use
                    point_lon_npy = TC0_lon_npy-radius_mslp_npy+point[1]
                    point_lon = point_lon_npy*resulation_use
                    print("[{}, {}]".format(90.0-point_lat, point_lon))

                    dis = dis_point(point=point, center=[mslp_pre_data.shape[0]/2, mslp_pre_data.shape[1]/2])
                    best_dis = dis_point(point=best_point, center=[mslp_pre_data.shape[0]/2, mslp_pre_data.shape[1]/2])
            

                    msl_point = mslp_pre_data[point[0]][point[1]]
                    best_msl_point = mslp_pre_data[best_point[0]][best_point[1]]
                    if msl_point<best_msl_point:
                        best_point = point
                    
                    # if dis<best_dis:
                    #     best_point = point
                    
                
                # print(coordinates)
                # print([coordinates[0]])
                print("===============================")
            
                print(mslp_pre_data[best_point[0]][best_point[1]])
              
    

                TC_lat_npy = TC0_lat_npy-radius_mslp_npy+best_point[0]
                TC_lat = TC_lat_npy*resulation_use
                TC_lon_npy = TC0_lon_npy-radius_mslp_npy+best_point[1]
                TC_lon = TC_lon_npy*resulation_use
                print(mslp_pre[TC_lat_npy, TC_lon_npy])
                print("[{}, {}]".format(90.0-TC_lat, TC_lon))

                
                
                if cmp_wind:
                    speed = max_wind(wind10_pre, 90.0 - TC_lat, TC_lon, radius_wind=radius_wind, resulation=resulation_use)
         
                    wind_speed_max.append(speed)
                
                TC_location.append([90.0 - TC_lat, TC_lon])
                MSL_location.append(mslp_pre[TC_lat_npy, TC_lon_npy])
                if i ==0:
                    TC0_lat_npy = round(TC_lat_npy*init_resulation/resulation)
                    TC0_lon_npy = round(TC_lon_npy*init_resulation/resulation)
                else:
                    TC0_lat_npy = TC_lat_npy
                    TC0_lon_npy = TC_lon_npy
                if i>0 and is_wait_genesis:
                    genesis_flag=1
            else:
           
                print("Can't find TC")
            
                if is_wait_genesis and (genesis_flag==0):
                    if cmp_wind:
                        speed = max_wind(wind10_pre, 90.0 - TC0_lat, TC0_lon, radius_wind=radius_wind, resulation=resulation_use)
                       
                        wind_speed_max.append(speed)
                    
                    TC_location.append([90.0 - TC0_lat, TC0_lon])
                    MSL_location.append(mslp_pre[TC0_lat_npy, TC0_lon_npy])
                    continue
                empty_time +=1
                if empty_time >= empty_time_threshold:
                    print("no more TCs")
                    break
     
        print("===============================")
        

    if cmp_wind:
        return TC_location, MSL_location, wind_speed_max
    else:
        return TC_location, MSL_location
