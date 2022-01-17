# -*- coding: utf-8 -*-
"""
Created by Simon Melgaard and Kamilla Andersen

This script is meant for importing the data for the AHU from TMV23, as well as performing the KPI assessments from the CLIMA 2022 article.
"""
#%%
#Import packages
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
import datetime
import matplotlib.cm as cm
import matplotlib.font_manager
import math
import statistics
import sys
from scipy.optimize import curve_fit
from pytz import timezone

from Python_scripts.AHU.AHU import AHU_assessment
from Python_scripts.AHU.Air_flow_calculation_over_fan import calc_air_flow

#%%  #####Declare variables######################################

path_main = format(os.getcwd())

start_year = 2021
start_month = 8
start_day = 1
start_hour = 0
start_minute = 0
start_time = datetime.datetime(year=start_year,month=start_month,day=start_day,hour=start_hour,minute=start_minute)
del start_year
del start_month
del start_day
del start_hour
del start_minute

end_year = 2021
end_month = 12
end_day = 23
end_hour = 0
end_minute = 0
end_time = datetime.datetime(year=end_year,month=end_month,day=end_day,hour=end_hour,minute=end_minute)
del end_year
del end_month
del end_day
del end_hour
del end_minute

#%%   ########### Import data from the AHU file ####################################
# Do not change

# For explanation of all the variables, see the log map (Log_map_TMV23.xlsx)

KOMF01_df=pd.read_excel(path_main + "\\Combined_data\\KOMF01_combined.xlsx")

########### Import data from energykey files ####################################

# Contains the points for the fans datasheet
KOMF01_FAN_df=pd.read_excel(path_main+"\\Combined_fan_curve.xlsx",sheet_name="Columns",usecols="A:D")

# Contains the electric energy use for the AHU
KOMF01_EL_df=pd.read_excel(path_main+"\\Energy Key\\KOMF01\\KOMF01_EL_hour_2017-2021.xlsx")
# Fix summertime offset problems for the EnergyKey data, by setting winter time to be the standard time
tz = timezone('Europe/Copenhagen')
for i in range(len(KOMF01_EL_df)):
    offset=tz.dst(KOMF01_EL_df['Periode'][i],is_dst=True)
    KOMF01_EL_df['Periode'][i]=KOMF01_EL_df['Periode'][i]-offset
    if i>0:
        if (datetime.timedelta(seconds=3600)==tz.dst(KOMF01_EL_df['Periode'][i-1],is_dst=True))==True and (datetime.timedelta(seconds=3600)==tz.dst(KOMF01_EL_df['Periode'][i],is_dst=True))==False:
            offset_prev=tz.dst(KOMF01_EL_df['Periode'][i-1],is_dst=True)
            KOMF01_EL_df['Periode'][i-1]=KOMF01_EL_df['Periode'][i-1]+offset_prev

#%% ######################## Analysis for the AHU and all components #########################################
# Set which modules to run, to run several modules at once, input as "x.x.x","x.x.x"...
modules_to_run=["0.0.0", "1.0.0", "1.1.0"]

# Contains specs for the AHu and components
specs={"spec_sheet_fan" : KOMF01_FAN_df,
       "spec_flow_fan" : [10000,10000],                 #m^3/h
       "spec_pressure_diff_fan" : [400,386],            #Pa
       "spec_rot_speed_fan" : [1242,1232],              #RPM
       "spec_max_rot_speed_fan" : [1380],               #RPM
       "spec_efficiency_HE_ref" : [86],                 #%
       "spec_flow_HE_ref" : [10000],                    #m^3/h
       "spec_temp_inlet_before_HE_HE_ref" : [-15],      #degree C
       "spec_temp_inlet_after_HE_HE_ref" : [15.1],      #degree C
       "spec_temp_exhaust_before_HE_HE_ref" : [20],     #degree C
       "spec_temp_exhaust_after_HE_HE_ref" : [-10.1],   #degree C
       "spec_rel_hum_inlet_before_HE_HE_ref" : [90],    #%
       "spec_rel_hum_inlet_after_HE_HE_ref" : [25.6],   #%
       "spec_rel_hum_exhaust_before_HE_HE_ref" : [25],  #%
       "spec_rel_hum_exhaust_after_HE_HE_ref" : [97.9]  #%     
       }


# Runs the assessment for the selected modules in the chosen time period
AHU_ass = AHU_assessment(modules_to_run,KOMF01_df, KOMF01_EL_df, start_time, end_time, specs)


