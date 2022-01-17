# -*- coding: utf-8 -*-
"""
Created by Simon Melgaard and Kamilla Andersen

This script contains the different modules called by the main script (Data_processing.py)
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import statistics
import math
from .Air_flow_calculation_over_fan import calc_air_flow

def AHU_assessment(module_to_run, meas_BMS_data=0, meas_EK_data=0, start_time=datetime.datetime(1,1,1), end_time=datetime.datetime(1,1,1), specs=dict()):
    """
    

    Parameters
    ----------
    module_to_run : Float
        A list containing the index to run, to determine which part should be run (if the index has # before it, it is not done yet)
        Index 0.x.x is for the AHU
        Index 1.x.x is for the rotary heat exchanger
        Index 2.x.x is for the fan(s)
        Index 3.x.x is for the heating coil
        Index 4.x.x is for the damper(s)
                                        
        Index 0.0.0 = (Historical) Specific AHU power comparred to flow (Figure 8 in the CLIMA 2022 article)
        Index 1.0.0 = (Historical) Heat recovery comparred to Outdoor temperature and rotation speed (Figure 9 in the CLIMA 2022 article)
        Index 1.1.0 = (Reference) Heat exchanger efficiency under conditions near the reference (manufacturers data) (Figure 10 in the CLIMA 2022 article)
        

    """

    # Time
    meas_time=meas_BMS_data['Time stamp']
    EK_AHU_time=meas_EK_data['Periode']
    
    # External data
    meas_ext_temp=meas_BMS_data['Supply_temperature_curve_X']
    
    # Control input
    meas_supply_temp_set_point=meas_BMS_data['Supply_temperature_set-point']
    
    # Temperature
    meas_temp_intake=meas_BMS_data['Intake_temperature']                        # Temperature of the intake before the HE
    meas_temp_after_HE=meas_BMS_data['Temperature_after_HE_before_HC']          # Temperature of the intake after the HE (before the heating coil)
    meas_temp_extract=meas_BMS_data['Extraction_temperature']                   # Temperature of the extraction before the HE
    meas_temp_exhaust=meas_BMS_data['Exhaust_temperature']                      # Temperature of the exhaust after the HE
    
    # AHU
    EK_AHU_el=meas_EK_data['Energi']
    
    # Fan
    meas_exhaust_flow_signal=meas_BMS_data['Exhaust_fan_signal']
    meas_intake_flow_signal=meas_BMS_data['Supply_fan_signal']
    
    # Fan specs
    spec_fan_flow=specs["spec_sheet_fan"]['Flow [m3/h]']
    spec_fan_pressure=specs["spec_sheet_fan"]['pressure_difference [Pa]']
    spec_fan_rot_speed=specs["spec_sheet_fan"]['Rotation_speed [rpm]']
    spec_fan_max_rot_speed=specs["spec_max_rot_speed_fan"]
    spec_pressure_diff=specs["spec_pressure_diff_fan"]
    
    
    # Rotary heat exchanger
    meas_eff = meas_BMS_data['HE_efficiency']  
    meas_HE_signal = meas_BMS_data['HE_signal']
    
    # Rotary heat exchanger specs
    
    spec_HE_efficiency=specs["spec_efficiency_HE_ref"]
    spec_HE_flow=specs["spec_flow_HE_ref"]
    spec_HE_temp_inlet_before=specs["spec_temp_inlet_before_HE_HE_ref"]
    spec_HE_temp_inlet_after=specs["spec_temp_inlet_after_HE_HE_ref"]
    spec_HE_temp_exhaust_before=specs["spec_temp_exhaust_before_HE_HE_ref"]
    spec_HE_temp_exhaust_after=specs["spec_temp_exhaust_after_HE_HE_ref"]

    

    if "0.0.0" in module_to_run:
        begin_time=datetime.datetime.now()
        # Calculate the intake and exhaust flow from fan signal
        flow_intake = calc_air_flow(spec_fan_rot_speed,spec_fan_pressure,spec_fan_flow,meas_intake_flow_signal,spec_fan_max_rot_speed,spec_pressure_diff[0])
        flow_exhaust = calc_air_flow(spec_fan_rot_speed,spec_fan_pressure,spec_fan_flow,meas_exhaust_flow_signal,spec_fan_max_rot_speed,spec_pressure_diff[1])
        
        
        
        
        # Filter out the non physical values (due to uncertainty in the model for the low signal values)
        flow_intake[meas_intake_flow_signal==0]=0
        flow_intake[flow_intake<0]=0
        flow_exhaust[meas_exhaust_flow_signal==0]=0
        flow_exhaust[flow_exhaust<0]=0
        
        # Resample the flow, to get the same time step as the electricity use (hourly values)
        flow=pd.concat([meas_time,flow_intake,flow_exhaust], axis=1)
        flow.columns=['Time_stamp','flow_intake','flow_exhaust']
        flow=flow.set_index('Time_stamp')
        flow_1_hour=flow.resample(datetime.timedelta(hours=1)).sum()
        flow_1_hour.reset_index(inplace=True)

        # The input values for the comparrison
        flow_exhaust=flow_1_hour['flow_exhaust']
        flow_intake=flow_1_hour['flow_intake']
        flow_time=flow_1_hour['Time_stamp']
        flow_time_intake=flow_time
        flow_time_exhaust=flow_time
        electricity_time = EK_AHU_time
        electricity = EK_AHU_el
             
        # Crop the time series, so all have the same length (start to end)
        flow_exhaust=flow_exhaust[flow_time>=start_time]
        flow_intake=flow_intake[flow_time>=start_time]
        flow_time_exhaust=flow_time_exhaust[flow_time>=start_time]
        flow_time_intake=flow_time_intake[flow_time>=start_time]
        flow_time=flow_time[flow_time>=start_time]

        flow_exhaust=flow_exhaust[flow_time<end_time]
        flow_intake=flow_intake[flow_time<end_time]
        flow_time_exhaust=flow_time_exhaust[flow_time<end_time]
        flow_time_intake=flow_time_intake[flow_time<end_time]
        flow_time=flow_time[flow_time<end_time]
        
        electricity=electricity[electricity_time>=start_time]
        electricity_time=electricity_time[electricity_time>=start_time]
        electricity=electricity[electricity_time<end_time]
        electricity_time=electricity_time[electricity_time<end_time]
        
        # Reset the index for each series
        flow_exhaust.reset_index(drop=True,inplace=True)
        flow_intake.reset_index(drop=True,inplace=True)
        flow_time_exhaust.reset_index(drop=True,inplace=True)
        flow_time_intake.reset_index(drop=True,inplace=True)
        flow_time.reset_index(drop=True,inplace=True)
        electricity.reset_index(drop=True,inplace=True)
        electricity_time.reset_index(drop=True,inplace=True)
        
        print("Data created after ",datetime.datetime.now()-begin_time)
        
        # Calculates the Specific AHU power (electricity) and total air flow (both intake and exhaust)
        SAP_1=[]
        flow_total_1=[]
        for i in range(len(electricity_time)):
            flow_total_temp=flow_intake[i]+flow_exhaust[i]
            if flow_total_temp==0:
                SAP_temp=0
            else:
                SAP_temp=electricity[i]/((flow_intake[i]+flow_exhaust[i])/3600)
        

            SAP_1.append(SAP_temp)
            flow_total_1.append(flow_total_temp)
        
        SAP_2=pd.DataFrame(data=SAP_1)
        flow_total_2=pd.DataFrame(data=flow_total_1)
        SAP=pd.concat([flow_time, SAP_2, flow_total_2], axis=1)
        SAP.columns=['Time','Specific AHU power [kW/(m^3/s)]','Total flow [m^3/h]']
               
        print("SAP created after ",datetime.datetime.now()-begin_time)
        
        
        
        
        
        # Plots the cleaned data with histogram limits and no intervals
        def scatter_hist(x, y, cx, cx_histx, cx_histy):
            # no labels
            cx_histx.tick_params(axis="x", labelbottom=False)
            cx_histy.tick_params(axis="y", labelleft=False)
            print("Scatter 1 after ",datetime.datetime.now()-begin_time)
            # the scatter plot:
            cx.scatter(x, y,color='#41B5B3',alpha=0.8,s=3)
            print("Scatter 2 after ",datetime.datetime.now()-begin_time)
            # now determine nice limits by hand:
            binwidth_x = 1000
            binwidth_y = 0.1
            #xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
            lim_x = (int(np.max(np.abs(x))/binwidth_x) + 1) * binwidth_x
            lim_y = (int(np.max(np.abs(y))/binwidth_y) + 1) * binwidth_y
            print("Scatter 3 after ",datetime.datetime.now()-begin_time)
            bins_x = np.arange(-lim_x, lim_x + binwidth_x, binwidth_x)
            bins_y = np.arange(-lim_y, lim_y + binwidth_y, binwidth_y)
            print("Scatter 4 after ",datetime.datetime.now()-begin_time)
            cx_histx.hist(x, bins=bins_x,color='#41B5B3')
            cx_histx.set_yticklabels(cx_histx.get_yticks().astype(int), size=12)
            print("Scatter 5 after ",datetime.datetime.now()-begin_time)
            cx_histy.hist(y, bins=bins_y, orientation='horizontal',color='#41B5B3')
            cx_histy.set_xticklabels(cx_histy.get_xticks().astype(int), size=12)
            print("Scatter 6 after ",datetime.datetime.now()-begin_time)
            
        # start with a square Figure
        fig1 = plt.figure(figsize=(8, 8))

        # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
        # the size of the marginal axes and the main axes in both directions.
        # Also adjust the subplot parameters for a square plot.
        gs = fig1.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),
                      left=0.1, right=0.9, bottom=0.15, top=0.9,
                      wspace=0.1, hspace=0.1) # if xticks are rotated, bottom must be 0.1.5 otherwise 0.1

        cx = fig1.add_subplot(gs[1, 0])
        cx_histx = fig1.add_subplot(gs[0, 0], sharex=cx)
        cx_histy = fig1.add_subplot(gs[1, 1], sharey=cx)
        
        print("Plot initialized after ",datetime.datetime.now()-begin_time)
        
        # Remove 0 flow points for the scatter plot
        SAP_scatter=SAP['Specific AHU power [kW/(m^3/s)]']
        flow_scatter=SAP['Total flow [m^3/h]']
        SAP_scatter=SAP_scatter[flow_scatter>0]
        flow_scatter=flow_scatter[flow_scatter>0]
        
        
        
        print("Scatter data created after ",datetime.datetime.now()-begin_time)
        
        # use the previously defined function
        scatter_hist(flow_scatter,SAP_scatter, cx, cx_histx, cx_histy) # plots historic data
        
        print("Scatter plot added after ",datetime.datetime.now()-begin_time)
        
        #cx.scatter(ext_temp_clean.iloc[new_day_start:new_day_end],meas_eff_clean.iloc[new_day_start:new_day_end],color='#85C552',alpha=1,s=10) # plots data for next day
        cx.set_xlim(0,20000)
        cx.set_ylim(0,3)
        cx.set_xticklabels(cx.get_xticks().astype(int), rotation = 90, size=12)
        cx.set_yticklabels(cx.get_yticks(), size=12)
        cx.grid(True)
        cx.set_xlabel('Total air flow [$m^3/h$]', fontsize=14, fontname='Cambria')
        cx.set_ylabel('Specific AHU power [$kW/(m^3/s)$]', fontsize=14, fontname='Cambria')
        cx_histx.set_title('AHU\nSpecific AHU power with historical benchmark', fontsize=20, fontname='Cambria')
        fig1.savefig("CLIMA22_SAP_VS_air_flow.svg",dpi=600)
        
        print("Figure plotted after ",datetime.datetime.now()-begin_time)
        
        
        
     
    
    if "1.0.0" in module_to_run:
        # Plots the data without cleaning
        start=min(np.where(meas_time>start_time)[0])
        end=max(np.where(meas_time<end_time)[0])
        
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.scatter(meas_ext_temp.iloc[start:end],meas_eff.iloc[start:end],s=5,alpha=0.1)
        ax.set_xlim(-10,30)
        ax.set_ylim(0,100)
        ax.set_xlabel('Outdoor air temperature ['+u"\N{DEGREE SIGN}"+'C]')
        ax.set_ylabel('Heat recovery [%]')
        ax.set_title('Outdoor air temperature VS heat recovery')
        
        
        # Generates the clean data, by removing the 0 % HE efficiency and 0% HE signal values
        # because these are in the start and stop fases, which are not relevant for this KPI        
        meas_ext_temp_clean=meas_ext_temp[meas_HE_signal>0]
        meas_eff_clean=meas_eff[meas_HE_signal>0]
        meas_time_clean=meas_time[meas_HE_signal>0]
        meas_HE_signal_clean=meas_HE_signal[meas_HE_signal>0]
        
        
        meas_ext_temp_clean=meas_ext_temp_clean[meas_eff_clean>0]
        meas_time_clean=meas_time_clean[meas_eff_clean>0]
        meas_HE_signal_clean=meas_HE_signal_clean[meas_eff_clean>0]
        meas_eff_clean=meas_eff_clean[meas_eff_clean>0]
            
        start_clean=min(np.where(meas_time_clean>start_time)[0])
        end_clean=max(np.where(meas_time_clean<end_time)[0])
        
        fig2=plt.figure()
        bx=fig2.add_subplot(111)
        bx.scatter(meas_ext_temp_clean.iloc[start_clean:end_clean],meas_eff_clean.iloc[start_clean:end_clean],s=5,color='k',alpha=0.1,label='measured data')
        bx.set_xlim(-10,30)
        bx.set_ylim(0,100)
        bx.set_xlabel('Outdoor air temperature ['+u"\N{DEGREE SIGN}"+'C]')
        bx.set_ylabel('Heat recovery [%]')
        bx.set_title('Outdoor air temperature VS heat recovery')
        
        # Calculates the IQR curves for each area of the data
        step=1
        range_start=math.floor(min(meas_ext_temp_clean))+step
        range_end=math.floor(max(meas_ext_temp_clean))+step
        meas_eff_median=[]
        meas_eff_IQR=[]
        meas_eff_IQR_q1=[]
        meas_eff_IQR_q3=[]
        meas_eff_length=[]
        for out_temp in range(range_start,range_end,step): 
            meas_eff_temporary=meas_eff_clean[(meas_ext_temp_clean>out_temp-step)&(meas_ext_temp_clean<=out_temp)]
            meas_time_temporary=meas_time_clean[(meas_ext_temp_clean>out_temp-step)&(meas_ext_temp_clean<=out_temp)]
            try:
                start_temporary=min(np.where(meas_time_temporary>start_time)[0])
                end_temporary=max(np.where(meas_time_temporary<end_time)[0])
            except:
                meas_eff_median.append(0)
                meas_eff_IQR.append(0)
                meas_eff_IQR_q1.append(0)
                meas_eff_IQR_q3.append(0)
                meas_eff_length.append(0)
            else:
                meas_eff_temporary=meas_eff_temporary.iloc[start_temporary:end_temporary]
                if len(meas_eff_temporary)==0:
                    meas_eff_median.append(0)
                    meas_eff_IQR.append(0)
                    meas_eff_IQR_q1.append(0)
                    meas_eff_IQR_q3.append(0)  
                    meas_eff_length.append(0)
                else:
                    meas_eff_temporary_median=statistics.median(meas_eff_temporary)
                    meas_eff_temporary_IQR_q1=np.percentile(meas_eff_temporary,25)
                    meas_eff_temporary_IQR_q3=np.percentile(meas_eff_temporary,75)
                    meas_eff_temporary_IQR=meas_eff_temporary_IQR_q3-meas_eff_temporary_IQR_q1
                
                    meas_eff_median.append(meas_eff_temporary_median)
                    meas_eff_IQR.append(meas_eff_temporary_IQR)
                    meas_eff_IQR_q1.append(meas_eff_temporary_IQR_q1)
                    meas_eff_IQR_q3.append(meas_eff_temporary_IQR_q3)
                    meas_eff_length.append(len(meas_eff_temporary))

        
        
        meas_eff_top_lim_1_5_IQR=[1.5*x for x in meas_eff_IQR]
        meas_eff_top_lim_1_5_IQR=[sum(x) for x in zip(meas_eff_IQR_q3, meas_eff_top_lim_1_5_IQR)]
        meas_eff_bot_lim_1_5_IQR=[-1.5*x for x in meas_eff_IQR]
        meas_eff_bot_lim_1_5_IQR=[sum(x) for x in zip(meas_eff_IQR_q1, meas_eff_bot_lim_1_5_IQR)]
        
        
        data_points_for_stats=range(range_start-step,range_end,step)
        data_points_for_stats=[val for val in data_points_for_stats for _ in (0, 1)]
        data_points_for_stats.pop()
        data_points_for_stats.pop(0)
        
        
        # Plots the IQR limits (and median) onto the datapoints, to make it clear if any potential outliers exist.
        bx.fill_between(data_points_for_stats,[val for val in meas_eff_bot_lim_1_5_IQR for _ in (0, 1)],[val for val in meas_eff_top_lim_1_5_IQR for _ in (0, 1)],color='b',alpha=0.5, label='Area where the efficiency is expected to be')
        bx.plot(data_points_for_stats,[val for val in meas_eff_median for _ in (0, 1)],color='r', label='Median of the data')
        bx.legend(loc='lower right')
        
        
        # Plots the cleaned data with histogram limits and no intervals
        def scatter_hist(x, y, cx, cx_histx, cx_histy):
            # no labels
            cx_histx.tick_params(axis="x", labelbottom=False)
            cx_histy.tick_params(axis="y", labelleft=False)
            
            # the scatter plot:
            cx.scatter(x, y,color='#41B5B3',alpha=0.3,s=3)

            # now determine nice limits by hand:
            binwidth = 1
            xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
            lim = (int(xymax/binwidth) + 1) * binwidth

            bins = np.arange(-lim, lim + binwidth, binwidth)
            cx_histx.hist(x, bins=bins,color='#41B5B3')
            cx_histx.set_yticklabels(cx_histx.get_yticks().astype(int), size=8)
            cx_histy.hist(y, bins=bins, orientation='horizontal',color='#41B5B3')
            cx_histy.set_xticklabels(cx_histy.get_xticks().astype(int), size=8)
            
        def scatter_hist_multi(x, y, cx, cx_histx, cx_histy, color_var):
            # no labels
            cx_histx.tick_params(axis="x", labelbottom=False)
            cx_histy.tick_params(axis="y", labelleft=False)
            
            # the scatter plot:
            cx.scatter(x, y,color=color_var, cmap=plt.cm.coolwarm ,alpha=0.3,s=3)

            # now determine nice limits by hand:
            binwidth = 1
            xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
            lim = (int(xymax/binwidth) + 1) * binwidth

            bins = np.arange(-lim, lim + binwidth, binwidth)
            cx_histx.hist(x, bins=bins,color='#41B5B3')
            cx_histx.set_yticklabels(cx_histx.get_yticks().astype(int), size=12)
            cx_histy.hist(y, bins=bins, orientation='horizontal',color='#41B5B3')
            cx_histy.set_xticklabels(cx_histy.get_xticks().astype(int), size=12)
            
            
        new_day_start=min(np.where(meas_time_clean>end_time)[0])
        new_day_end=max(np.where(meas_time_clean<end_time+datetime.timedelta(days=1))[0])
        

        
        what_to_plot=2

        # start with a square Figure
        fig3 = plt.figure(figsize=(8,8))

        # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
        # the size of the marginal axes and the main axes in both directions.
        # Also adjust the subplot parameters for a square plot.
        gs = fig3.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.1, hspace=0.1)

        cx = fig3.add_subplot(gs[1, 0])
        cx_histx = fig3.add_subplot(gs[0, 0], sharex=cx)
        cx_histy = fig3.add_subplot(gs[1, 1], sharey=cx)
    
        # use the previously defined function
        if what_to_plot==1:
            scatter_hist(meas_ext_temp_clean.iloc[start_clean:end_clean],meas_eff_clean.iloc[start_clean:end_clean], cx, cx_histx, cx_histy) # plots historic data
            #cx.scatter(meas_ext_temp_clean.iloc[new_day_start:new_day_end],meas_eff_clean.iloc[new_day_start:new_day_end],color='#85C552',alpha=1,s=10) # plots data for next day
        
        elif what_to_plot==2:
            colors = plt.cm.coolwarm(meas_HE_signal_clean.iloc[start_clean:end_clean]/100)
            scatter_hist_multi(meas_ext_temp_clean.iloc[start_clean:end_clean],meas_eff_clean.iloc[start_clean:end_clean], cx, cx_histx, cx_histy, colors) # plots historic data
            cax=fig3.add_subplot(gs[0, 1],aspect=5, anchor='W')
            colorbar = fig3.colorbar(plt.cm.ScalarMappable(norm=None, cmap=plt.cm.coolwarm),cax=cax)
            colorbar_yticks=colorbar.ax.get_yticks()*100
            colorbar.ax.set_yticklabels(colorbar_yticks.astype(int), size=8)
            colorbar.ax.set_ylabel('Rotation speed [%]', fontsize=10, fontname='Cambria')
            
        cx.set_xlim(-5,20)
        cx.set_ylim(0,100)
        cx.set_xticklabels(cx.get_xticks().astype(int), size=12)
        cx.set_yticklabels(cx.get_yticks().astype(int), size=12)
        cx.grid(True)
        cx.set_xlabel('Outdoor air temperature ['+u"\N{DEGREE SIGN}"+'C]', fontsize=14, fontname='Cambria')
        cx.set_ylabel('Heat recovery [%]', fontsize=14, fontname='Cambria')
        cx_histx.set_title('Rotary heat exchanger\nHeat recovery with historical benchmark', fontsize=20, fontname='Cambria')
        
        if what_to_plot==1:
            fig3.savefig("CLIMA22_OAT_VS_eff.svg",dpi=600)
        
        elif what_to_plot==2:
            fig3.savefig("CLIMA22_OAT+Rot_speed_VS_eff.svg",dpi=600)
        
        
        # Ends the module and returns a dataframe containing the analysis results
        return pd.DataFrame(np.column_stack([meas_eff_median, meas_eff_IQR, meas_eff_IQR_q1, meas_eff_IQR_q3, meas_eff_bot_lim_1_5_IQR, meas_eff_top_lim_1_5_IQR, meas_eff_length]), columns=['Median', 'IQR', 'Q1', 'Q3', 'Bottom limit', 'Top limit', 'Number of datapoints'])
        
    if "1.1.0" in module_to_run:
        
        
        # Clean data to ensure only max rotation of HE
        meas_temp_intake_clean=meas_temp_intake[meas_HE_signal==100]
        meas_temp_after_HE_clean=meas_temp_after_HE[meas_HE_signal==100]
        meas_temp_extract_clean=meas_temp_extract[meas_HE_signal==100]
        meas_temp_exhaust_clean=meas_temp_exhaust[meas_HE_signal==100]
        meas_intake_flow_signal_clean=meas_intake_flow_signal[meas_HE_signal==100]
        meas_exhaust_flow_signal_clean=meas_exhaust_flow_signal[meas_HE_signal==100]
        meas_eff_clean=meas_eff[meas_HE_signal==100]        
        
        # Clean data to only include measured efficiency above 0        
        meas_temp_intake_clean=meas_temp_intake_clean[meas_eff_clean>0]
        meas_temp_after_HE_clean=meas_temp_after_HE_clean[meas_eff_clean>0]
        meas_temp_extract_clean=meas_temp_extract_clean[meas_eff_clean>0]
        meas_temp_exhaust_clean=meas_temp_exhaust_clean[meas_eff_clean>0]
        meas_intake_flow_signal_clean=meas_intake_flow_signal_clean[meas_eff_clean>0]
        meas_exhaust_flow_signal_clean=meas_exhaust_flow_signal_clean[meas_eff_clean>0]
        meas_eff_clean=meas_eff_clean[meas_eff_clean>0]
        
        # Clean data to ensure supply temperature before the HE is within the range of the manufacturers dimension point
        HE_supply_temp_range=15 # Points must have an intake temperature no further than this number from the dimensioning point
        HE_supply_temp_range_upper=spec_HE_temp_inlet_before[0]+HE_supply_temp_range
        HE_supply_temp_range_lower=spec_HE_temp_inlet_before[0]-HE_supply_temp_range
        
        meas_temp_after_HE_clean=meas_temp_after_HE_clean[meas_temp_intake_clean>HE_supply_temp_range_lower]
        meas_temp_extract_clean=meas_temp_extract_clean[meas_temp_intake_clean>HE_supply_temp_range_lower]
        meas_temp_exhaust_clean=meas_temp_exhaust_clean[meas_temp_intake_clean>HE_supply_temp_range_lower]
        meas_eff_clean=meas_eff_clean[meas_temp_intake_clean>HE_supply_temp_range_lower]
        meas_intake_flow_signal_clean=meas_intake_flow_signal_clean[meas_temp_intake_clean>HE_supply_temp_range_lower]
        meas_exhaust_flow_signal_clean=meas_exhaust_flow_signal_clean[meas_temp_intake_clean>HE_supply_temp_range_lower]
        meas_temp_intake_clean=meas_temp_intake_clean[meas_temp_intake_clean>HE_supply_temp_range_lower]
        
        meas_temp_after_HE_clean=meas_temp_after_HE_clean[meas_temp_intake_clean<HE_supply_temp_range_upper]
        meas_temp_extract_clean=meas_temp_extract_clean[meas_temp_intake_clean<HE_supply_temp_range_upper]
        meas_temp_exhaust_clean=meas_temp_exhaust_clean[meas_temp_intake_clean<HE_supply_temp_range_upper]
        meas_eff_clean=meas_eff_clean[meas_temp_intake_clean<HE_supply_temp_range_upper]
        meas_intake_flow_signal_clean=meas_intake_flow_signal_clean[meas_temp_intake_clean<HE_supply_temp_range_upper]
        meas_exhaust_flow_signal_clean=meas_exhaust_flow_signal_clean[meas_temp_intake_clean<HE_supply_temp_range_upper]
        meas_temp_intake_clean=meas_temp_intake_clean[meas_temp_intake_clean<HE_supply_temp_range_upper]    
        
        # Clean data to ensure extraction temperature before the HE is within the range of the manufacturers dimension point
        HE_extraction_temp_range=3 # Points must have an extraction temperature no further than this number from the dimensioning point
        HE_extraction_temp_range_upper=spec_HE_temp_exhaust_before[0]+HE_extraction_temp_range
        HE_extraction_temp_range_lower=spec_HE_temp_exhaust_before[0]-HE_extraction_temp_range
        
        meas_temp_intake_clean=meas_temp_intake_clean[meas_temp_extract_clean>HE_extraction_temp_range_lower]
        meas_temp_after_HE_clean=meas_temp_after_HE_clean[meas_temp_extract_clean>HE_extraction_temp_range_lower]
        meas_temp_exhaust_clean=meas_temp_exhaust_clean[meas_temp_extract_clean>HE_extraction_temp_range_lower]
        meas_eff_clean=meas_eff_clean[meas_temp_extract_clean>HE_extraction_temp_range_lower]
        meas_intake_flow_signal_clean=meas_intake_flow_signal_clean[meas_temp_extract_clean>HE_extraction_temp_range_lower]
        meas_exhaust_flow_signal_clean=meas_exhaust_flow_signal_clean[meas_temp_extract_clean>HE_extraction_temp_range_lower]
        meas_temp_extract_clean=meas_temp_extract_clean[meas_temp_extract_clean>HE_extraction_temp_range_lower]
        
        meas_temp_intake_clean=meas_temp_intake_clean[meas_temp_extract_clean<HE_extraction_temp_range_upper]
        meas_temp_after_HE_clean=meas_temp_after_HE_clean[meas_temp_extract_clean<HE_extraction_temp_range_upper]
        meas_temp_exhaust_clean=meas_temp_exhaust_clean[meas_temp_extract_clean<HE_extraction_temp_range_upper]
        meas_eff_clean=meas_eff_clean[meas_temp_extract_clean<HE_extraction_temp_range_upper]
        meas_intake_flow_signal_clean=meas_intake_flow_signal_clean[meas_temp_extract_clean<HE_extraction_temp_range_upper]
        meas_exhaust_flow_signal_clean=meas_exhaust_flow_signal_clean[meas_temp_extract_clean<HE_extraction_temp_range_upper]
        meas_temp_extract_clean=meas_temp_extract_clean[meas_temp_extract_clean<HE_extraction_temp_range_upper]
        
        
        
        
        
        
        # Calculate the intake and extraction flow (and recalculate it from m^3/minute to m^3/hour)
        #flow_intake = calc_air_flow(spec_fan_rot_speed,spec_fan_pressure,spec_fan_flow,meas_intake_flow_signal_clean,spec_fan_max_rot_speed,spec_pressure_diff[0])*60
        #flow_exhaust = calc_air_flow(spec_fan_rot_speed,spec_fan_pressure,spec_fan_flow,meas_exhaust_flow_signal_clean,spec_fan_max_rot_speed,spec_pressure_diff[1])*60


        
        # Filter values outside the flow range
        #HE_flow_range=10000 # Points must have a flow no further than this number from the dimensioning point
        #HE_flow_range_upper=spec_HE_flow[0]+HE_flow_range
        #HE_flow_range_lower=spec_HE_flow[0]-HE_flow_range
        
        #meas_temp_intake_clean=meas_temp_intake_clean[flow_exhaust>HE_flow_range_lower]
        #meas_temp_after_HE_clean=meas_temp_after_HE_clean[flow_exhaust>HE_flow_range_lower]
        #meas_temp_exhaust_clean=meas_temp_exhaust_clean[flow_exhaust>HE_flow_range_lower]
        #meas_eff_clean=meas_eff_clean[flow_exhaust>HE_flow_range_lower]
        #meas_temp_extract_clean=meas_temp_extract_clean[flow_exhaust>HE_flow_range_lower]         
        #flow_intake_clean=flow_intake[flow_exhaust>HE_flow_range_lower]
        #flow_exhaust_clean=flow_exhaust[flow_exhaust>HE_flow_range_lower]
       
        #meas_temp_intake_clean=meas_temp_intake_clean[flow_exhaust<HE_flow_range_upper]
        #meas_temp_after_HE_clean=meas_temp_after_HE_clean[flow_exhaust<HE_flow_range_upper]
        #meas_temp_exhaust_clean=meas_temp_exhaust_clean[flow_exhaust<HE_flow_range_upper]
        #meas_eff_clean=meas_eff_clean[flow_exhaust<HE_flow_range_upper]
        #meas_temp_extract_clean=meas_temp_extract_clean[flow_exhaust<HE_flow_range_upper]         
        #flow_intake_clean=flow_intake_clean[flow_exhaust<HE_flow_range_upper]
        #flow_exhaust_clean=flow_exhaust_clean[flow_exhaust<HE_flow_range_upper]        
        
        #meas_temp_intake_clean=meas_temp_intake_clean[flow_intake>HE_flow_range_lower]
        #meas_temp_after_HE_clean=meas_temp_after_HE_clean[flow_intake>HE_flow_range_lower]
        #meas_temp_exhaust_clean=meas_temp_exhaust_clean[flow_intake>HE_flow_range_lower]
        #meas_eff_clean=meas_eff_clean[flow_intake>HE_flow_range_lower]
        #flow_exhaust_clean=flow_exhaust_clean[flow_intake>HE_flow_range_lower]        
        #flow_intake_clean=flow_intake_clean[flow_intake>HE_flow_range_lower]
        
        #meas_temp_intake_clean=meas_temp_intake_clean[flow_intake<HE_flow_range_upper]
        #meas_temp_after_HE_clean=meas_temp_after_HE_clean[flow_intake<HE_flow_range_upper]
        #meas_temp_exhaust_clean=meas_temp_exhaust_clean[flow_intake<HE_flow_range_upper]
        #meas_eff_clean=meas_eff_clean[flow_intake<HE_flow_range_upper]
        #flow_exhaust_clean=flow_exhaust_clean[flow_intake<HE_flow_range_upper]        
        #flow_intake_clean=flow_intake_clean[flow_intake<HE_flow_range_upper]       
        
        
        
        
        # Calculate the average value and the standard deviation of all the points in the range
        average_meas_eff_clean=np.mean(meas_eff_clean)
        std_meas_eff_clean=np.std(meas_eff_clean)
        average_intake_temp=np.mean(meas_temp_intake_clean)
        std_intake_temp=np.std(meas_temp_intake_clean)
        average_extraction_temp=np.mean(meas_temp_extract_clean)
        std_extraction_temp=np.std(meas_temp_extract_clean)
        
        print("Manufacturer temperature efficiency: "+str(spec_HE_efficiency[0])+"%\nAverage temperature efficiency for calculation: "+str(average_meas_eff_clean)+"%\nwith standard deviation: "+str(std_meas_eff_clean)+"%\nbased on "+str(len(meas_eff_clean))+" points\n")
        print("Manufacturer supply temp: "+str(spec_HE_temp_inlet_before[0])+u"\N{DEGREE SIGN}\n"+"Average supply temperature for calculation: "+str(average_intake_temp)+u"\N{DEGREE SIGN}\n"+"with standard deviation: "+str(std_intake_temp)+u"\N{DEGREE SIGN}\n")
        print("Manufacturer extraction temp: "+str(spec_HE_temp_exhaust_before[0])+u"\N{DEGREE SIGN}\n"+"Average extraction temperature for calculation: "+str(average_extraction_temp)+u"\N{DEGREE SIGN}\n"+"with standard deviation: "+str(std_extraction_temp)+u"\N{DEGREE SIGN}\n")
        
        # Make 3d plot with the text in it
        
        fig1 = plt.figure(figsize=(8,8))
        
        
        gs = fig1.add_gridspec(2,1,  height_ratios=(8, 1),
                      left=0, right=0.9, bottom=0, top=0.9,
                      wspace=0, hspace=0)

        cx = fig1.add_subplot(gs[0, 0],projection="3d")
        
        pos1=cx.get_position()
        pos2=[pos1.x0, pos1.y0+0.2,pos1.x1,pos1.y1+0.2]
        cx.set_position(pos2)
        
        
        cmap_min, cmap_max = min(meas_eff_clean),max(meas_eff_clean)
        cmap_min, cmap_max = min(cmap_min,spec_HE_efficiency[0]), max(cmap_max,spec_HE_efficiency[0])
        cmap_min, cmap_max = 76,88
        norm=plt.Normalize(cmap_min, cmap_max)
        scatter_measured=cx.scatter3D(meas_temp_intake_clean,meas_temp_extract_clean,meas_eff_clean,c=meas_eff_clean,cmap="brg",marker=".",s=50,norm=norm,alpha=1)
        scatter_manufacturer=cx.scatter3D(spec_HE_temp_inlet_before[0],spec_HE_temp_exhaust_before[0],spec_HE_efficiency[0],c=spec_HE_efficiency[0],cmap="brg",marker="*",s=100,norm=norm)
        
        colorbar = fig1.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="brg"),ax=cx, shrink=.4, pad=.1)
        colorbar.ax.set_yticklabels(["76%","78%","80%","82%","84%","86%","88%"], size=10)
        
    
        
        cx_text = fig1.add_subplot(gs[1,0])
        cx_text.set_axis_off()
        
        
        tab=cx_text.table(cellText=[["Temperature efficiency",str(spec_HE_efficiency[0])+"%",str(round(average_meas_eff_clean,1))+"%",str(round(std_meas_eff_clean,1))+"%"],["Intake temperature",str(spec_HE_temp_inlet_before[0])+u"\N{DEGREE SIGN}C",str(round(average_intake_temp,1))+u"\N{DEGREE SIGN}C",str(round(std_intake_temp,1))+u"\N{DEGREE SIGN}C"],["Extraction temperature",str(spec_HE_temp_exhaust_before[0])+u"\N{DEGREE SIGN}C",str(round(average_extraction_temp,1))+u"\N{DEGREE SIGN}C",str(round(std_extraction_temp,1))+u"\N{DEGREE SIGN}C"]],
                      colLabels=["The measured values are\nbased on "+str(len(meas_eff_clean))+" points","Manufacturer\n value","Measured\n average value","Measured\n standard deviation"],
                      cellLoc="center",
                      colLoc="center",
                      rowLoc="left",
                      bbox=[0.01,0.01,0.99,1.50])
        tab.auto_set_font_size(False)
        tab.set_fontsize(10)
        
        cell=tab[0,0]
        cell.set_fontsize(10)
        
        for r in range(0, 4):
            cell = tab[0, r]
            cell.set_height(0.7)
        
        for r in range(0, 4):
            cell = tab[1, r]
            cell.set_height(0.4)
            cell = tab[2, r]
            cell.set_height(0.4)
            cell = tab[3, r]
            cell.set_height(0.4)
            
        for r in range(0,4):
            cell = tab[r,0]
            cell.set_width(0.33)
            cell = tab[r,1]
            cell.set_width(0.18)
            cell = tab[r,2]
            cell.set_width(0.22)
            

        
        
        cx.set_xlabel(("Intake temperature ["+u"\N{DEGREE SIGN}C]"), fontsize=12, fontname='Cambria')
        cx.set_ylabel(("Extraction temperature ["+u"\N{DEGREE SIGN}C]"), fontsize=12, fontname='Cambria')
        cx.set_zlabel("Temperature efficiency [%]", fontsize=12, fontname='Cambria')
        fig1.suptitle("Rotary heat exchanger\nTemperature efficiency with reference bechmark",fontsize=20, fontname='Cambria',x=0.45,y=0.85)

        
        fig1.savefig("CLIMA22_eff_manufacturer_ref.svg",dpi=600)
        
        
        
        
        
        
        
