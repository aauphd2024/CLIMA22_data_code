# -*- coding: utf-8 -*-
"""
Created by Simon Melgaard and Kamilla Andersen

A function for creating a model for the airflow of a fan from its datasheet (air flow as a function of rotation speed and pressure over the fan)
"""

from scipy.optimize import curve_fit
import numpy as np


def calc_air_flow(fan_spec_rot_speed,fan_spec_pressure,fan_spec_flow,fan_signal,fan_spec_max_rot_speed,fan_pressure_difference):
    def func(data,a,b,c,d,e):
        x=data[0]
        y=data[1]
        return (x**a)*b+(y**c)*d+(x*y)**e

    initial_param=[1,1,0.5,-50,1]
    popt,pcov = curve_fit(func,[fan_spec_rot_speed, fan_spec_pressure],fan_spec_flow,p0=initial_param)
        
    print('fitted parameters', popt)
        
    modelPredictions = func([fan_spec_rot_speed, fan_spec_pressure], *popt) 

    absError = modelPredictions - fan_spec_flow
    SE = np.square(absError) # squared errors
    MSE = np.mean(SE) # mean squared errors
    RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
    Rsquared = 1.0 - (np.var(absError) / np.var(fan_spec_flow))
    CVRMSE = RMSE/np.mean(fan_spec_flow)
    print('RMSE:', RMSE)
    print('CVRMSE:', CVRMSE*100, '%')
    print('R-squared:', Rsquared)
        

    return func([fan_signal*fan_spec_max_rot_speed/100,fan_pressure_difference], *popt)/60