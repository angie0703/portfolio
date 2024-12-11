# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 16:23:30 2024

@author: alegn
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


tsim = np.linspace(0, 120*24, 120*24+1)# [d]


#disturbances

# Disturbances
data_weather = 'C:/Users/alegn/Documents/WUR/Thesis/rfmodel/data/Weather/DIY_202210_202309_Hourly.csv'
weather = pd.read_csv(data_weather, header=3, sep=';')

t_ini = '20221001T08:00'
t_end = '20230129T08:00'
# t_end = '20230930T23:00'
t_weather = np.linspace(0, 120*24, 120*24+1)
weather['time'] = pd.to_datetime(weather['time'], format='ISO8601')  # Adjust the format if necessary
weather.set_index('time', inplace=True)
Th = weather.loc[t_ini:t_end,'temperature'].values #[°C] Mean daily temperature
Rain = weather.loc[t_ini:t_end,'rain'].values #[mm] Daily precipitation
Igl = weather.loc[t_ini:t_end, 'shortwave_radiation'].values #[W/m2] Hourly shortwave radiation

I0 = Igl*3600 #Convert [MJ m-2 d-1] to [J m-2 d-1] PAR

d = {
      'I0' : np.array([tsim, I0]).T, 
      'Th':np.array([tsim, Th]).T,
      'Rain': np.array([tsim, Rain]).T
      }

# Given start date and time
start_datetime_str = '20221001T00:00'
start_datetime = datetime.strptime(start_datetime_str, '%Y%m%dT%H:%M')

# Duration to add (120 days)
duration = timedelta(days=120)

# Calculate end datetime
end_datetime = start_datetime + duration

# Format the end datetime back to string if needed
end_datetime_str = end_datetime.strftime('%Y%m%dT%H:%M')

print("End Date and Time:", end_datetime_str)
x_ticks = np.linspace (1, 2880, 120)

plt.figure(1)
plt.plot(tsim, Th, label = '$T_{h}$')
plt.legend()
plt.xlabel('Time [d]')
plt.xticks(x_ticks, labels=np.arange(1,120+1))
plt.ylabel('Temperature (\u00B0C)')
plt.show()

# fig, ax1 = plt.subplots()
# ax1.plot(tsim, I0, 'b-')  # 'r-' is a red solid line
# ax1.set_xlabel('Days')
# ax1.set_ylabel(r'Solar Irradiation (J m$^{-2}$ h$^{-1}$)', color='b')
# ax1.tick_params(axis='y', labelcolor='b')

# ax2 = ax1.twinx()
# ax2.bar(tsim, Rain)  # 'b-' is a blue solid line
# ax2.set_ylabel('Precipitation (mm)')
# ax2.tick_params(axis='y')

plt.figure(2)
plt.plot(tsim, Rain, label = 'rainfall [mm]')
plt.legend()
plt.xlabel('Time [d]')
plt.ylabel('Rainfall (mm)')
plt.xticks(x_ticks, labels=np.arange(1,120+1))
plt.show()

plt.figure(3)
plt.plot(tsim, I0)
plt.legend()
plt.xlabel('Time [d]')
plt.ylabel(r'Solar Irradiation (J m$^{-2}$ h$^{-1}$)')
plt.xticks(x_ticks, labels=np.arange(1,120+1))
plt.show()