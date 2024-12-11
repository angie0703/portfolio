# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 16:23:30 2024

@author: alegn
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

t = 364
tsim = np.linspace(0.0, t, t+1) # [d]


#disturbances

# Disturbances
data_weather = 'C:/Users/alegn/Documents/WUR/Thesis/rfmodel/data/Weather/DIY_202210_202309_Daily.csv'
weather = pd.read_csv(data_weather, header=1, sep=';')

t_ini = '20221001'
t_end = '20230930'
weather['Time'] = pd.to_datetime(weather['Time'], format='%Y%m%d')  # Adjust the format if necessary
weather.set_index('Time', inplace=True)
Tmax = weather.loc[t_ini:t_end,'Tmax'].values
Tmin = weather.loc[t_ini:t_end,'Tmin'].values
Tavg = weather.loc[t_ini:t_end,'Tavg'].values #[°C] Mean daily temperature
Rain = weather.loc[t_ini:t_end,'Rain'].values #[mm] Daily precipitation
Igl = weather.loc[t_ini:t_end, 'I0'].values #[MJ m-2] Sum of shortwave radiation daily

I0 = 0.45*Igl*1E6 #Convert [MJ m-2 d-1] to [J m-2 d-1] PAR

d = {
      'I0' : np.array([tsim, I0]).T, #[J m-2 d-1] Sum of Global Solar Irradiation (6 AM to 6 PM)
      'Tmax':np.array([tsim, Tmax]).T,
      'Tavg':np.array([tsim, Tavg]).T,
      'Tmin':np.array([tsim, Tmin]).T,
      'Rain': np.array([tsim, Rain]).T
      }

plt.figure(1)
plt.plot(tsim, Tavg, label = 'mean temperature')
plt.plot(tsim, Tmax, label = 'max temperature')
plt.plot(tsim, Tmin, label = 'min temperature')
plt.legend()
plt.xlabel('Days')
plt.ylabel('Temperature (\u00B0C)')
plt.title('Daily Temperature of Central Java Region 2022 - 2023')
plt.show()

fig, ax1 = plt.subplots()
ax1.plot(tsim, I0, 'darkblue')  
ax1.set_xlabel('Days')
ax1.set_ylabel(r'Solar Irradiation (J m$^{-2}$ d$^{-1}$)')
ax1.tick_params(axis='y')

ax2 = ax1.twinx()
ax2.bar(tsim, Rain)  # 'b-' is a blue solid line
ax2.set_ylabel('Precipitation (mm)')
ax2.tick_params(axis='y')
ax1.set_title("Solar Irradiation and Daily Precipitation of Central Java Region 2022-2023")
# plt.figure(2)
# plt.plot(tsim, Rain, label = 'rainfall [mm]')
# plt.legend()
# plt.xlabel('Days')
# plt.ylabel('Rainfall (mm)')
# plt.show()

# plt.figure(3)
# plt.plot(tsim, I0)
# plt.legend()
# plt.xlabel('Days')
# plt.ylabel(r'Solar Irradiation (J m$^{-2}$ d$^{-1}$)')
plt.show()