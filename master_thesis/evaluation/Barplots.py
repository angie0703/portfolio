# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:08:45 2024

@author: alegn
"""
import matplotlib.pyplot as plt
import numpy as np 

categories =['No fertilizer', 'Organic fertilizer', 'Organic and inorganic fertilizer']
fish_weight = [12.925, 13.225, 13.36]
rice_weight = [3.316, 3.316, 3.316]
N_net = [301.59, 320.526, 354.692]
P_net = [36.83, 174.717, 175.027]

# Create positions for the bars
bar_width = 0.35  # Width of each bar
positions = np.arange(len(categories))  # Positions for the first set of bars

# Create the bar plots
fig, ax = plt.subplots()
ax.bar(positions, fish_weight, width=bar_width, label='Fish Weight', color='blue')
ax.bar(positions + bar_width, rice_weight, width=bar_width, label='Rice Weight', color='orange')

# Customize the plot
ax.set_xlabel('Fertilizer application')
ax.set_ylabel('Yield (ton/ha)')
# ax.set_title('Fish Weight vs. Rice Weight')
ax.set_xticks(positions + bar_width / 2)  # Set x-ticks in the middle of the grouped bars
ax.set_xticklabels(categories)  # Set category names as x-tick labels
ax.legend()

# Display the plot
plt.show()