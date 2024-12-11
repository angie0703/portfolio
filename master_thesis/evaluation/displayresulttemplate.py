# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:20:20 2024

@author: alegn
"""

import matplotlib.pyplot as plt
import numpy as np
plt.style.use('default')
# Data CRF
# F1R1 -> Rice yield 8.78437 ton/ha, fish yield 1.71562 ton/ha, NumGrain = 6064978.768
# F2R1 -> Rice yield 8.78437 ton/ha, fish yield 3.32852 ton/ha, NumGrain = 6064978.772
# F3R1 -> Rice yield 8.78437 ton/ha, fish yield 4.94142 ton/ha, NumGrain = 6064978.775
# F1R2 -> Rice yield 8.78437 ton/ha, fish yield 1.71562 ton/ha, NumGrain = 6064977.541
# F2R2 -> Rice yield 8.78437 ton/ha, fish yield 3.32852 ton/ha, phyto yield 62.9288 g/m3, NumGrain =  6064977.544
# F3R2 -> Rice yield 8.78437 ton/ha, fish yield 4.94142 ton/ha, phyto yield 62.9289 g/m3, NumGrain = 6064977.548

#After checking the and revise the DVS
# F1R1 -> Rice yield 12.987 ton/ha, fish yield 1.71562 ton/ha, NumGrain = 7726379.224
# F2R1 -> Rice yield 8.78437 ton/ha, fish yield 3.32852 ton/ha, NumGrain = 7726379.132
# F3R1 -> Rice yield 8.78437 ton/ha, fish yield 4.94142 ton/ha, NumGrain = 7726379.041
# F1R2 -> Rice yield 8.78437 ton/ha, fish yield 1.71562 ton/ha, NumGrain = 7726377.801
# F2R2 -> Rice yield 8.78437 ton/ha, fish yield 3.32852 ton/ha, NumGrain =  7726377.709
# F3R2 -> Rice yield 8.78437 ton/ha, fish yield 4.94142 ton/ha,  NumGrain = 7726377.618

# Example data for fish and rice yields
treatments = ['F1R1', 'F2R1', 'F3R1', 'F1R2', 'F2R2', 'F3R2']
fish_yields = [1.715, 3.328, 4.941, 1.715, 3.328, 4.941]  # fish yields in ton/ha
rice_yields = [12.987, 12.987, 12.987, 12.987, 12.987, 12.987]  # rice yields in ton/ha
num_grain = [7726379.224, 7726379.132, 7726379.041, 7726377.801, 7726377.709, 7726377.618]
num_grain_rounded = [round(y) for y in num_grain]

fig, ax1 = plt.subplots()
plt.rcParams.update({
    'font.size': 14,        # Global font size
    'axes.titlesize': 16,   # Title font size
    'axes.labelsize': 14,   # X and Y axis labels font size
    'xtick.labelsize': 12,  # X tick labels font size
    'ytick.labelsize': 12,  # Y tick labels font size
    'legend.fontsize': 12,  # Legend font size
    'figure.titlesize': 18  # Figure title font size
})
# Create bars for fish and rice yields
indices = np.arange(len(treatments))  # the x locations for the groups
width = 0.35  # the width of the bars

# Plot fish yields
fish_bars = ax1.bar(indices - width/2, fish_yields, width, label='Nile tilapia', color='#191970' )

# Plot rice yields
rice_bars = ax1.bar(indices + width/2, rice_yields, width, label='Rice', color='orange')

# Annotate bars with the numerical values
for bar in fish_bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

for bar in rice_bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

# Configure the primary y-axis
ax1.set_xlabel('Treatment')
ax1.set_ylabel('Yields (ton/ha)')
ax1.set_xticks(indices)
ax1.set_xticklabels(treatments)
ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), bbox_transform=ax1.transAxes)

# # Create a second y-axis for the number of grains using the same x-axis
# ax2 = ax1.twinx()
# ax2.plot(indices, num_grain_rounded, 'ro-', label='Number of Grains')
# ax2.set_ylabel('Number of Grains')

# # Add a legend for the line plot
# ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 0.9), bbox_transform=ax1.transAxes)

# Title of the graph
# plt.title('Fish and Rice Yields and Number of Grains per Treatment')

plt.tight_layout(rect=[0, 0, 0.85, 1])

# Show the plot
plt.show()