# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:20:20 2024

@author: alegn
"""

import matplotlib.pyplot as plt
import numpy as np
plt.style.use('default')
# Data CRF
# F1R1 -> Rice yield 12.008 ton/ha, fish yield 1.612+0.972 ton/ha, NumGrain = 26830134.753
# F2R1 -> Rice yield 12.008 ton/ha, fish yield 3.225+1.939 ton/ha, NumGrain = 26830134.753
# F3R1 -> Rice yield 12.008 ton/ha, fish yield 4.83872+ 2.90763 ton/ha, NumGrain = 26830134.753
# F1R2 -> Rice yield 12.008 ton/ha, fish yield 1.612+0.972 ton/ha, NumGrain = 26830125.198
# F2R2 -> Rice yield 12.008 ton/ha, fish yield 3.225+1.939 ton/ha, NumGrain =  26830125.198
# F3R2 -> Rice yield 12.008 ton/ha, fish yield 4.83872+ 2.90763 ton/ha, NumGrain = 26830125.198

#After revising calculation and DVS
# F1R1 -> Rice yield 12.008 ton/ha, fish yield 1.612+0.972 ton/ha, NumGrain = 26830134.753
# F2R1 -> Rice yield 12.008 ton/ha, fish yield 3.225+1.939 ton/ha, NumGrain = 26830134.753
# F3R1 -> Rice yield 12.008 ton/ha, fish yield 4.83872+ 2.90763 ton/ha, NumGrain = 26830134.753
# F1R2 -> Rice yield 12.008 ton/ha, fish yield 1.612+0.972 ton/ha, NumGrain = 26830125.198
# F2R2 -> Rice yield 12.008 ton/ha, fish yield 3.225+1.939 ton/ha, NumGrain =  26830125.198
# F3R2 -> Rice yield 12.008 ton/ha, fish yield 4.83872+ 2.90763 ton/ha, NumGrain = 26830125.198

# Example data for fish and rice yields
treatments = ['F1R1', 'F2R1', 'F3R1', 'F1R2', 'F2R2', 'F3R2']
fish_yields = [1.612+0.972, 3.225+1.939, 4.838+2.907, 1.612+0.972, 3.225+1.939, 4.838+2.907]  # fish yields in ton/ha
rice_yields = [12.008, 12.008, 12.008, 12.008, 12.008, 12.008]  # rice yields in ton/ha
num_grain = [26830134.753, 26830134.753, 26830134.753, 26830125.198, 26830125.198, 26830125.198]
num_grain_rounded = [round(y) for y in num_grain]

fig, ax1 = plt.subplots()
plt.rcParams.update({
    'font.size': 14,        # Global font size
    'axes.titlesize': 16,   # Title font size
    'axes.labelsize': 14,   # X and Y axis labels font size
    'xtick.labelsize': 16,  # X tick labels font size
    'ytick.labelsize': 16,  # Y tick labels font size
    'legend.fontsize': 16,  # Legend font size
    'figure.titlesize': 18  # Figure title font size
})
fish_color =  '#191970'  
rice_color = 'orange'  
line_color = 'red'  
line_marker = 'o'       # Circle marker

# Create bars for fish and rice yields
indices = np.arange(len(treatments))  # the x locations for the groups
width = 0.35  # the width of the bars

# Plot fish yields
fish_bars = ax1.bar(indices - width/2, fish_yields, width, label='Nile tilapia', color=fish_color)

# Plot rice yields
rice_bars = ax1.bar(indices + width/2, rice_yields, width, label='Rice', color=rice_color)

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
# ax2.plot(indices, num_grain_rounded, color=line_color, marker=line_marker, label='Number of Grains')
# ax2.set_ylabel('Number of Grains')

# # Add a legend for the line plot
# ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 0.9), bbox_transform=ax1.transAxes)

# # Title of the graph
# plt.title('Fish and Rice Yields and Number of Grains per Treatment')
# Adjust layout to make space for the legend
plt.tight_layout(rect=[0, 0, 0.85, 1])

# Show the plot
plt.show()