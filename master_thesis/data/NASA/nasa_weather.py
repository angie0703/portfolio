# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 17:31:38 2024

@author: alegn
"""
import pandas as pd
import numpy as np
import netCDF4 as ncd

# Open the NetCDF file
ds = ncd.Dataset('C:\\Users\\alegn\\Documents\\WUR\\Thesis\\rfmodel\\data\\NASA\\nasa.nc', 'r')

# List all variables
print("Variables in the file:")
for var in ds.variables:
    print(var)

# print("\nDetailed information on each variable:")
# for name, variable in ds.variables.items():
#     print(f"{name}:")
#     print(f"    data type: {variable.dtype}")
#     print(f"    dimensions: {variable.dimensions}")
#     print(f"    size: {variable.size}")
#     print(f"    shape: {variable.shape}")
#     print(f"    description: {variable.long_name if 'long_name' in variable.ncattrs() else 'No description'}")
#     print(f"    units: {variable.units if 'units' in variable.ncattrs() else 'No units'}\n")

# # Create a dictionary to store data
# data_dict = {}

# # Loop through all variables and store them in the dictionary
# for var_name in ds.variables:
#     data = ds.variables[var_name][:]
#     data_dict[var_name] = np.array(data)

# print(data_dict)

# Create an empty DataFrame
df = pd.DataFrame()

# Loop through all variables and add them to the DataFrame
for var_name in ds.variables:
    var_data = ds.variables[var_name][:]
    # Check if the variable is multi-dimensional and needs flattening
    if var_data.ndim > 1:
        var_data = var_data.flatten()  # Flatten the data if it is multi-dimensional
    df[var_name] = var_data

# Convert the DataFrame to CSV
csv_path = 'C:\\Users\\alegn\\Documents\\WUR\\Thesis\\rfmodel\\data\\NASA\\output.csv'
df.to_csv(csv_path, index=False)

print(f"CSV file has been created at: {csv_path}")  
# Close the dataset
ds.close()