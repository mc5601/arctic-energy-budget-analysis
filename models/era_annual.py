# -*- coding: utf-8 -*-
"""
Created on Thu May 22 10:47:20 2025
@author: Martin

ERA5 Energy Flux Time Series Analysis
Adapted from CERES TOA flux analysis
"""
# Add at the TOP of each .py file

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### Define NetCDF data file name
file_name = '../ERA5/ERA5_energy_200001-202504.nc' 

### Open file with read only
root = nc.Dataset(file_name, 'r')

### Get the variables as a Python dictionary
variables = root.variables
            
### Write the variable names
print('\n----- VARIABLE NAMES -----')
for key in variables:
    print(key)

### Write all the variable info including attributes
print('\n----- VARIABLE SUMMARY -----')
print(variables)

### Retrieve the energy flux data and geographic information
lon = np.array(root['longitude'][:])
lat = np.array(root['latitude'][:])
lat_w = np.cos(np.deg2rad(lat))
time = np.array(root['valid_time'][:])
energy_flux = np.array(root['vited'][:,:,:])  # ERA5 energy flux variable
vitoe = np.array(root['vitoe'][:, :, :])




def compute_lat_weighted_mean(data, lat):
    """
    Compute the global mean time series with latitude weighting.
    
    Parameters:
        data : np.ndarray
            3D array with shape [time, lat, lon]
        lat : np.ndarray
            1D array of latitude values (degrees)
    
    Returns:
        global_mean : np.ndarray
            1D array with shape [time], the global mean at each time step
    """
    # Convert latitudes to weights (cosine of latitude in radians)
    lat_weights = np.cos(np.deg2rad(lat))
    
    # Initialize result array
    global_mean = np.zeros(data.shape[0])  # one value per time step
    
    for t in range(data.shape[0]):
        monthly_data = data[t, :, :]  # shape [lat, lon]
        zonal_mean = np.mean(monthly_data, axis=1)  # shape [lat]
        global_mean[t] = np.average(zonal_mean, weights=lat_weights)
    
    return global_mean


def compute_annual_mean(monthly_series, start_year=2000):
    """
    Convert a monthly time series to an annual time series by averaging every 12 months.

    Parameters:
        monthly_series : np.ndarray
            1D array of monthly data [time]
        start_year : int
            The first year in the time series (default is 2000)

    Returns:
        years : np.ndarray
            Array of years
        annual_series : np.ndarray
            Array of annual means
    """
    n_months = len(monthly_series)
    n_years = (n_months-2) // 12
    annual_series = np.zeros(n_years)

    for y in range(n_years):
        start_idx = y * 12 + 2         # March
        end_idx = start_idx + 12       # until feb for the next year
        annual_series[y] = np.mean(monthly_series[start_idx:end_idx])
    years = np.arange(start_year, start_year + n_years)
    return years, annual_series



def plot_energy_flux_timeseries(x, energy_data, is_annual=False, time_label='Time', save_path=None, title=None):
    """
    Plot ERA5 energy flux time series with trend line.
    
    Parameters:
        x : np.ndarray
            Time array (years or months since start)
        energy_data : np.ndarray
            Energy flux data array
        is_annual : bool
            Whether the data is annual (True) or monthly (False)
        time_label : str
            Label for the x-axis
        save_path : str
            Path to save the figure as PNG (optional)
    """
    lw_width = 2 if is_annual else 1
    
    # Calculate trend line
    energy_trend = np.poly1d(np.polyfit(x, energy_data, 1))
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    plt.plot(x, energy_data, color='red', linewidth=lw_width, label='ERA5 Energy Flux')
    plt.plot(x, energy_trend(x), 'k--', linewidth=1.5, label='Trend')
    plt.xlabel(time_label)
    plt.ylabel('Energy Flux (W/m²)')
    plt.grid(True)
    plt.legend()
    
    if title is not None:
        plt.title(title, fontsize=16)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ Saved plot as {save_path}")
    plt.show()
    




# Calculate global means using latitude weighting
energy_flux_mon_mean = compute_lat_weighted_mean(energy_flux, lat)
vitoe_mon_mean = compute_lat_weighted_mean(vitoe, lat)

# Calculate annual means
years, energy_flux_annual = compute_annual_mean(energy_flux_mon_mean, start_year=2000)
years, vitoe_annual = compute_annual_mean(vitoe_mon_mean , start_year=2000)

N = len(vitoe_annual)
dt = 365 * 24 * 3600  # seconds in a year 

dE_dt = np.zeros_like(vitoe_annual)
for i in range(N):
    if i == 0:
        dE_dt[i] = (vitoe_annual[i+1] - vitoe_annual[i]) / dt
    elif i == N-1:
        dE_dt[i] = (vitoe_annual[i] - vitoe_annual[i-1]) / dt
    else:
        dE_dt[i] = (vitoe_annual[i+1] - vitoe_annual[i-1]) / (2 * dt)



# Plot annual energy flux
plot_energy_flux_timeseries(
    x=years,
    energy_data=energy_flux_annual,
    is_annual=True,
    time_label='Year',
    save_path='plots/era5_annual_energy_flux.png',
    title="Annual Atmospheric Energy Flux in the Arctic"
)

plot_energy_flux_timeseries(
    x=years,
    energy_data=dE_dt,
    is_annual=True,
    time_label='Year',
    save_path='plots/era5_annual_dE_dt.png',
    title="Annual Change in Arctic Atmospheric Energy (dE/dt)"
)


root.close()