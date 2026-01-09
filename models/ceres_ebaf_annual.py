"""
Created on Tue May  6 17:50:01 2025

@author: Martin
"""

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### Define NetCDF data file name
file_name = '../CERES/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202502.nc' 

### Open file with read only
root = nc.Dataset(file_name, 'r')

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
    For CERES data, calculates March-to-March years (e.g., Mar 2000 - Feb 2001 = year 2000).

    Parameters:
        monthly_series : np.ndarray
            1D array of monthly data [time]
        start_year : int
            The first year in the time series (default is 2000)
        start_month : int
            The starting month (0-indexed, so 2 = March, default for CERES)

    Returns:
        years : np.ndarray
            Array of years
        annual_series : np.ndarray
            Array of annual means (March-to-March cycles)
    """
    n_months = len(monthly_series)
    
    n_years = n_months // 12
    annual_series = np.zeros(n_years)
    for y in range(n_years):
        annual_series[y] = np.mean(monthly_series[y * 12:(y + 1) * 12])
    years = np.arange(start_year, start_year + n_years)
    return years, annual_series




 

def plot_flux_subplots(x, sw, lw, net, is_annual=False, time_label='Time', save_path=None, sw_label='TOA SW Flux'):
   

    lw_width = 2 if is_annual else 1
    sw_trend = np.poly1d(np.polyfit(x, sw, 1))
    lw_trend = np.poly1d(np.polyfit(x, lw, 1))
    net_trend = np.poly1d(np.polyfit(x, net, 1))

    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axs[0].plot(x, sw, color='orange', linewidth=lw_width, label=sw_label)
    axs[0].plot(x, sw_trend(x), 'k--', linewidth=1.5, label='Trend')
    axs[0].set_ylabel('Flux (W/m²)')
    axs[0].set_title(sw_label)
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(x, lw, color='blue', linewidth=lw_width, label='TOA LW Flux')
    axs[1].plot(x, lw_trend(x), 'k--', linewidth=1.5, label='Trend')
    axs[1].set_ylabel('Flux (W/m²)')
    axs[1].set_title('TOA LW Flux')
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(x, net, color='green', linewidth=lw_width, label='TOA Net Flux')
    axs[2].plot(x, net_trend(x), 'k--', linewidth=1.5, label='Trend')
    axs[2].set_xlabel(time_label)
    axs[2].set_ylabel('Flux (W/m²)')
    axs[2].set_title('TOA Net Flux')
    axs[2].grid(True)
    axs[2].legend()

    fig.suptitle('Arctic: ' + ('Annual' if is_annual else 'Monthly') + ' TOA Fluxes', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()




def plot_sw_components(x, incident, reflected, net, title="SW Flux Components", time_label='Year', save_path=None):
    """
    Plot the three shortwave flux components (incident, reflected, net) in three stacked subplots.
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    valid_inc = np.isfinite(x) & np.isfinite(incident)
    valid_ref = np.isfinite(x) & np.isfinite(reflected)
    valid_net = np.isfinite(x) & np.isfinite(net)

    inc_trend = np.poly1d(np.polyfit(x[valid_inc], incident[valid_inc], 1)) if np.sum(valid_inc) > 1 else lambda x: np.full_like(x, np.nan)
    ref_trend = np.poly1d(np.polyfit(x[valid_ref], reflected[valid_ref], 1)) if np.sum(valid_ref) > 1 else lambda x: np.full_like(x, np.nan)
    net_trend = np.poly1d(np.polyfit(x[valid_net], net[valid_net], 1)) if np.sum(valid_net) > 1 else lambda x: np.full_like(x, np.nan)

    axs[0].plot(x, incident, color='red', linewidth=2, label='Incident Solar Flux')
    axs[0].plot(x, inc_trend(x), 'k--', linewidth=1.5, label='Trend')
    axs[0].set_ylabel('Flux (W/m²)')
    axs[0].set_title('Incident Solar Flux')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(x, reflected, color='orange', linewidth=2, label='Reflected SW Flux')
    axs[1].plot(x, ref_trend(x), 'k--', linewidth=1.5, label='Trend')
    axs[1].set_ylabel('Flux (W/m²)')
    axs[1].set_title('Reflected SW Flux')
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(x, net, color='purple', linewidth=2, label='Net Incoming SW Flux')
    axs[2].plot(x, net_trend(x), 'k--', linewidth=1.5, label='Trend')
    axs[2].set_xlabel(time_label)
    axs[2].set_ylabel('Flux (W/m²)')
    axs[2].set_title('Net Incoming SW Flux')
    axs[2].grid(True)
    axs[2].legend()

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

### Retrieve the TOA flux data and geographic information
lon = np.array(root['lon'][:])
lat = np.array(root['lat'][:])
lat_w = np.cos(np.deg2rad(lat))
time = np.array(root['time'][:])

# Clear naming convention for different flux components
toa_sw_refl_mon = np.array(root['toa_sw_all_mon'][:,:,:])  # Reflected SW flux
toa_lw_mon = np.array(root['toa_lw_all_mon'][:,:,:])       # Outgoing LW flux
toa_net_mon = np.array(root['toa_net_all_mon'][:,:,:])     # Net TOA flux
toa_sw_inc_mon = np.array(root['solar_mon'][:,:,:])        # Incoming solar flux
toa_sw_net_mon = toa_sw_inc_mon - toa_sw_refl_mon          # Net incoming SW flux

# Calculate global means using latitude weighting
toa_sw_refl_mon_mean = compute_lat_weighted_mean(toa_sw_refl_mon, lat)
toa_lw_mon_mean = compute_lat_weighted_mean(toa_lw_mon, lat)
toa_net_mon_mean = compute_lat_weighted_mean(toa_net_mon, lat)
toa_sw_inc_mon_mean = compute_lat_weighted_mean(toa_sw_inc_mon, lat)
toa_sw_net_mon_mean = compute_lat_weighted_mean(toa_sw_net_mon, lat)

# Calculate annual means
years, toa_sw_refl_annual = compute_annual_mean(toa_sw_refl_mon_mean)
years, toa_lw_annual = compute_annual_mean(toa_lw_mon_mean)
years, toa_net_annual = compute_annual_mean(toa_net_mon_mean)
years, toa_sw_inc_annual = compute_annual_mean(toa_sw_inc_mon_mean)
years, toa_sw_net_annual = compute_annual_mean(toa_sw_net_mon_mean)

# Time arrays
months = np.arange(len(time))
years_from_time = 2000 + months / 12

# Plots

plot_flux_subplots(years, toa_sw_refl_annual, toa_lw_annual, toa_net_annual, is_annual=True, time_label='Year', save_path='plots/ceres_plot_annual_fluxes.png')
plot_flux_subplots(years, toa_sw_net_annual, toa_lw_annual, toa_net_annual, is_annual=True, time_label='Year', save_path='plots/ceres_annual_fluxes_sw_net.png')

plot_sw_components(years, toa_sw_inc_annual, toa_sw_refl_annual, toa_sw_net_annual, title="Annual Shortwave Flux Components", save_path='plots/ceres_annual_sw_components.png')

