# -*- coding: utf-8 -*-
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def compute_annual_mean(monthly_series, start_year=2000):
       # monthly_series: (time, lat, lon)
    n_months = monthly_series.shape[0]
    ny, nx = monthly_series.shape[1], monthly_series.shape[2]
    n_years = (n_months-2) // 12  # 
    annual_series = np.zeros((n_years, ny, nx))
    for y in range(n_years):
        start_idx = y * 12 + 2      # march
        end_idx = start_idx + 12    # until feb
        annual_series[y] = np.nanmean(monthly_series[start_idx:end_idx, :, :], axis=0)
    years = np.arange(start_year, start_year + n_years)
    return years, annual_series



def plot_flux_timeseries(x, energy_data, is_annual=False, time_label='Time', save_path=None, title=None):
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
    
    plt.plot(x, energy_data, color='red', marker='o', linewidth=lw_width, label='ORA5 Energy Flux')
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
    


# Define NetCDF data file name
file_name = '../ORAS5/oras5_2000-2025_m.nc' 
root = nc.Dataset(file_name, 'r')
variables = root.variables
lon = np.array(root['nav_lon'][:])
lat = np.array(root['nav_lat'][:])
time = np.array(root['time_counter'][:])
SHF = np.array(root['sohefldo'][:])
OHC = np.array(root['sohtcbtm'][:])
SHF = np.where(SHF > 1e20, np.nan, SHF)
OHC = np.where(OHC > 1e20, np.nan, OHC)


years, SHF_annual = compute_annual_mean(SHF, start_year=2000)
years, OHC_annual = compute_annual_mean(OHC, start_year=2000)

N = len(OHC_annual)
dt = 365 * 24 * 3600  # seconds in a year (adjust as needed)

dOHC_dt = np.zeros_like(OHC_annual)
for i in range(N):
    if i == 0:
        dOHC_dt[i] = (OHC_annual[i+1] - OHC_annual[i]) / dt
    elif i == N-1:
        dOHC_dt[i] = (OHC_annual[i] - OHC_annual[i-1]) / dt
    else:
        dOHC_dt[i] = (OHC_annual[i+1] - OHC_annual[i-1]) / (2 * dt)

F_div = SHF_annual - dOHC_dt

mesh = nc.Dataset('../ORAS5/mesh_mask.nc', 'r')
e1t = np.array(mesh['e1t'][:])
e2t = np.array(mesh['e2t'][:])
area_cell = (e1t * e2t)[0, :, :]
area_cell = np.where(np.abs(area_cell) > 1e20, np.nan, area_cell)
arctic_mask = lat >= 60

dOHC_dt_masked = np.where(arctic_mask[None, :, :], dOHC_dt, np.nan)
dOHC_dt_masked = np.where(np.abs(dOHC_dt_masked) > 1e20, np.nan, dOHC_dt_masked)
dOHC_dt_W = dOHC_dt_masked * area_cell[np.newaxis, :, :]
dOHC_dt_total = np.nansum(dOHC_dt_W, axis=(1, 2))

F_div_masked = np.where(arctic_mask, F_div, np.nan)
F_div_masked = np.where(np.abs(F_div_masked) > 1e20, np.nan, F_div_masked)
F_div_W = F_div_masked * area_cell[np.newaxis, :, :]
F_div_total = np.nansum(F_div_W, axis=(1, 2))

R = 6.371e6
Area_Arctic = 2 * np.pi * R**2 * (1 - np.sin(np.radians(60)))

F_div_artic = F_div_total / Area_Arctic
dOHC_dt_artic = dOHC_dt_total / Area_Arctic



plot_flux_timeseries(
    x=years,
    energy_data=F_div_artic,
    is_annual=True,
    time_label='Year',
    save_path='plots/ORA5_annual_Fdiv_flux.png',
    title="Annual Ocean Heat Flux Divergence in the Arctic"
)

plot_flux_timeseries(
    x=years,
    energy_data=dOHC_dt_artic,
    is_annual=True,
    time_label='Year',
    save_path='plots/ORA5_annual_dOhc_flux.png',
    title="Annual Change in Arctic Ocean Heat Content (dOHC/dt)"
)


root.close()
mesh.close()