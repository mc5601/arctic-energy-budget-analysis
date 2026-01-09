# Arctic Energy Budget Analysis

Analysis of the **Arctic energy budget** using **satellite (CERES)**, **reanalysis (ERA5)**, and **ocean reanalysis (ORAS5)** data.  
The repository is organized as three notebooks (CERES, ERA5, NETFLUX) plus small helper modules used to compute annual series and save plots.

---

## What’s in here

- **CERES** (`ceres/Arctic_TOA_Flux_Analysis.ipynb`)  
  Top-of-Atmosphere (TOA) radiative fluxes (shortwave, longwave, net) over the Arctic.

- **ERA5** (`era5/Arctic_ERA5_Analysis.ipynb`)  
  Atmospheric energy flux diagnostics and time tendencies over the Arctic.

- **NETFLUX** (`netflux/net_flux-annual.ipynb`)  
  Annual synthesis combining CERES + ERA5 + ORAS5 components (energy budget closure / net flux + ocean heat uptake).

---

## Repository structure

```text
arctic-energy-budget-analysis/
├── ceres/
│   ├── Arctic_TOA_Flux_Analysis.ipynb
│   └── CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202502.nc   (NOT tracked; can be a symlink)
├── era5/
│   ├── Arctic_ERA5_Analysis.ipynb
│   └── ERA5_energy_200001-202504.nc                     (NOT tracked; can be a symlink)
├── netflux/
│   ├── net_flux-annual.ipynb
│   └── plots/                                           (optional; some runs may write here)
├── models/                                              (helper modules used by netflux)
│   ├── __init__.py
│   ├── era_annual.py
│   ├── ceres_ebaf_annual.py
│   └── ora5_annual.py
├── plots/                                               (saved figures for README / outputs)
├── requirements.txt
└── README.md
