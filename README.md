# Arctic Energy Budget Analysis

Analysis of Arctic Top-of-Atmosphere (TOA) energy fluxes and ocean heat uptake using satellite and reanalysis data.

## 📊 Project Overview

This repository contains analyses of Arctic energy budget components:

- **CERES**: Top-of-Atmosphere radiative fluxes from CERES EBAF satellite data
- **ERA5**: Atmospheric energy fluxes from ERA5 reanalysis  
- **NETFLUX**: Net energy flux synthesis combining multiple data sources

## 📁 Repository Structure
```
arctic-energy-budget-analysis/
├── ceres/                    # CERES TOA flux analysis
│   ├── Arctic_TOA_Flux_Analysis.ipynb
│   └── *.png                 # Visualization outputs
├── era5/                     # ERA5 atmospheric energy analysis
│   ├── Arctic_ERA5_Analysis.ipynb
│   └── *.png
├── netflux/                  # Net flux synthesis
│   ├── net_flux-annual.ipynb
│   └── plots/
└── results/                  # Combined results
```

## 🚀 Getting Started

### Prerequisites
```bash
python >= 3.8
numpy
pandas
xarray
matplotlib
cartopy
```

### Installation
```bash
git clone https://github.com/mc5601/arctic-energy-budget-analysis.git
cd arctic-energy-budget-analysis
pip install -r requirements.txt
```

### Data Access

**CERES EBAF data**: Download from [NASA CERES](https://ceres.larc.nasa.gov/)  
**ERA5 data**: Download from [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)

Place downloaded `.nc` files in respective directories or update notebook paths.

## 📓 Notebooks

1. **CERES Analysis** (`ceres/Arctic_TOA_Flux_Analysis.ipynb`)
   - Analyzes TOA shortwave and longwave fluxes
   - Seasonal and annual trends
   - Arctic amplification signals

2. **ERA5 Analysis** (`era5/Arctic_ERA5_Analysis.ipynb`)  
   - Atmospheric energy transport
   - dE/dt calculations
   - Energy flux convergence

3. **Net Flux** (`netflux/net_flux-annual.ipynb`)
   - Multi-dataset synthesis
   - Annual mean energy budgets

## 📈 Key Results

All visualization outputs are included in respective directories. Key findings include Arctic TOA flux trends and energy budget closure analysis.

## 🔬 Methods

- Spatial averaging over Arctic domain (>60°N)
- Seasonal decomposition (DJF, JJA)
- Trend analysis (2000-2024)

## 📚 Citation

If you use this code or analysis, please cite:
```
Celedón, M. (2024). Arctic Energy Budget Analysis. 
GitHub: https://github.com/mc5601/arctic-energy-budget-analysis
```

## 📧 Contact

For questions or collaborations: mc5601@columbia.edu

## 📄 License

MIT License - feel free to use and modify with attribution.
