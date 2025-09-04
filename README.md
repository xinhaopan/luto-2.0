# LUTO 2: The Land-Use Trade-Offs Model Version 2

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Introduction
The Land Use Trade-Offs model v2 (LUTO2) is an integrated land systems model designed to simulate the optimal spatial arrangement of land use and land management decisions over time in Australia. It aims to achieve climate and biodiversity targets without compromising economic growth, food production or water security. The model is implemented as a Python package, offering users the flexibility to run interactively or to execute batch processes through scripted automation.

LUTO2 was developed through a collaboration between Deakin University and Climateworks Centre, with research contributions from CSIRO. The model is a cornerstone of Climateworks’ Land Use Futures program, which supports Australia’s transition to sustainable food and land systems. The technical development of LUTO2 is led by Professor Brett Bryan at Deakin University. LUTO2 continues the approach to land-use change modelling of its predecessor, the original LUTO, which was developed by CSIRO from 2010 - 2015 (see also Pedigree, below) and published under the GNU GPLv3 in 2021.

## Pedigree
LUTO2 builds on the approach and pedigree of nearly two decades of land-use modelling expertise starting with the original LUTO model. The original LUTO model was developed by CSIRO for the Australian National Outlook in 2015 and was groundbreaking for quantifying and projecting land use changes and their sustainability impacts in Australia, illustrated by its published works in *Nature* in 2015 and 2017.

LUTO2 represents a generational leap in sophistication and functionality for national-scale land-use change modelling in Australia. Both LUTO versions are optimisation models but different commercial solvers are used (CPLEX in original LUTO, GUROBI in LUTO2). The spatial domains are different in extent, with LUTO2's being nearly 5 times as large. The data requirements to run LUTO2 are consequently different and heavier. There is no backwards compatibility whatsoever.

The original LUTO model is available online and should be cited as:
> Bryan, Brett; Nolan, Martin; Stock, Florian; Graham, Paul; Dunstall, Simon; Ernst, Andreas; Connor, Jeff (2021): Land Use Trade-Offs (LUTO) Model. v1. CSIRO. Software Collection. https://doi.org/10.25919/y8ee-sk45.

This new version represents an entirely new model featuring a complete rewrite of the codebase and comprehensive upgrades to data and functionality. Enhancements to the original model include extended spatial coverage and timespan (2010 to 2100), a complete refresh of input data, additional land-use options and sustainability indicators and management solutions, the ability to model demand-side solutions, and additional environmental indicators and reporting. Due to LUTO2’s model complexity, the computational requirements to run the model are far more intensive.

LUTO2’s modelling approach, indicators and solutions have been guided by extensive stakeholder consultation (documented here: https://doi.org/10.1007/s11625-024-01574-7) following principles of participatory model co-design.

## Authors
Coordinating lead author: **Bryan, B.A.**  

Lead authors (in order of contribution): **Williams, N., Archibald, C.L., de Haan, F., Wang, J., van Schoten, N., Hadjikakou, M., Sanson, J., Zyngier, R., Marcos-Martinez, R., Navarro, J., Gao, L., Aghighi, H.**  

Other significant contributors (in alphabetical order): **Armstrong, T., Bohl, H., Jaffe, P., Khan, M.S., Moallemi, E.A., Nazari, A., Pan, X., Steyl, D., Thiruvady, D.R.**

## Documentation
Documentation, including instructions on how to set up and run LUTO2, can be found at `docs/luto2-overview.pdf`.

LUTO2 comes with a full diagram to illustrate its data preparation, workflow, and code logics. The diagram link can be found in this link.  
*Replace with updated documentation currently in preparation when ready.*

## Project Structure

The LUTO2 codebase is organized into the following structure:

```
luto/                                # Main package directory
├── data.py                          # Core data management and loading
├── simulation.py                    # Main simulation engine
├── settings.py                      # Configuration parameters
├── dataprep.py                      # Data preprocessing utilities
├── helpers.py                       # Utility functions
├── economics/                       # Economic models and calculations
│   ├── agricultural/                # Agricultural economics modules
│   │   ├── biodiversity.py          # Biodiversity calculations
│   │   ├── cost.py                  # Cost calculations
│   │   ├── ghg.py                   # GHG emissions calculations
│   │   ├── quantity.py              # Production quantity calculations
│   │   ├── revenue.py               # Revenue calculations
│   │   ├── transitions.py           # Land use transition costs
│   │   └── water.py                 # Water yield calculations
│   ├── non_agricultural/            # Non-agricultural economics modules
│   │   ├── biodiversity.py          # Non-ag biodiversity impacts
│   │   ├── cost.py                  # Non-ag establishment costs
│   │   ├── ghg.py                   # Non-ag GHG calculations
│   │   ├── quantity.py              # Non-ag production quantities
│   │   ├── revenue.py               # Non-ag revenue streams
│   │   ├── transitions.py           # Non-ag transition costs
│   │   └── water.py                 # Non-ag water impacts
│   ├── off_land_commodity/          # Off-land commodity economics
│   └── land_use_culling.py          # Land use optimization culling
├── solvers/                         # Optimization solvers and algorithms
│   ├── input_data.py                # GUROBI solver input preparation
│   └── solver.py                    # GUROBI solver interface
├── tests/                           # Unit and integration tests
└── tools/                           # Utility tools and scripts
    ├── create_task_runs/            # Task execution and batch processing
    │   ├── bash_scripts/            # Shell scripts and conda environment
    │   ├── create_grid_search_tasks.py  # Grid search task generation
    │   ├── helpers.py               # Task run utilities
    │   └── parameters.py            # Task run parameters
    ├── Manual_jupyter_books/        # Documentation notebooks
    │   ├── helpers/                 # Notebook helper functions
    │   └── asset/                   # Notebook assets and data descriptions
    ├── report/                      # Reporting and visualization system
    │   ├── VUE_modules/             # Vue.js 3 interactive reporting dashboard
    │   │   ├── components/          # Reusable Vue components
    │   │   ├── data/                # Chart data files (68 JS files)
    │   │   │   ├── chart_option/    # Chart configuration options
    │   │   │   ├── geo/             # Geographic boundary data
    │   │   │   └── map_layers/      # Map layer data (24 JS files)
    │   │   ├── dataTransform/       # Data transformation scripts
    │   │   ├── lib/                 # JavaScript libraries (Vue, Leaflet, Highcharts)
    │   │   ├── routes/              # Vue router configuration
    │   │   ├── services/            # Data and map services
    │   │   ├── views/               # Vue view components (11 modules)
    │   │   ├── index.html           # Main HTML entry point
    │   │   └── index.js             # Vue application entry
    │   ├── Assets/                  # Color schemes and styling assets
    │   ├── data_tools/              # Data processing for reports
    │   ├── map_tools/               # Spatial visualization utilities
    │   ├── create_report_data.py    # Generate chart data files
    │   └── create_report_layers.py  # Generate map layer files
    ├── plotmap.py                   # Mapping utilities
    ├── spatializers.py              # Spatial data processing and upsampling
    └── write.py                     # Output writing functions

input/                               # Input data directory (requires separate download)
output/                              # Simulation outputs with interactive HTML reports
docs/                                # Documentation files
requirements.txt                     # Python package dependencies
pyproject.toml                       # Project configuration
```

## Troubleshooting

### Common Issues

**Memory Errors:**
- Ensure you have at least 32 GB RAM available
- Close other applications during simulation
- Consider running smaller scenarios first

**GUROBI License Issues:**
- Verify your license file location
- Check license expiration date
- Ensure your license supports the required model size

**Data Loading Errors:**
- Verify all required input files are present in `/input/`
- Check file permissions
- Ensure sufficient disk space


### Getting Help

1. Check the documentation in `docs/luto2-overview.pdf`
2. Review log files in `/output/<run_dir>/logs/`
3. Contact the development team: **b.bryan@deakin.edu.au**
4. Submit issues on GitHub: [github.com/land-use-trade-offs/luto-2.0](https://github.com/land-use-trade-offs/luto-2.0)

## System Requirements

**Minimum Requirements:**
- Python 3.10 or higher
- 16 GB RAM (32 GB recommended for large simulations)
- 50 GB available disk space for input data and outputs
- GUROBI optimization solver license (academic licenses available)

**Supported Operating Systems:**
- Windows 10/11
- macOS 10.15+
- Linux (Ubuntu 18.04+, CentOS 7+)

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/land-use-trade-offs/luto-2.0.git
cd luto-2.0
```

### 2. Set Up Environment

#### Option A: Conda Environment (Recommended)
```bash
# Create and activate the LUTO environment
conda env create -f luto/tools/create_task_runs/bash_scripts/conda_env.yml
conda activate luto

# Install additional pip dependencies
pip install gurobipy==11.0.2 numpy_financial==1.0.0 tables==3.9.2
```

#### Option B: pip install
```bash
# Create virtual environment
python -m venv luto_env
source luto_env/bin/activate  # On Windows: luto_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure GUROBI Solver
LUTO2 requires GUROBI for optimization. Follow these steps:
```bash
# 1) Set up your GUROBI license (academic license available at gurobi.com)
# 2) Place your gurobi.lic file in the appropriate directory
```

### 4. Obtain Input Data
The LUTO2 input database is approximately 40 GB and contains sensitive data. 
Please contact **b.bryan@deakin.edu.au** to request access to the input dataset.

### 5. Run Tests (Optional)
```bash
# Run all tests using pytest
python -m pytest

# Run specific test modules
python -m pytest luto/tests/
```

## Running LUTO2

### Basic Simulation
```python
import luto.simulation as sim

# Load input data and settings
data = sim.load_data()

# Run simulation with default parameters
results = sim.run(data=data)
```

### Advanced Configuration
```python
import luto.simulation as sim
import luto.settings as settings

# Customize simulation settings
settings.RESFACTOR = 10                             # 10 makes the spatial resolution to ~10km. 
settings.SIM_YEARS = [2010, 2020, 2030, 2040, 2050]

settings.WATER_LIMITS = 'on'                        # 'on' or 'off'. 
settings.GHG_EMISSIONS_LIMITS = 'high'              # 'off', 'low', 'medium', or 'high'
settings.BIODIVERSITY_TARGET_GBF_2 = 'high'         # 'off', 'low', 'medium', or 'high'
settings.BIODIVERSITY_TARGET_GBF_3  = 'off'        # 'off', 'medium', 'high', or 'USER_DEFINED'   
settings.BIODIVERSITY_TARGET_GBF_4_SNES =  'off'    # 'on' or 'off'.
settings.BIODIVERSITY_TARGET_GBF_4_ECNES = 'off'    # 'on' or 'off'.
settings.BIODIVERSITY_TARGET_GBF_8 = 'off'          # 'on' or 'off'.

# Load data with custom parameters
data = sim.load_data()

# Run simulation
sim.run(data=data)
```

### Viewing Results
After execution, results are saved in the `/output/<timestamp>/` directory:

1. **Interactive HTML Dashboard:** 
   ```
   /output/<run_dir>/DATA_REPORT/index.html
   ```
   A Vue.js 3 based interactive dashboard featuring:
   - **Multi-module Analysis:** Area, Economics, GHG, Production, Water, Biodiversity
   - **Progressive Data Selection:** Region → Category → Water/AgMgt → Landuse hierarchies
   - **Dual Visualization:** Charts (Highcharts) and Maps (Leaflet) for all data types
   - **Dynamic Filtering:** Responsive dropdowns with cascading selection updates
   - **Export Capabilities:** Chart and map export functionality
   - **11 Specialized Views:** Individual modules for detailed analysis

2. **Raw Data Outputs:**
   - **NetCDF Files:** Spatial datasets (`.nc`) for each year and variable
   - **CSV Files:** Tabular data summaries for regional analysis
   - **GeoTIFF Files:** Raster outputs for GIS integration

3. **Execution Logs:** 
   - `LUTO_RUN__stdout.log`: Standard output logs
   - `LUTO_RUN__stderr.log`: Error and warning logs
   - Memory usage logs for performance monitoring

## Configuration

LUTO2 behavior can be customized through the `luto.settings` module. Key parameters include:

### Core Simulation Parameters
- `SIM_YEARS`: Simulation time period (default: 2020-2050 in 5-year steps)
- `SCENARIO`: Shared Socioeconomic Pathway (SSP1-SSP5)
- `RCP`: Representative Concentration Pathway (e.g., 'rcp4p5')
- `OBJECTIVE`: Optimization objective ('maxprofit' or 'mincost')

### Environmental Constraints
- `GHG_EMISSIONS_LIMITS`: Greenhouse gas emission targets ('off', 'low', 'medium', 'high')
- `WATER_LIMITS`: Whether to enforce water yield constraints ('on' or 'off')
- `BIODIVERSITY_TARGET_GBF_2`: Global Biodiversity Framework Target 2 ('off', 'low', 'medium', 'high')
- `BIODIVERSITY_TARGET_GBF_3`: Conservation targets for vegetation types ('off', 'medium', 'high')

### Land Use Options
- `NON_AG_LAND_USES`: Enable/disable non-agricultural land uses (Environmental Plantings, Carbon Plantings, etc.)
- `AG_MANAGEMENTS`: Enable/disable agricultural management practices (Precision Agriculture, Biochar, etc.)
- `EXCLUDE_NO_GO_LU`: Whether to exclude certain land uses from specific areas

### Economic Parameters
- `CARBON_PRICES_FIELD`: Carbon pricing scenario ('Default', 'CONSTANT', etc.)
- `AMORTISE_UPFRONT_COSTS`: Whether to amortize establishment costs
- `DISCOUNT_RATE`: Discount rate for economic calculations (default: 7%)

### Solver Configuration
- `SOLVE_METHOD`: GUROBI algorithm selection (default: 2 for barrier method)
- `THREADS`: Number of parallel threads for optimization
- `FEASIBILITY_TOLERANCE`: Solver tolerance settings
- `VERBOSE`: Control solver output verbosity

### Output Control
- `PARALLEL_WRITE`: Use parallel processing for output generation
- `RESFACTOR`: Spatial resolution factor (1 = full resolution, >1 = coarser)

Refer to `luto/settings.py` for a complete list of configurable parameters and detailed descriptions.

## Copyright
Copyright 2024-now **Bryan, B.A., Williams, N., Archibald, C.L., de Haan, F., Wang, J., van Schoten, N., Hadjikakou, M., Sanson, J., Zyngier, R., Marcos-Martinez, R., Navarro, J., Gao, L., Aghighi, H., Armstrong, T., Bohl, H., Jaffe, P., Khan, M.S., Moallemi, E.A., Nazari, A., Pan, X., Steyl, D., and Thiruvady, D.R.**  

Copyright 2021-2023 **Fjalar J. de Haan and Brett A. Bryan, Deakin University.** (see `CITATION.cff`).

## License
LUTO2 is free software: you can redistribute it and/or modify it under the terms of the **GNU General Public License** as published by the **Free Software Foundation**, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but **without any warranty**; without even the implied warranty of **merchantability** or **fitness for a particular purpose**. See the **GNU General Public License** for more details.

You should have received a copy of the **GNU General Public License** along with this program. If not, see <https://www.gnu.org/licenses/>.

## Citation
> Bryan, B.A., Williams, N., Archibald, C.L., de Haan, F., Wang, J., van Schoten, N., Hadjikakou, M., Sanson, J., Zyngier, R., Marcos-Martinez, R., Navarro, J., Gao, L., Aghighi, H., Armstrong, T., Bohl, H., Jaffe, P., Khan, M.S., Moallemi, E.A., Nazari, A., Pan, X., Steyl, D., and Thiruvady, D.R. (2025). The Land-Use Trade-Offs Model Version 2 (LUTO2): an integrated land system model for Australia. Software Collection. https://github.com/land-use-trade-offs/luto-2.0

## Contributing

We welcome contributions to LUTO2! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

For major changes, please open an issue first to discuss proposed modifications.

## Acknowledgments

LUTO2 was developed through a collaboration between:
- **Deakin University** - Centre for Integrative Ecology
- **Climateworks Centre** - Land Use Futures program
- **CSIRO** - Research contributions

This work is supported by funding from various Australian research councils and industry partners. We acknowledge the traditional custodians of the lands on which this research was conducted.

