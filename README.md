# Carbon-Aware Electricity Pricing for Switzerland

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Simulation framework for evaluating carbon-aware dynamic electricity pricing in Switzerland. This repository accompanies the Master's thesis:

> **Reducing Emissions Through Carbon-Aware Electricity Pricing: A Swiss Simulation Study**  

## Overview

This project investigates whether incorporating real-time carbon intensity signals into electricity prices can improve the temporal alignment between residential demand and low-carbon electricity availability in Switzerland.

### Key Findings

- **Carbon-aware pricing reverses demand-carbon misalignment**: Correlation changes from +0.24 (TOU) to -0.39 (carbon-aware), a Δρ of -0.63
- **19.1% emission reduction** through temporal demand shifting alone
- **73% more effective** than Critical Peak Pricing for emission reduction
- Results robust across 4 years (2021-2024) and 27 parameter sensitivity combinations

## Repository Structure

```
carbon-aware-pricing-ch/
├── data/                       # Data directory
│   ├── CH_2021_hourly.csv
│   ├── CH_2022_hourly.csv
│   ├── CH_2023_hourly.csv
│   ├── CH_2024_hourly.csv
│   └── Energy_Statistic_CH_*.xlsx
├── modules/                    # Core simulation modules
│   ├── __init__.py
│   ├── parameters.py           # Model parameters and configuration
│   ├── data_pipeline.py        # Data loading and preprocessing
│   ├── eda.py                  # Exploratory data analysis
│   ├── pricing.py              # Pricing scheme implementations
│   ├── behavioral.py           # Demand response model
│   ├── hypothesis.py           # Hypothesis testing framework
│   ├── sensitivity.py          # Sensitivity analysis
│   ├── exporter.py             # Results export utilities
│   └── visualization.py        # Figure generation
├── results/                    # Output directory
│   ├── figures/                # Generated figures (PDF)
│   └── *.csv                   # Simulation results
├── run_simulation.py           # Main entry point
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # This file
```

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/hymekeci/carbon-aware-pricing-ch.git
cd carbon-aware-pricing-ch
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Sources

Due to licensing restrictions, raw data files are not included. You can obtain them from:

### Carbon Intensity Data
- **Source**: [Electricity Maps](https://www.electricitymaps.com/)
- **Files**: `CH_YYYY_hourly.csv` for years 2021-2024
- **Variables**: Hourly carbon intensity (gCO₂eq/kWh), renewable percentage

### Consumption Data
- **Source**: [Swiss Federal Office of Energy (SFOE)](https://www.bfe.admin.ch/) via Swissgrid
- **Files**: `Energy_Statistic_CH_YYYY.xlsx`
- **Variables**: 15-minute national electricity consumption

Place downloaded files in the `data/` directory.

## Usage

### Quick Start

Run the complete simulation for a single year:
```bash
python run_simulation.py
```

### Command Line Options

```bash
# Single year analysis (default: 2024)
python run_simulation.py --year 2024

# Multi-year analysis (2021-2024)
python run_simulation.py --all

# Include sensitivity analysis
python run_simulation.py --sensitivity

# Skip figure generation
python run_simulation.py --no-figures

# Full analysis with all options
python run_simulation.py --all --sensitivity
```

### Example Output

```
========================================================================
COMPLETE PIPELINE
Mode: MULTI-YEAR (2021-2024)
Sensitivity: True | Figures: True
========================================================================

[1/6] Loading and cleaning data...
[2/6] Running exploratory analysis...
[3/6] Applying pricing schemes...
[4/6] Modeling behavioral response...
[5/6] Testing hypotheses...
[6/6] Exporting results...

Key Results:
   Δρ = -0.635 (threshold: ≤ -0.15)
   H₁ supported: ✓ YES
   Emission reduction: 19.1%
```

## Methodology

### Pricing Schemes

1. **Time-of-Use (TOU)** - Baseline
   - Peak: 07:00-09:00, 18:00-21:00 (1.30×)
   - Off-peak: 23:00-07:00 (0.70×)
   - Standard: all other hours (1.00×)

2. **Carbon-Aware Dynamic Pricing** - Intervention
   ```
   P(t) = P_base × [1 + α × (CI(t) - CI_ref) / CI_ref]
   ```
   - α = 0.40 (carbon sensitivity)
   - Price bounds: [0.50×, 2.00×] of base price

3. **Critical Peak Pricing (CPP)** - Alternative
   - 3× multiplier when CI > 90th percentile
   - TOU structure otherwise

### Behavioral Model

```
D_new(t) = D_baseline(t) × (P_new / P_baseline)^ε_eff
```

Where effective elasticity incorporates:
- Base elasticity: ε = -0.35 (Swiss-specific)
- Technology amplification: τ = 0.47
- Loss aversion: λ = 2.5 (for price increases)
- Thermal comfort floor: 60% minimum demand

## Results

### Hypothesis 1: Temporal Alignment

| Year | ρ_TOU | ρ_Carbon | Δρ | Decision |
|------|-------|----------|-----|----------|
| 2021 | +0.251 | -0.503 | -0.754 | Reject H₀ |
| 2022 | +0.096 | -0.515 | -0.611 | Reject H₀ |
| 2023 | +0.335 | -0.451 | -0.786 | Reject H₀ |
| 2024 | +0.370 | -0.464 | -0.834 | Reject H₀ |

### Emission Reductions

| Year | TOU (ktCO₂) | Carbon-Aware (ktCO₂) | Reduction |
|------|-------------|----------------------|-----------|
| 2021 | 4,079 | 3,284 | 19.5% |
| 2022 | 5,186 | 4,332 | 16.5% |
| 2023 | 1,914 | 1,478 | 22.8% |
| 2024 | 916 | 693 | 24.4% |
| **Total** | **12,095** | **9,787** | **19.1%** |

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{mekeci2025carbon,
  author  = {Mekeci, Halil Yavuzhan},
  title   = {Reducing Emissions Through Carbon-Aware Electricity Pricing: 
             A Swiss Simulation Study},
  school  = {University of Geneva},
  year    = {2025},
  type    = {Master's Thesis}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Electricity Maps](https://www.electricitymaps.com/) for carbon intensity data
- Swiss Federal Office of Energy for consumption data


Project Link: [https://github.com/hymekeci/carbon-aware-pricing-ch](https://github.com/hymekeci/carbon-aware-pricing-ch)