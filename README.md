# Carbon-Aware Electricity Pricing for Switzerland

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Simulation framework for evaluating carbon-aware dynamic electricity pricing in Switzerland. This repository accompanies the Master's thesis:

> **Reducing Emissions Through Carbon-Aware Electricity Pricing: A Swiss Simulation Study**  

## Overview

This project investigates whether incorporating real-time carbon intensity signals into electricity prices can improve the temporal alignment between residential demand and low-carbon electricity availability in Switzerland.

### Key Findings

- **CA-Hourly reverses demand-carbon misalignment**: Correlation changes from +0.24 (TOU) to -0.39 (CA-Hourly), a Δρ of -0.63
- **19.1% emission reduction** through temporal demand shifting alone
- **CA-Hourly outperforms CA-CPP by 73%** for emission reduction (19.1% vs 11.0%)
- Results robust across 4 years (2021-2024) and 27 parameter sensitivity combinations

## Repository Structure

```
carbon-aware-pricing-ch/
├── data/                       # Data directory
│   ├── CH_2021_hourly.csv
│   ├── CH_2022_hourly.csv
│   ├── CH_2023_hourly.csv
│   ├── CH_2024_hourly.csv
│   ├── Energy_Statistic_CH_2021.xlsx
│   ├── Energy_Statistic_CH_2022.xlsx
│   ├── Energy_Statistic_CH_2023.xlsx
│   └── Energy_Statistic_CH_2024.xlsx
├── modules/                    # Core simulation modules
│   ├── __init__.py
│   ├── parameters.py           # Model parameters and configuration
│   ├── data_pipeline.py        # Data loading and preprocessing
│   ├── eda.py                  # Exploratory data analysis
│   ├── pricing.py              # Pricing scheme implementations
│   ├── behavioral.py           # Demand response model
│   ├── hypothesis.py           # Hypothesis testing framework
│   ├── sensitivity.py          # Sensitivity analysis
│   ├── rebound.py              # Rebound effect analysis
│   ├── exporter.py             # Results export utilities
│   └── visualization.py        # Figure generation
├── results/                    # Output directory
│   ├── figures/                # Generated figures (PDF)
│   ├── #.txt                   # Summary reports
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
- **Source**: [Electricity Maps](https://app.electricitymaps.com/datasets?zone=CH)
- **Files**: `CH_YYYY_hourly.csv` for years 2021-2024
- **Variables**: Hourly carbon intensity (gCO₂eq/kWh), renewable percentage

### Consumption Data
- **Source**: [Swiss Federal Office of Energy (SFOE)](https://www.swissgrid.ch/en/home/operation/grid-data/generation.html) via Swissgrid
- **Files**: `Energy_Statistic_CH_YYYY.xlsx`
- **Variables**: 15-minute national electricity consumption

Place downloaded files in the `data/` directory with the exact filenames as specified.

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

The simulation compares three pricing schemes:

1. **TOU (Time-of-Use)** — Baseline
   - Peak: 07:00-09:00, 18:00-21:00 (1.30×)
   - Off-peak: 23:00-07:00 (0.70×)
   - Standard: all other hours (1.00×)
   - No carbon signal

2. **CA-CPP (Carbon-Aware Critical Peak Pricing)** — Event-Based Intervention
   ```
   P_CA-CPP(t) = P_TOU(t) × C(t)
   
   where C(t) = 3.0  if CI(t) > CI_90  (critical event)
                1.0  otherwise
   ```
   - Discrete signal: high prices only during top 10% carbon hours
   - Builds on TOU structure

3. **CA-Hourly (Carbon-Aware Hourly Pricing)** — More Frequent Intervention
   ```
   P_CA-Hourly(t) = P_base × [1 + α × (CI(t) - CI_ref) / CI_ref]
   ```
   - α = 0.40 (carbon sensitivity)
   - Price bounds: [0.50×, 2.00×] of base price
   - Hourly signal: prices reflect full carbon intensity distribution

### Behavioral Model

```
D_new(t) = D_baseline(t) × (P_new / P_baseline)^ε_eff
```

Where effective elasticity incorporates:
- **Base elasticity**: ε = -0.35 (Swiss-specific, Filippini 2011)
- **Technology amplification**: τ = 0.47 (Faruqui & Sergici 2010)
- **Loss aversion**: λ = 2.5 for price increases (Kahneman & Tversky 1979)
- **Thermal comfort floor**: 60% minimum demand (Chen et al. 2023)

## Results

### Hypothesis 1: Temporal Alignment

| Year | ρ_TOU | ρ_CA-Hourly | Δρ | Decision |
|------|-------|-------------|-----|----------|
| 2021 | +0.251 | -0.503 | -0.754 | Reject H₀ |
| 2022 | +0.096 | -0.515 | -0.611 | Reject H₀ |
| 2023 | +0.335 | -0.451 | -0.786 | Reject H₀ |
| 2024 | +0.370 | -0.464 | -0.834 | Reject H₀ |

**Threshold**: Δρ ≤ -0.15 (derived from Hao et al. 2024 meta-analysis)

### Emission Reductions

| Year | TOU (ktCO₂) | CA-Hourly (ktCO₂) | CA-CPP (ktCO₂) | CA-Hourly Red. | CA-CPP Red. |
|------|-------------|-------------------|----------------|----------------|-------------|
| 2021 | 4,079 | 3,284 | 3,672 | 19.5% | 10.0% |
| 2022 | 5,186 | 4,332 | 4,722 | 16.5% | 9.0% |
| 2023 | 1,914 | 1,478 | 1,616 | 22.8% | 15.5% |
| 2024 | 916 | 693 | 757 | 24.4% | 17.3% |
| **Total** | **12,095** | **9,787** | **10,768** | **19.1%** | **11.0%** |

### Key Insight

CA-Hourly outperforms CA-CPP because emission reduction requires responding to the *full distribution* of carbon intensity variation, not just extreme peaks. By triggering only during hours exceeding the 90th percentile, CA-CPP ignores 90% of the variation that CA-Hourly exploits.

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
- [Swiss Federal Office of Energy](https://www.swissgrid.ch/en/home/operation/grid-data/generation.html) for consumption data
