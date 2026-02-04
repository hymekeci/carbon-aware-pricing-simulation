"""
Complete Simulation Pipeline
=============================

Runs the full carbon-aware pricing simulation including:
- Data loading and cleaning
- Exploratory data analysis
- Pricing scheme application
- Behavioral response modeling
- Hypothesis testing
- Multi-year analysis
- Sensitivity analysis
- Rebound effect analysis
- Figure generation

Usage:
    python run_simulation.py                    # Single year (2024)
    python run_simulation.py --year 2023        # Different single year
    python run_simulation.py --all              # ALL YEARS (2021-2024) + sensitivity + rebound
    python run_simulation.py --sensitivity      # 2024 + sensitivity
    python run_simulation.py --rebound          # 2024 + rebound analysis
    python run_simulation.py --no-figures       # Skip figures
"""

import sys
import argparse
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional, List

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from modules import (
    ModelParameters,
    DataPipeline,
    ExploratoryAnalysis,
    PricingModels,
    BehavioralResponseModel,
    HypothesisTesting,
    ResultsExporter,
    ThesisFigures,
    run_sensitivity_analysis,
    run_rebound_analysis  # NEW
)


def run_single_year_simulation(data_dir: str = "data",
                                output_dir: str = "results",
                                analysis_year: int = 2024) -> Tuple[pd.DataFrame, Dict]:
    """Execute simulation for a single year."""
    print(f"\n{'='*70}")
    print(f"SINGLE YEAR SIMULATION - {analysis_year}")
    print(f"{'='*70}")

    params = ModelParameters()

    print(f"\n[1/6] Loading and cleaning data for {analysis_year}...")
    pipeline = DataPipeline(data_dir)
    data = pipeline.run_full_pipeline(
        carbon_years=[2021, 2022, 2023, 2024],
        consumption_years=[2021, 2022, 2023, 2024],
        analysis_year=analysis_year
    )

    print("\n[2/6] Running exploratory analysis...")
    eda = ExploratoryAnalysis(data, pipeline.carbon_clean)
    eda_results = eda.run_full_eda()
    params.carbon_reference = eda_results['carbon_variability']['mean']

    print("\n[3/6] Applying pricing schemes...")
    pricing = PricingModels(params)
    data = pricing.apply_all_pricing_schemes(data)

    print("\n[4/6] Modeling behavioral response...")
    behavioral = BehavioralResponseModel(params)
    data = behavioral.apply_demand_response(data)

    print("\n[5/6] Testing hypotheses...")
    hypothesis = HypothesisTesting(params)
    test_results = hypothesis.run_all_tests(data)

    print("\n[6/6] Exporting results...")
    exporter = ResultsExporter(output_dir)
    exporter.export_simulation_data(data, filename=f"simulation_results_{analysis_year}.csv")
    exporter.export_summary_report(params, eda_results, test_results, 
                                   filename=f"summary_report_{analysis_year}.txt")

    all_results = {
        'year': analysis_year,
        'parameters': params.__dict__,
        'eda': eda_results,
        'hypothesis_tests': test_results,
        'data_cleaning': pipeline.cleaning_report
    }

    h1 = test_results['rq1']
    emissions = test_results['emissions']
    print(f"\n  Year {analysis_year}: Δρ = {h1['delta_rho_carbon']:+.3f}, "
          f"H₁: {'✓' if h1['h1_supported'] else '✗'}, "
          f"Emissions: {emissions['reduction_carbon_pct']:.1f}%")

    return data, all_results


def run_multi_year_analysis(data_dir: str = "data",
                             output_dir: str = "results",
                             years: List[int] = [2021, 2022, 2023, 2024]) -> Tuple[pd.DataFrame, Dict]:
    """Execute simulation for all years and create combined analysis."""
    print(f"\n{'='*70}")
    print(f"MULTI-YEAR ANALYSIS ({years[0]}-{years[-1]})")
    print(f"{'='*70}")

    all_year_data = []
    all_year_results = []

    for year in years:
        data, results = run_single_year_simulation(data_dir, output_dir, year)
        all_year_data.append(data)
        all_year_results.append(results)

    combined_data = pd.concat(all_year_data, ignore_index=True)

    exporter = ResultsExporter(output_dir)
    exporter.export_simulation_data(combined_data, filename="simulation_results_combined.csv")

    multi_year_summary = {
        'years_analyzed': years,
        'total_hours': len(combined_data),
        'year_results': [{
            'year': r['year'],
            'delta_rho': r['hypothesis_tests']['rq1']['delta_rho_carbon'],
            'h1_supported': r['hypothesis_tests']['rq1']['h1_supported'],
            'emission_reduction_pct': r['hypothesis_tests']['emissions']['reduction_carbon_pct'],
            'mean_ci': r['eda']['carbon_variability']['mean'],
            'cv_ci': r['eda']['carbon_variability']['cv_percent']
        } for r in all_year_results]
    }

    # Print summary
    print(f"\n{'='*70}\nMULTI-YEAR SUMMARY\n{'='*70}")
    print(f"{'Year':<6} {'Δρ':>8} {'H₁':>8} {'Reduction':>10}")
    print("-" * 40)
    for yr in multi_year_summary['year_results']:
        print(f"{yr['year']:<6} {yr['delta_rho']:>+8.3f} {'✓' if yr['h1_supported'] else '✗':>8} "
              f"{yr['emission_reduction_pct']:>9.1f}%")

    return combined_data, {'summary': multi_year_summary, 'individual_results': all_year_results}


def generate_figures(data: pd.DataFrame, 
                    sensitivity_data: Optional[pd.DataFrame] = None,
                    rebound_data: Optional[pd.DataFrame] = None,
                    output_dir: str = "results"):
    """Generate all thesis figures."""
    print(f"\n{'='*70}\nGENERATING FIGURES\n{'='*70}")

    figures = ThesisFigures(f"{output_dir}/figures")
    figures.generate_all_figures(data, sensitivity_data, rebound_data)


def main(analysis_year: Optional[int] = None,
         run_all_years: bool = False,
         run_sensitivity: bool = False,
         run_rebound: bool = False,
         generate_figs: bool = True):
    """Run complete analysis pipeline."""

    print(f"\n{'='*80}")
    print(" "*25 + "CARBON-AWARE PRICING SIMULATION")
    mode = "MULTI-YEAR (2021-2024)" if run_all_years else f"SINGLE YEAR ({analysis_year or 2024})"
    print(" "*20 + f"Mode: {mode}")
    print(f"{'='*80}")

    # Main simulation
    if run_all_years:
        data, results = run_multi_year_analysis()
    else:
        data, results = run_single_year_simulation(analysis_year=analysis_year or 2024)

    # Optional analyses
    sensitivity_data = run_sensitivity_analysis(data, verbose=True) if run_sensitivity else None
    rebound_data, rebound_summary = run_rebound_analysis(data) if run_rebound else (None, None)

    # Generate figures
    if generate_figs:
        generate_figures(data, sensitivity_data, rebound_data)

    # Print final summary
    print(f"\n{'='*80}\n" + " "*30 + "COMPLETE\n" + "="*80)

    if rebound_summary:
        print(f"\n📉 Rebound: {rebound_summary['baseline_reduction']:.1f}% → "
              f"{rebound_summary['worst_case_reduction']:.1f}% (worst-case)")

    return data, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Carbon-aware pricing simulation")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--rebound", action="store_true")
    parser.add_argument("--no-figures", action="store_true")

    args = parser.parse_args()

    if args.all:
        run_all_years, run_sensitivity, run_rebound = True, True, True
        analysis_year = None
    else:
        run_all_years = False
        run_sensitivity, run_rebound = args.sensitivity, args.rebound
        analysis_year = args.year or 2024

    main(analysis_year, run_all_years, run_sensitivity, run_rebound, not args.no_figures)
