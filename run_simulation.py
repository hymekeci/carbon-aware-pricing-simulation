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
- Figure generation

Usage:
    python run_simulation.py                    # Single year (2024)
    python run_simulation.py --year 2023        # Different single year
    python run_simulation.py --all              # ALL YEARS (2021-2024) + sensitivity
    python run_simulation.py --sensitivity      # 2024 + sensitivity
    python run_simulation.py --no-figures       # Skip figures
"""

import sys
import argparse
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import pandas as pd

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
    run_sensitivity_analysis
)


def run_single_year_simulation(data_dir: str = "data",
                                output_dir: str = "results",
                                analysis_year: int = 2024) -> Tuple[pd.DataFrame, Dict]:
    """
    Execute simulation for a single year.
    
    Args:
        data_dir: Path to data directory
        output_dir: Path to output directory
        analysis_year: Year for analysis
        
    Returns:
        Tuple of (simulation data, all results)
    """
    print(f"\n{'='*70}")
    print(f"SINGLE YEAR SIMULATION - {analysis_year}")
    print(f"{'='*70}")
    
    # Initialize parameters
    params = ModelParameters()
    
    # Step 1: Data Loading and Cleaning
    print(f"\n[1/6] Loading and cleaning data for {analysis_year}...")
    pipeline = DataPipeline(data_dir)
    data = pipeline.run_full_pipeline(
        carbon_years=[2021, 2022, 2023, 2024],
        consumption_years=[2021, 2022, 2023, 2024],
        analysis_year=analysis_year
    )
    
    # Step 2: Exploratory Data Analysis
    print("\n[2/6] Running exploratory analysis...")
    eda = ExploratoryAnalysis(data, pipeline.carbon_clean)
    eda_results = eda.run_full_eda()
    
    # Update carbon reference
    params.carbon_reference = eda_results['carbon_variability']['mean']
    
    # Step 3: Apply Pricing Schemes
    print("\n[3/6] Applying pricing schemes...")
    pricing = PricingModels(params)
    data = pricing.apply_all_pricing_schemes(data)
    
    # Step 4: Apply Behavioral Response
    print("\n[4/6] Modeling behavioral response...")
    behavioral = BehavioralResponseModel(params)
    data = behavioral.apply_demand_response(data)
    
    # Step 5: Hypothesis Testing
    print("\n[5/6] Testing hypotheses...")
    hypothesis = HypothesisTesting(params)
    test_results = hypothesis.run_all_tests(data)
    
    # Step 6: Export Results
    print("\n[6/6] Exporting results...")
    exporter = ResultsExporter(output_dir)
    exporter.export_simulation_data(data, filename=f"simulation_results_{analysis_year}.csv")
    exporter.export_summary_report(params, eda_results, test_results, 
                                   filename=f"summary_report_{analysis_year}.txt")
    
    # Compile results
    all_results = {
        'year': analysis_year,
        'parameters': params.__dict__,
        'eda': eda_results,
        'hypothesis_tests': test_results,
        'data_cleaning': pipeline.cleaning_report
    }
    
    # Print summary
    h1 = test_results['rq1']
    emissions = test_results['emissions']
    print(f"\n  Year {analysis_year} Results:")
    print(f"    Δρ = {h1['delta_rho_carbon']:+.3f}")
    print(f"    H₁: {'✓ SUPPORTED' if h1['h1_supported'] else '✗ NOT SUPPORTED'}")
    print(f"    Emissions: {emissions['reduction_carbon_pct']:.1f}%")
    
    print(f"\n{'='*70}")
    print(f"✓ {analysis_year} COMPLETE")
    print(f"{'='*70}")
    
    return data, all_results


def run_multi_year_analysis(data_dir: str = "data",
                             output_dir: str = "results",
                             years: List[int] = [2021, 2022, 2023, 2024]) -> Tuple[pd.DataFrame, Dict]:
    """
    Execute simulation for all years and create combined analysis.
    
    Args:
        data_dir: Path to data directory
        output_dir: Path to output directory
        years: List of years to analyze
        
    Returns:
        Tuple of (combined data, multi-year results)
    """
    print(f"\n{'='*70}")
    print(f"MULTI-YEAR ANALYSIS ({years[0]}-{years[-1]})")
    print(f"{'='*70}")
    
    all_year_data = []
    all_year_results = []
    
    # Run simulation for each year
    for year in years:
        data, results = run_single_year_simulation(data_dir, output_dir, year)
        all_year_data.append(data)
        all_year_results.append(results)
    
    # Combine all years
    print(f"\n{'='*70}")
    print("COMBINING MULTI-YEAR DATA")
    print(f"{'='*70}")
    
    combined_data = pd.concat(all_year_data, ignore_index=True)
    
    # Save combined data
    exporter = ResultsExporter(output_dir)
    exporter.export_simulation_data(combined_data, filename="simulation_results_combined.csv")
    
    # Calculate multi-year statistics
    multi_year_summary = {
        'years_analyzed': years,
        'total_hours': len(combined_data),
        'year_results': []
    }
    
    for results in all_year_results:
        year = results['year']
        h1 = results['hypothesis_tests']['rq1']
        emissions = results['hypothesis_tests']['emissions']
        
        multi_year_summary['year_results'].append({
            'year': year,
            'delta_rho': h1['delta_rho_carbon'],
            'h1_supported': h1['h1_supported'],
            'emission_reduction_pct': emissions['reduction_carbon_pct'],
            'mean_ci': results['eda']['carbon_variability']['mean'],
            'cv_ci': results['eda']['carbon_variability']['cv_percent']
        })
    
    # Print multi-year summary
    print(f"\n{'='*70}")
    print("MULTI-YEAR SUMMARY")
    print(f"{'='*70}")
    print(f"\nTotal hours analyzed: {len(combined_data):,}")
    print(f"\nYear-by-Year Results:")
    print(f"{'Year':<6} {'Δρ':>8} {'H₁':>12} {'Emissions':>12} {'Mean CI':>10} {'CV':>8}")
    print("-" * 70)
    
    for yr in multi_year_summary['year_results']:
        h1_status = '✓ YES' if yr['h1_supported'] else '✗ NO'
        print(f"{yr['year']:<6} {yr['delta_rho']:>+8.3f} {h1_status:>12} "
              f"{yr['emission_reduction_pct']:>11.1f}% {yr['mean_ci']:>10.1f} {yr['cv_ci']:>7.1f}%")
    
    # Calculate averages
    avg_delta_rho = sum(yr['delta_rho'] for yr in multi_year_summary['year_results']) / len(years)
    avg_emission_red = sum(yr['emission_reduction_pct'] for yr in multi_year_summary['year_results']) / len(years)
    all_supported = all(yr['h1_supported'] for yr in multi_year_summary['year_results'])
    
    print("-" * 70)
    print(f"{'AVG':<6} {avg_delta_rho:>+8.3f} {'✓ ALL' if all_supported else '✗ MIXED':>12} "
          f"{avg_emission_red:>11.1f}%")
    
    print(f"\n{'='*70}")
    print("✓ MULTI-YEAR ANALYSIS COMPLETE")
    print(f"{'='*70}")
    
    multi_year_results = {
        'combined_data': combined_data,
        'summary': multi_year_summary,
        'individual_results': all_year_results
    }
    
    return combined_data, multi_year_results


def run_sensitivity_analysis_wrapper(data_dir: str = "data",
                                      output_dir: str = "results",
                                      analysis_year: int = 2024) -> pd.DataFrame:
    """
    Wrapper for sensitivity analysis.
    
    Loads data and calls the module's run_sensitivity_analysis function.
    
    Args:
        data_dir: Path to data directory
        output_dir: Path to output directory
        analysis_year: Year for analysis
        
    Returns:
        DataFrame with sensitivity results
    """
    print(f"\n{'='*70}")
    print(f"SENSITIVITY ANALYSIS - {analysis_year}")
    print(f"{'='*70}")
    
    # Load and prepare data
    pipeline = DataPipeline(data_dir)
    data = pipeline.run_full_pipeline(
        carbon_years=[2021, 2022, 2023, 2024],
        consumption_years=[2021, 2022, 2023, 2024],
        analysis_year=analysis_year
    )
    
    # Call the module function
    results_df = run_sensitivity_analysis(data, output_dir=output_dir, verbose=True)
    
    return results_df


def generate_figures(data: pd.DataFrame, 
                    sensitivity_data: Optional[pd.DataFrame] = None,
                    output_dir: str = "results"):
    """
    Generate all thesis figures.
    
    Args:
        data: Simulation results DataFrame (can be single year or combined)
        sensitivity_data: Optional sensitivity analysis results
        output_dir: Output directory for figures
    """
    print(f"\n{'='*70}")
    print("GENERATING FIGURES")
    print(f"{'='*70}")
    
    n_hours = len(data)
    years_covered = data['datetime'].dt.year.unique()
    print(f"  Data: {n_hours:,} hours from years {sorted(years_covered)}")
    
    figures = ThesisFigures(f"{output_dir}/figures")
    figures.generate_all_figures(data, sensitivity_data)
    
    print(f"\n{'='*70}")
    print("✓ FIGURE GENERATION COMPLETE")
    print(f"{'='*70}")


def main(analysis_year: Optional[int] = None,
         run_all_years: bool = False,
         run_sensitivity: bool = False,
         generate_figs: bool = True):
    """
    Run complete analysis pipeline.
    
    Args:
        analysis_year: Single year for analysis (if not run_all_years)
        run_all_years: Run analysis for all years (2021-2024)
        run_sensitivity: Whether to run sensitivity analysis
        generate_figs: Whether to generate figures
    
    Returns:
        Tuple of (data, results)
    """
    print(f"\n{'='*80}")
    print(" "*25 + "COMPLETE PIPELINE")
    
    if run_all_years:
        print(f" "*20 + "Mode: MULTI-YEAR (2021-2024)")
    else:
        print(f" "*25 + f"Mode: SINGLE YEAR ({analysis_year})")
    
    print(f" "*15 + f"Sensitivity: {run_sensitivity} | Figures: {generate_figs}")
    print(f"{'='*80}")
    
    # Main simulation
    if run_all_years:
        # Run all years and combine
        data, results = run_multi_year_analysis(
            data_dir="data",
            output_dir="results",
            years=[2021, 2022, 2023, 2024]
        )
    else:
        # Run single year
        if analysis_year is None:
            analysis_year = 2024
        
        data, results = run_single_year_simulation(
            data_dir="data",
            output_dir="results",
            analysis_year=analysis_year
        )
    
    # Optional: Sensitivity analysis
    sensitivity_data = None
    if run_sensitivity:
        # Run sensitivity on 2024 data (most recent)
        sensitivity_data = run_sensitivity_analysis_wrapper(
            data_dir="data",
            output_dir="results",
            analysis_year=2024
        )
    
    # Optional: Generate figures
    if generate_figs:
        generate_figures(data, sensitivity_data, output_dir="results")
    
    # Final summary
    print(f"\n{'='*80}")
    print(" "*30 + "PIPELINE COMPLETE")
    print(f"{'='*80}")
    
    print(f"\n📁 Outputs in results/:")
    if run_all_years:
        print(f"   ✓ simulation_results_2021.csv")
        print(f"   ✓ simulation_results_2022.csv")
        print(f"   ✓ simulation_results_2023.csv")
        print(f"   ✓ simulation_results_2024.csv")
        print(f"   ✓ simulation_results_combined.csv  (ALL YEARS)")
        print(f"   ✓ summary_report_YYYY.txt (4 files)")
    else:
        print(f"   ✓ simulation_results_{analysis_year}.csv")
        print(f"   ✓ summary_report_{analysis_year}.txt")
    
    if run_sensitivity:
        print(f"   ✓ sensitivity_analysis.csv")
    if generate_figs:
        print(f"   ✓ figures/ (14 PDFs)")
    
    print(f"\n📊 Key Results:")
    if run_all_years:
        summary = results['summary']
        for yr in summary['year_results']:
            status = '✓' if yr['h1_supported'] else '✗'
            print(f"   {yr['year']}: Δρ = {yr['delta_rho']:+.3f} ({status}), "
                  f"Emissions: {yr['emission_reduction_pct']:.1f}%")
    else:
        h1 = results['hypothesis_tests']['rq1']
        emissions = results['hypothesis_tests']['emissions']
        print(f"   Δρ = {h1['delta_rho_carbon']:+.3f} (threshold: ≤ -0.15)")
        print(f"   H₁ supported: {'✓ YES' if h1['h1_supported'] else '✗ NO'}")
        print(f"   Emission reduction: {emissions['reduction_carbon_pct']:.1f}%")
    
    print(f"\n{'='*80}\n")
    
    return data, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run complete carbon-aware pricing simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python run_simulation.py                    # Single year (2024)
            python run_simulation.py --year 2023        # Different single year
            python run_simulation.py --all              # ALL YEARS + sensitivity + figures
            python run_simulation.py --sensitivity      # 2024 + sensitivity
            python run_simulation.py --no-figures       # Skip figure generation
                    """
    )
    parser.add_argument("--year", type=int, default=None,
                       help="Single year for analysis (default: 2024)")
    parser.add_argument("--all", action="store_true",
                       help="Run ALL YEARS (2021-2024) + sensitivity")
    parser.add_argument("--sensitivity", action="store_true",
                       help="Run sensitivity analysis")
    parser.add_argument("--no-figures", action="store_true",
                       help="Skip figure generation")
    
    args = parser.parse_args()
    
    # --all flag enables multi-year + sensitivity
    if args.all:
        run_all_years = True
        run_sensitivity = True
        analysis_year = None
    else:
        run_all_years = False
        run_sensitivity = args.sensitivity
        analysis_year = args.year if args.year else 2024
    
    data, results = main(
        analysis_year=analysis_year,
        run_all_years=run_all_years,
        run_sensitivity=run_sensitivity,
        generate_figs=not args.no_figures
    )