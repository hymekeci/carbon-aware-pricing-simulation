"""
Sensitivity Analysis Module
============================
Performs a comprehensive sensitivity analysis on the carbon-aware pricing simulation model.

Tests 27 combinations of behavioral parameters:
- α (alpha): Carbon weight [0.30, 0.40, 0.50]
- τ (tau): Technology amplification [0.30, 0.47, 0.60]
- λ (lambda): Loss aversion [1.5, 2.5, 3.5]

Outputs:
- sensitivity_full_grid.csv: All 27 combinations
- sensitivity_main_effects.csv: Main effects summary (9 rows)
- sensitivity_report.txt: Text summary report
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import itertools

from .parameters import ModelParameters
from .pricing import PricingModels
from .behavioral import BehavioralResponseModel


@dataclass
class SensitivityConfig:
    """Configuration for sensitivity analysis."""
    
    # Parameter values (Low, Mid, High)
    alpha_values: List[float] = field(default_factory=lambda: [0.30, 0.40, 0.50])
    tau_values: List[float] = field(default_factory=lambda: [0.30, 0.47, 0.60])
    lambda_values: List[float] = field(default_factory=lambda: [1.5, 2.5, 3.5])
    
    # Baseline values
    alpha_baseline: float = 0.40
    tau_baseline: float = 0.47
    lambda_baseline: float = 2.5
    
    # Fixed parameters
    base_elasticity: float = -0.35
    threshold: float = -0.15


class SensitivityAnalysis:
    """
    Sensitivity analysis for carbon-aware pricing simulation.
    
    Designed to work with existing run_simulation.py pipeline.
    """
    
    def __init__(self, config: Optional[SensitivityConfig] = None):
        """Initialize with optional custom config."""
        self.config = config or SensitivityConfig()
        self.full_grid_results = None
        self.main_effects_results = None
    
    def run_single_scenario(self, 
                            data: pd.DataFrame,
                            alpha: float,
                            tau: float,
                            lambda_la: float) -> Dict:
        """
        Run simulation for single parameter combination.
        
        Args:
            data: DataFrame with carbon_intensity and consumption_kwh
            alpha: Carbon weight parameter
            tau: Technology amplification factor
            lambda_la: Loss aversion coefficient
            
        Returns:
            Dictionary with scenario results
        """
        # Create parameters
        params = ModelParameters()
        params.carbon_weight_alpha = alpha
        params.base_elasticity = self.config.base_elasticity
        params.loss_aversion_factor = lambda_la
        params.carbon_reference = data['carbon_intensity'].mean()
        
        # Set technology amplification by adjusting penetration rates
        # τ = smart_meter_pen × 0.40 + thermostat_pen × 1.00
        if tau <= 0.32:
            params.smart_meter_penetration = tau / 0.40
            params.thermostat_penetration = 0.0
        else:
            params.smart_meter_penetration = 0.80
            params.thermostat_penetration = (tau - 0.32) / 1.00
        
        # Apply pricing
        pricing = PricingModels(params)
        test_data = pricing.apply_all_pricing_schemes(data.copy())
        
        # Apply behavioral response
        behavioral = BehavioralResponseModel(params)
        test_data = behavioral.apply_demand_response(test_data)
        
        # Calculate metrics
        rho_tou = test_data['demand_tou'].corr(test_data['carbon_intensity'])
        rho_carbon = test_data['demand_carbon'].corr(test_data['carbon_intensity'])
        rho_cpp = test_data['demand_cpp'].corr(test_data['carbon_intensity'])
        
        delta_rho = rho_carbon - rho_tou
        delta_rho_cpp = rho_cpp - rho_tou
        
        # Calculate emissions (ktCO2)
        e_tou = (test_data['demand_tou'] * test_data['carbon_intensity']).sum() / 1e9
        e_carbon = (test_data['demand_carbon'] * test_data['carbon_intensity']).sum() / 1e9
        e_cpp = (test_data['demand_cpp'] * test_data['carbon_intensity']).sum() / 1e9
        
        emission_reduction = (e_tou - e_carbon) / e_tou * 100
        emission_reduction_cpp = (e_tou - e_cpp) / e_tou * 100
        
        return {
            'alpha': alpha,
            'tau': tau,
            'lambda': lambda_la,
            'rho_tou': round(rho_tou, 4),
            'rho_carbon': round(rho_carbon, 4),
            'rho_cpp': round(rho_cpp, 4),
            'delta_rho': round(delta_rho, 4),
            'delta_rho_cpp': round(delta_rho_cpp, 4),
            'emission_reduction_pct': round(emission_reduction, 2),
            'emission_reduction_cpp_pct': round(emission_reduction_cpp, 2),
            'threshold_met': delta_rho <= self.config.threshold
        }
    
    def run_full_grid(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run all 27 parameter combinations.
        
        Args:
            data: Simulation data with carbon_intensity and consumption_kwh
            
        Returns:
            DataFrame with all 27 results
        """
        combinations = list(itertools.product(
            self.config.alpha_values,
            self.config.tau_values,
            self.config.lambda_values
        ))
        
        results = []
        for alpha, tau, lambda_la in combinations:
            result = self.run_single_scenario(data, alpha, tau, lambda_la)
            results.append(result)
        
        self.full_grid_results = pd.DataFrame(results)
        return self.full_grid_results
    
    def calculate_main_effects(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate main effects (vary one parameter, hold others at baseline).
        
        Returns DataFrame with 9 rows (3 per parameter).
        """
        results = []
        cfg = self.config
        
        # Alpha effect
        for i, alpha in enumerate(cfg.alpha_values):
            level = ['Low', 'Mid', 'High'][i]
            result = self.run_single_scenario(data, alpha, cfg.tau_baseline, cfg.lambda_baseline)
            results.append({
                'parameter': 'alpha',
                'level': level,
                'value': alpha,
                **result
            })
        
        # Tau effect
        for i, tau in enumerate(cfg.tau_values):
            level = ['Low', 'Mid', 'High'][i]
            result = self.run_single_scenario(data, cfg.alpha_baseline, tau, cfg.lambda_baseline)
            results.append({
                'parameter': 'tau',
                'level': level,
                'value': tau,
                **result
            })
        
        # Lambda effect
        for i, lambda_la in enumerate(cfg.lambda_values):
            level = ['Low', 'Mid', 'High'][i]
            result = self.run_single_scenario(data, cfg.alpha_baseline, cfg.tau_baseline, lambda_la)
            results.append({
                'parameter': 'lambda',
                'level': level,
                'value': lambda_la,
                **result
            })
        
        self.main_effects_results = pd.DataFrame(results)
        return self.main_effects_results
    
    def generate_report(self) -> str:
        """Generate text summary report."""
        if self.full_grid_results is None or self.main_effects_results is None:
            raise ValueError("Run analysis first")
        
        fg = self.full_grid_results
        me = self.main_effects_results
        cfg = self.config
        
        lines = []
        lines.append("=" * 75)
        lines.append("SENSITIVITY ANALYSIS REPORT")
        lines.append("Carbon-Aware Electricity Pricing - Robustness Check")
        lines.append("=" * 75)
        lines.append("")
        
        # Configuration
        lines.append("PARAMETER CONFIGURATION")
        lines.append("-" * 40)
        lines.append(f"  Alpha (carbon weight):     {cfg.alpha_values}")
        lines.append(f"  Tau (tech amplification):  {cfg.tau_values}")
        lines.append(f"  Lambda (loss aversion):    {cfg.lambda_values}")
        lines.append(f"  Baseline values:           alpha={cfg.alpha_baseline}, tau={cfg.tau_baseline}, lambda={cfg.lambda_baseline}")
        lines.append(f"  Total combinations:        {len(fg)}")
        lines.append("")
        
        # Main Effects Table
        lines.append("MAIN EFFECTS (one parameter varies, others at baseline)")
        lines.append("-" * 75)
        lines.append(f"{'Parameter':<12} {'Level':<6} {'Value':<8} {'Delta_rho':<12} {'Emission_Red':<12} {'Threshold':<10}")
        lines.append("-" * 75)
        
        for _, row in me.iterrows():
            threshold_str = "PASS" if row['threshold_met'] else "FAIL"
            lines.append(f"{row['parameter']:<12} {row['level']:<6} {row['value']:<8.2f} "
                        f"{row['delta_rho']:<+12.3f} {row['emission_reduction_pct']:<12.1f}% {threshold_str:<10}")
        
        lines.append("-" * 75)
        lines.append("")
        
        # Summary by Parameter
        lines.append("PARAMETER INFLUENCE SUMMARY")
        lines.append("-" * 40)
        
        for param in ['alpha', 'tau', 'lambda']:
            param_data = me[me['parameter'] == param]
            low = param_data[param_data['level'] == 'Low'].iloc[0]
            high = param_data[param_data['level'] == 'High'].iloc[0]
            
            delta_range = high['delta_rho'] - low['delta_rho']
            em_range = high['emission_reduction_pct'] - low['emission_reduction_pct']
            
            lines.append(f"  {param}:")
            lines.append(f"    Delta_rho: {low['delta_rho']:.3f} -> {high['delta_rho']:.3f} (change: {delta_range:+.3f})")
            lines.append(f"    Emission:  {low['emission_reduction_pct']:.1f}% -> {high['emission_reduction_pct']:.1f}% (change: {em_range:+.1f}%)")
        
        lines.append("")
        
        # Full Grid Summary
        lines.append("FULL GRID ANALYSIS (27 combinations)")
        lines.append("-" * 40)
        lines.append(f"  Delta_rho range:    {fg['delta_rho'].min():.3f} to {fg['delta_rho'].max():.3f}")
        lines.append(f"  Delta_rho mean:     {fg['delta_rho'].mean():.3f} (+/- {fg['delta_rho'].std():.3f})")
        lines.append(f"  Emission reduction: {fg['emission_reduction_pct'].min():.1f}% to {fg['emission_reduction_pct'].max():.1f}%")
        lines.append(f"  Emission mean:      {fg['emission_reduction_pct'].mean():.1f}% (+/- {fg['emission_reduction_pct'].std():.1f}%)")
        lines.append("")
        lines.append(f"  Threshold test (delta_rho <= {cfg.threshold}):")
        lines.append(f"    PASS: {fg['threshold_met'].sum()}/27 ({fg['threshold_met'].mean()*100:.0f}%)")
        lines.append("")
        
        # Best/Worst Cases
        best_idx = fg['emission_reduction_pct'].idxmax()
        worst_idx = fg['emission_reduction_pct'].idxmin()
        best = fg.loc[best_idx]
        worst = fg.loc[worst_idx]
        
        lines.append("EXTREME CASES")
        lines.append("-" * 40)
        lines.append(f"  Best case:  alpha={best['alpha']:.2f}, tau={best['tau']:.2f}, lambda={best['lambda']:.1f}")
        lines.append(f"              delta_rho={best['delta_rho']:.3f}, emission_red={best['emission_reduction_pct']:.1f}%")
        lines.append(f"  Worst case: alpha={worst['alpha']:.2f}, tau={worst['tau']:.2f}, lambda={worst['lambda']:.1f}")
        lines.append(f"              delta_rho={worst['delta_rho']:.3f}, emission_red={worst['emission_reduction_pct']:.1f}%")
        lines.append("")
        
        # Conclusion
        lines.append("ROBUSTNESS CONCLUSION")
        lines.append("-" * 40)
        if fg['threshold_met'].all():
            lines.append("  [ROBUST] All 27 parameter combinations exceed effectiveness threshold.")
            lines.append("  Findings do NOT depend critically on specific parameter choices.")
            lines.append("  Internal validity: SUPPORTED")
        else:
            n_fail = (~fg['threshold_met']).sum()
            lines.append(f"  [PARTIAL] {n_fail}/27 combinations fall below threshold.")
            lines.append("  Some sensitivity to parameter choices observed.")
        
        lines.append("")
        lines.append("=" * 75)
        
        return "\n".join(lines)
    
    def save_results(self, output_dir: str = "results"):
        """Save all results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Full grid CSV
        if self.full_grid_results is not None:
            self.full_grid_results.to_csv(output_path / 'sensitivity_full_grid.csv', index=False)
        
        # Main effects CSV
        if self.main_effects_results is not None:
            self.main_effects_results.to_csv(output_path / 'sensitivity_main_effects.csv', index=False)
        
        # Text report
        report = self.generate_report()
        with open(output_path / 'sensitivity_report.txt', 'w') as f:
            f.write(report)


def run_sensitivity_analysis(data: pd.DataFrame,
                             output_dir: str = "results",
                             verbose: bool = True) -> pd.DataFrame:
    """
    Run complete sensitivity analysis.
    
    Drop-in replacement for existing run_sensitivity_analysis in run_simulation.py
    
    Args:
        data: Simulation data (must have carbon_intensity and consumption_kwh columns)
        output_dir: Output directory for results
        verbose: Print progress
        
    Returns:
        DataFrame with main effects results (compatible with visualization)
    """
    if verbose:
        print(f"\n{'='*70}")
        print("SENSITIVITY ANALYSIS")
        print(f"{'='*70}")
    
    analysis = SensitivityAnalysis()
    
    # Run full grid (27 combinations)
    if verbose:
        print("\n  Running full grid analysis (27 combinations)...")
    analysis.run_full_grid(data)
    
    # Calculate main effects (9 scenarios)
    if verbose:
        print("  Calculating main effects...")
    analysis.calculate_main_effects(data)
    
    # Save results
    if verbose:
        print("  Saving results...")
    analysis.save_results(output_dir)
    
    # Print summary
    if verbose:
        fg = analysis.full_grid_results
        print(f"\n  Results:")
        print(f"    Delta_rho range: {fg['delta_rho'].min():.3f} to {fg['delta_rho'].max():.3f}")
        print(f"    Emission reduction: {fg['emission_reduction_pct'].min():.1f}% to {fg['emission_reduction_pct'].max():.1f}%")
        print(f"    Threshold pass: {fg['threshold_met'].sum()}/27 ({fg['threshold_met'].mean()*100:.0f}%)")
        print(f"\n  Saved:")
        print(f"    - {output_dir}/sensitivity_full_grid.csv")
        print(f"    - {output_dir}/sensitivity_main_effects.csv")
        print(f"    - {output_dir}/sensitivity_report.txt")
        print(f"\n{'='*70}")
        print("SENSITIVITY ANALYSIS COMPLETE")
        print(f"{'='*70}")
    
    # Return main effects for visualization compatibility
    return analysis.main_effects_results
