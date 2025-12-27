"""
Hypothesis Testing Module
==========================
Tests research hypotheses using simulation results.

This module implements threshold-based evaluation for two research questions.
Since the analysis uses complete hourly data (not a sample) and deterministic
behavioral models, we use benchmark comparisons rather than p-value based tests.

Research Question 1: Temporal Alignment Improvement
  H0: Carbon-aware pricing does not improve alignment (Δρ ≥ -0.15)
  H1: Carbon-aware pricing improves alignment (Δρ < -0.15)

Research Question 2: Seasonal Effect Heterogeneity
  H0: Effect magnitude is uniform across time periods
  H1: Effect is stronger in winter evening peaks
"""

import pandas as pd
from typing import Dict
from .parameters import ModelParameters


class HypothesisTesting:
    """
    Hypothesis testing framework using threshold-based evaluation.
    
    Primary metric: Pearson correlation (ρ) between demand and carbon intensity
    - ρ > 0: Consumers use more electricity when grid is dirtier (problem state)
    - ρ < 0: Consumers use more electricity when grid is cleaner (goal state)
    
    Improvement metric: Δρ = ρ_intervention - ρ_baseline
    - Negative Δρ indicates improved temporal alignment
    """
    
    def __init__(self, params: ModelParameters):
        """Initialize with model parameters."""
        self.params = params
        self.results = {}
    
    def calculate_temporal_alignment(self, 
                                     demand: pd.Series, 
                                     carbon: pd.Series) -> float:
        """
        Calculate temporal alignment using Pearson correlation.
        
        Formula:
            ρ(D, CI) = Cov(D, CI) / (σ_D × σ_CI)
        
        Args:
            demand: Electricity demand time series (kWh)
            carbon: Carbon intensity time series (gCO₂/kWh)
            
        Returns:
            Pearson correlation coefficient
        """
        return demand.corr(carbon)
    
    def test_rq1_temporal_alignment(self, data: pd.DataFrame) -> Dict:
        """
        Test Research Question 1: Temporal Alignment Improvement
        
        H0: Carbon-aware pricing does not meaningfully improve temporal 
            alignment compared to TOU (Δρ ≥ -0.15)
        
        H1: Carbon-aware pricing meaningfully improves temporal alignment
            compared to TOU (Δρ < -0.15)
        
        Decision rule:
            - If Δρ < -0.15: Reject H0, support H1
            - If Δρ ≥ -0.15: Fail to reject H0
        
        The threshold -0.15 represents the midpoint of correlation improvements
        observed in effective dynamic pricing interventions (Hao et al., 2024).
        
        Args:
            data: DataFrame with demand and carbon columns for all scenarios
            
        Returns:
            Dictionary with test results
        """
        print("\n" + "="*70)
        print("RESEARCH QUESTION 1: TEMPORAL ALIGNMENT IMPROVEMENT")
        print("="*70)
        print("\nH0: Δρ ≥ -0.15 (no meaningful improvement)")
        print(f"H1: Δρ < -0.15 (meaningful improvement)")
        
        # Calculate correlations for each scenario
        rho_tou = self.calculate_temporal_alignment(
            data['demand_tou'], 
            data['carbon_intensity']
        )
        
        rho_carbon = self.calculate_temporal_alignment(
            data['demand_carbon'],
            data['carbon_intensity']
        )
        
        rho_cpp = self.calculate_temporal_alignment(
            data['demand_cpp'],
            data['carbon_intensity']
        )
        
        # Calculate improvements
        delta_rho_carbon = rho_carbon - rho_tou
        delta_rho_cpp = rho_cpp - rho_tou
        
        # Decision: Reject H0 if Δρ < -0.15
        reject_h0 = delta_rho_carbon < self.params.h1_threshold
        
        print(f"\n  Observed Correlations:")
        print(f"    ρ(Demand, Carbon) under TOU:          {rho_tou:+.3f}")
        print(f"    ρ(Demand, Carbon) under Carbon-Aware: {rho_carbon:+.3f}")
        print(f"    ρ(Demand, Carbon) under CPP:          {rho_cpp:+.3f}")
        
        print(f"\n  Correlation Changes:")
        print(f"    Δρ (Carbon-Aware vs TOU): {delta_rho_carbon:+.3f}")
        print(f"    Δρ (CPP vs TOU):          {delta_rho_cpp:+.3f}")
        
        print(f"\n  Decision:")
        print(f"    Threshold: Δρ < {self.params.h1_threshold}")
        print(f"    Observed: Δρ = {delta_rho_carbon:+.3f}")
        
        if reject_h0:
            print(f"    → REJECT H0, SUPPORT H1")
            print(f"      Carbon-aware pricing achieves meaningful improvement")
            improvement_factor = abs(delta_rho_carbon / self.params.h1_threshold)
            print(f"      Effect is {improvement_factor:.1f}× larger than threshold")
        else:
            print(f"    → FAIL TO REJECT H0")
            print(f"      Improvement does not exceed threshold")
        
        self.results['rq1'] = {
            'h0_statement': f'Δρ ≥ {self.params.h1_threshold}',
            'h1_statement': f'Δρ < {self.params.h1_threshold}',
            'rho_tou': rho_tou,
            'rho_carbon': rho_carbon,
            'rho_cpp': rho_cpp,
            'delta_rho_carbon': delta_rho_carbon,
            'delta_rho_cpp': delta_rho_cpp,
            'threshold': self.params.h1_threshold,
            'reject_h0': reject_h0,
            'h1_supported': reject_h0
        }

        return self.results['rq1']
    
    def test_rq2_seasonal_heterogeneity(self, data: pd.DataFrame) -> Dict:
        """
        Test Research Question 2: Seasonal Effect Heterogeneity

        H0: The temporal alignment improvement is uniform across time periods

        H1: The temporal alignment improvement is stronger during winter 
            evening peak hours (January 18:00-21:00)

        Decision rule:
            - If |Δρ_winter| > |Δρ_overall|: Reject H0, support H1
            - Otherwise: Fail to reject H0

        Rationale: Winter evenings represent the highest carbon intensity
        period and the greatest opportunity for emission reduction through
        demand shifting. If the intervention is particularly effective during
        these critical hours, it suggests targeted impact where most needed.
        """
        print("\n" + "="*70)
        print("RESEARCH QUESTION 2: SEASONAL EFFECT HETEROGENEITY")
        print("="*70)
        print("\nH0: Effect magnitude is uniform across time periods")
        print("H1: Effect is stronger in winter evening peaks")
        
        # Filter to winter evening (January 18:00-21:00)
        winter_evening = data[
            (data['month'] == 1) & 
            (data['hour'].isin([18, 19, 20]))
        ]
        
        if len(winter_evening) == 0:
            print("  [WARN] No winter evening data available for this test")
            return {}
        
        # Calculate correlations for winter evening subset
        rho_tou_we = self.calculate_temporal_alignment(
            winter_evening['demand_tou'],
            winter_evening['carbon_intensity']
        )
        
        rho_carbon_we = self.calculate_temporal_alignment(
            winter_evening['demand_carbon'],
            winter_evening['carbon_intensity']
        )
        
        delta_rho_we = rho_carbon_we - rho_tou_we
        
        # Compare to overall effect
        overall_delta = self.results.get('rq1', {}).get('delta_rho_carbon', 0)
        
        # Decision: Reject H0 if winter effect is stronger
        reject_h0 = abs(delta_rho_we) > abs(overall_delta)
        amplification = abs(delta_rho_we) / abs(overall_delta) if overall_delta != 0 else 0
        
        print(f"\n  Winter Evening Subset (n={len(winter_evening)}):")
        print(f"    ρ_TOU:          {rho_tou_we:+.3f}")
        print(f"    ρ_Carbon-Aware: {rho_carbon_we:+.3f}")
        print(f"    Δρ_winter:      {delta_rho_we:+.3f}")
        
        print(f"\n  Comparison:")
        print(f"    Δρ_overall:     {overall_delta:+.3f}")
        print(f"    Δρ_winter:      {delta_rho_we:+.3f}")
        
        print(f"\n  Decision:")
        if reject_h0:
            print(f"    → REJECT H0, SUPPORT H1")
            print(f"      Effect is {amplification:.1%} stronger in winter evenings")
            print(f"      Intervention targets high-carbon periods effectively")
        else:
            print(f"    → FAIL TO REJECT H0")
            print(f"      Effect magnitude is similar across time periods")
        
        self.results['rq2'] = {
            'h0_statement': 'Effect is uniform across time periods',
            'h1_statement': 'Effect is stronger in winter evenings',
            'n_observations': len(winter_evening),
            'rho_tou': rho_tou_we,
            'rho_carbon': rho_carbon_we,
            'delta_rho_winter': delta_rho_we,
            'delta_rho_overall': overall_delta,
            'reject_h0': reject_h0,
            'amplification_factor': amplification,
            'stronger_in_winter': reject_h0  # Alias for backward compatibility
        }
        
        return self.results['rq2']
    
    def calculate_emission_impacts(self, data: pd.DataFrame) -> Dict:
        """
        Calculate total emissions under each pricing scenario.
        
        This is a descriptive analysis, not a hypothesis test.
        It quantifies the practical impact of the interventions.
        
        Formula:
            E = Σ[D(t) × CI(t)] for all hours t
        
        Where:
            E = Total emissions (kgCO₂)
            D(t) = Demand at hour t (kWh)
            CI(t) = Carbon intensity at hour t (gCO₂/kWh)
        """
        print("\n" + "="*70)
        print("EMISSION IMPACT ANALYSIS")
        print("="*70)
        
        # Calculate total emissions for each scenario (in kgCO2, then convert to ktCO2)
        e_tou = (data['demand_tou'] * data['carbon_intensity']).sum() / 1e9
        e_carbon = (data['demand_carbon'] * data['carbon_intensity']).sum() / 1e9
        e_cpp = (data['demand_cpp'] * data['carbon_intensity']).sum() / 1e9
        
        # Calculate reductions relative to TOU baseline
        reduction_carbon = (e_tou - e_carbon) / e_tou * 100
        reduction_cpp = (e_tou - e_cpp) / e_tou * 100
        
        # Absolute reduction
        absolute_reduction_carbon = e_tou - e_carbon
        absolute_reduction_cpp = e_tou - e_cpp
        
        print(f"\n  Total Annual Emissions:")
        print(f"    TOU (Baseline):   {e_tou:,.2f} ktCO₂")
        print(f"    Carbon-Aware:     {e_carbon:,.2f} ktCO₂")
        print(f"    CPP:              {e_cpp:,.2f} ktCO₂")
        
        print(f"\n  Emission Reductions:")
        print(f"    Carbon-Aware vs TOU: {absolute_reduction_carbon:,.2f} ktCO₂ ({reduction_carbon:+.1f}%)")
        print(f"    CPP vs TOU:          {absolute_reduction_cpp:,.2f} ktCO₂ ({reduction_cpp:+.1f}%)")
        
        # Context: Literature range
        print(f"\n  Context:")
        print(f"    Literature range for dynamic pricing: 5-25% emission reduction")
        if 5 <= reduction_carbon <= 25:
            print(f"    → Carbon-aware result ({reduction_carbon:.1f}%) falls within established range")
        elif reduction_carbon > 25:
            print(f"    → Carbon-aware result ({reduction_carbon:.1f}%) exceeds typical range")
        
        self.results['emissions'] = {
            'e_tou_ktco2': e_tou,
            'e_carbon_ktco2': e_carbon,
            'e_cpp_ktco2': e_cpp,
            'absolute_reduction_carbon_ktco2': absolute_reduction_carbon,
            'absolute_reduction_cpp_ktco2': absolute_reduction_cpp,
            'reduction_carbon_pct': reduction_carbon,
            'reduction_cpp_pct': reduction_cpp
        }
        
        return self.results['emissions']
    
    def run_all_tests(self, data: pd.DataFrame) -> Dict:
        """
        Execute all hypothesis tests and impact analyses.
        
        Order:
        1. RQ1: Temporal alignment (must run first, provides baseline)
        2. RQ2: Seasonal heterogeneity (uses RQ1 results)
        3. Emission impacts (descriptive, not a test)
        """
        self.test_rq1_temporal_alignment(data)
        self.test_rq2_seasonal_heterogeneity(data)
        self.calculate_emission_impacts(data)
        
        return self.results
