import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple


# Default rebound rates based on literature
DEFAULT_REBOUND_RATES = [0.00, 0.10, 0.20, 0.30]
DEFAULT_LABELS = ["No rebound", "Conservative (10%)", "Central (20%)", "High (30%)"]

"""
Models how lower prices during clean hours may induce additional consumption,
partially offsetting emission savings.

Literature basis:
    - Gillingham et al. (2016): 5-40% direct rebound (meta-analysis)
    - Madlener & Hauertmann (2011): 12-18% (German residential)
    - Swiss context: Conservative 0-30% range (high income, low electricity budget share)
"""

def calculate_rebound_demand(demand_carbon: pd.Series,
                              price_tou: pd.Series,
                              price_carbon: pd.Series,
                              rebound_rate: float) -> pd.Series:
    """
    Apply rebound effect to demand.
    
    Formula: D_rebound = D_carbon × [1 + r × max(0, (P_tou - P_carbon) / P_tou)]
    
    Rebound only applies when P_carbon < P_tou (clean hours with lower prices).
    """
    if rebound_rate == 0.0:
        return demand_carbon.copy()
    
    price_reduction = ((price_tou - price_carbon) / price_tou).clip(lower=0)
    return demand_carbon * (1 + rebound_rate * price_reduction)


def analyze_scenario(data: pd.DataFrame, rebound_rate: float) -> Dict:
    """Analyze a single rebound scenario."""
    demand_rebound = calculate_rebound_demand(
        data['demand_carbon'], data['price_tou'], data['price_carbon'], rebound_rate
    )
    
    # Correlation metrics
    rho_tou = data['demand_tou'].corr(data['carbon_intensity'])
    rho_rebound = demand_rebound.corr(data['carbon_intensity'])
    delta_rho = rho_rebound - rho_tou
    
    # Emission metrics
    e_tou = (data['demand_tou'] * data['carbon_intensity']).sum() / 1e9
    e_baseline = (data['demand_carbon'] * data['carbon_intensity']).sum() / 1e9
    e_rebound = (demand_rebound * data['carbon_intensity']).sum() / 1e9
    
    reduction_baseline = (e_tou - e_baseline) / e_tou * 100
    reduction_rebound = (e_tou - e_rebound) / e_tou * 100
    
    return {
        'rebound_rate': rebound_rate,
        'rebound_rate_pct': rebound_rate * 100,
        'delta_rho_rebound': delta_rho,
        'reduction_baseline_pct': reduction_baseline,
        'reduction_rebound_pct': reduction_rebound,
        'rebound_offset_pct': (reduction_baseline - reduction_rebound) / reduction_baseline * 100 if reduction_baseline > 0 else 0,
        'h1_supported': delta_rho <= -0.15
    }


def run_rebound_analysis(data: pd.DataFrame,
                          output_dir: str = "results",
                          rebound_rates: List[float] = None,
                          verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Run complete rebound effect analysis.

    Args:
        data: Simulation results with demand_carbon, price_tou, price_carbon columns
        output_dir: Directory for output CSV
        rebound_rates: List of rates to test (default: [0, 0.1, 0.2, 0.3])
        verbose: Print progress
    """
    rates = rebound_rates or DEFAULT_REBOUND_RATES
    
    if verbose:
        print(f"\n{'='*70}")
        print("REBOUND EFFECT ANALYSIS")
        print("="*70)
        print("Testing robustness to rebound rates: " + ", ".join(f"{r*100:.0f}%" for r in rates))
    
    results = [analyze_scenario(data, r) for r in rates]
    results_df = pd.DataFrame(results)
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path / "rebound_analysis.csv", index=False)
    
    if verbose:
        print(f"\n{'Rebound':<10} {'Δρ':>10} {'Reduction':>12} {'Offset':>10} {'H1':>6}")
        print("-" * 50)
        for _, row in results_df.iterrows():
            h1 = "✓" if row['h1_supported'] else "✗"
            print(f"{row['rebound_rate_pct']:>6.0f}%    {row['delta_rho_rebound']:>+10.3f} "
                  f"{row['reduction_rebound_pct']:>11.1f}% {row['rebound_offset_pct']:>9.1f}% {h1:>6}")
        print("-" * 50)
        print(f"✓ Saved: {output_path / 'rebound_analysis.csv'}")
    
    summary = {
        'all_h1_supported': results_df['h1_supported'].all(),
        'baseline_reduction': results_df.iloc[0]['reduction_baseline_pct'],
        'worst_case_reduction': results_df['reduction_rebound_pct'].min(),
        'cpp_comparison': results_df['reduction_rebound_pct'].min() > 11.0  # CPP baseline
    }
    
    return results_df, summary
