import pandas as pd
from .parameters import ModelParameters


class BehavioralResponseModel:
    """
    Demand response model with behavioral factors.
    
    Components:
    - Base elasticity (Swiss-specific)
    - Technology amplification (smart meters, thermostats)
    - Loss aversion (asymmetric response)
    - Thermal comfort constraints (minimum demand floor)
    """
    
    def __init__(self, params: ModelParameters):
        """Initialize with model parameters."""
        self.params = params
        self.tech_amplification = params.calculate_tech_amplification()
    
    def calculate_effective_elasticity(self, is_price_increase: bool) -> float:
        """
        Calculate effective elasticity with behavioral factors.
        
        Formula: ε_eff = ε_base × (1 + τ) × λ
        
        Returns:
            Effective elasticity value
        """
        p = self.params
        
        elasticity = p.base_elasticity * (1 + self.tech_amplification)
        
        if is_price_increase:
            elasticity *= p.loss_aversion_factor
        
        return elasticity
    
    def calculate_demand_response(self,
                                  baseline_demand: float,
                                  baseline_price: float,
                                  new_price: float) -> float:
        """
        Calculate new demand given price change.
        
        Formula: D_new = D_baseline × (P_new / P_baseline)^ε_eff
        Subject to: D_new ≥ D_baseline × 0.60
        
        Returns:
            New demand level (kWh)
        """
        p = self.params
        
        if baseline_price <= 0 or baseline_demand <= 0:
            return baseline_demand
        
        price_ratio = new_price / baseline_price
        is_price_increase = new_price > baseline_price
        
        elasticity = self.calculate_effective_elasticity(is_price_increase)
        
        new_demand = baseline_demand * (price_ratio ** elasticity)
        
        # Thermal comfort floor
        min_demand = baseline_demand * p.thermal_comfort_floor
        
        return max(new_demand, min_demand)
    
    def apply_demand_response(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply behavioral demand response to all scenarios."""
        print("\n" + "="*70)
        print("APPLYING BEHAVIORAL RESPONSE MODEL")
        print("="*70)
        
        result = data.copy()
        
        result['demand_baseline'] = result['consumption_kwh']
        result['demand_tou'] = result['consumption_kwh']
        
        result['demand_carbon'] = result.apply(
            lambda row: self.calculate_demand_response(
                row['consumption_kwh'],
                row['price_tou'],
                row['price_carbon']
            ),
            axis=1
        )
        
        result['demand_cpp'] = result.apply(
            lambda row: self.calculate_demand_response(
                row['consumption_kwh'],
                row['price_tou'],
                row['price_cpp']
            ),
            axis=1
        )
        
        carbon_change = (result['demand_carbon'].sum() - result['demand_tou'].sum()) / result['demand_tou'].sum() * 100
        cpp_change = (result['demand_cpp'].sum() - result['demand_tou'].sum()) / result['demand_tou'].sum() * 100
        
        print(f"\n  Demand response summary:")
        print(f"    Carbon-aware vs TOU: {carbon_change:+.2f}%")
        print(f"    CPP vs TOU: {cpp_change:+.2f}%")
        
        return result