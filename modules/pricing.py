import pandas as pd
import numpy as np
from .parameters import ModelParameters


class PricingModels:
    """
    Three pricing schemes for comparison.
    
    Formulas justified in Chapter 3 (Methodology) of thesis.
    """
    
    def __init__(self, params: ModelParameters):
        """Initialize with model parameters."""
        self.params = params
    
    def calculate_tou_price(self, hour: int) -> float:
        """
        Calculate Time-of-Use price.
        
        Formula: P_TOU(t) = P_base × M(t)
        
        Returns:
            Price in CHF/kWh
        """
        p = self.params
        
        if hour in [7, 8, 18, 19, 20]:
            multiplier = p.tou_peak_multiplier
        elif hour in [23, 0, 1, 2, 3, 4, 5, 6]:
            multiplier = p.tou_offpeak_multiplier
        else:
            multiplier = p.tou_standard_multiplier
        
        return p.base_price * multiplier
    
    def calculate_carbon_aware_price(self, 
                                     carbon_intensity: float,
                                     carbon_reference: float) -> float:
        """
        Calculate Carbon-Aware Dynamic Price.
        
        Formula: P_carbon(t) = P_base × [1 + α × Premium_CI(t)]
        where Premium_CI(t) = (CI(t) - CI_ref) / CI_ref
        
        Returns:
            Price in CHF/kWh with safety bounds
        """
        p = self.params
        
        if carbon_reference == 0:
            carbon_premium = 0
        else:
            carbon_premium = (carbon_intensity - carbon_reference) / carbon_reference
        
        price_multiplier = 1 + p.carbon_weight_alpha * carbon_premium
        price = p.base_price * price_multiplier
        
        # Safety bounds
        min_price = p.base_price * p.price_floor_ratio
        max_price = p.base_price * p.price_ceiling_ratio
        
        return np.clip(price, min_price, max_price)
    
    def calculate_cpp_price(self,
                           hour: int,
                           carbon_intensity: float,
                           ci_threshold: float) -> float:
        """
        Calculate Critical Peak Price.
        
        Formula: P_CPP(t) = P_TOU(t) × C(t)
        where C(t) = 3.0 if CI(t) > CI_90, else 1.0
        
        Returns:
            Price in CHF/kWh
        """
        base_price = self.calculate_tou_price(hour)
        
        if carbon_intensity > ci_threshold:
            return base_price * self.params.cpp_multiplier
        else:
            return base_price
    
    def apply_all_pricing_schemes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply all three pricing schemes to dataset."""
        print("\n" + "="*70)
        print("APPLYING PRICING SCHEMES")
        print("="*70)
        
        result = data.copy()
        
        # Calculate reference values
        carbon_reference = result['carbon_intensity'].mean()
        ci_90_threshold = result['carbon_intensity'].quantile(0.90)
        
        self.params.carbon_reference = carbon_reference
        
        print(f"\n  Carbon reference: {carbon_reference:.1f} gCO₂/kWh")
        print(f"  CPP threshold (90th pctl): {ci_90_threshold:.1f} gCO₂/kWh")
        
        # Apply pricing schemes
        result['price_tou'] = result['hour'].apply(self.calculate_tou_price)
        
        result['price_carbon'] = result['carbon_intensity'].apply(
            lambda ci: self.calculate_carbon_aware_price(ci, carbon_reference)
        )
        
        result['price_cpp'] = result.apply(
            lambda row: self.calculate_cpp_price(
                row['hour'], 
                row['carbon_intensity'], 
                ci_90_threshold
            ),
            axis=1
        )
        
        return result
