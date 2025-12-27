from dataclasses import dataclass


@dataclass
class ModelParameters:
    """
    Configuration parameters for the simulation model.

    All parameters are justified in Chapter 3 (Methodology) of the thesis.
    Sources are documented for reproducibility and transparency.
    """
    
    # ---------------------------------------------------------------------------
    # Base Price Parameters
    # Source: ElCom Swiss Electricity Price Statistics 2023
    # ---------------------------------------------------------------------------
    base_price: float = 0.20  # CHF/kWh, Swiss average retail price
    
    # ---------------------------------------------------------------------------
    # Time-of-Use (TOU) Multipliers
    # Source: Standard Swiss cantonal utility tariff structure (EKZ, BKW, Axpo)
    # ---------------------------------------------------------------------------
    tou_peak_multiplier: float = 1.30      # Peak hours: 07-09, 18-21
    tou_standard_multiplier: float = 1.00  # Standard hours: 09-18, 21-23
    tou_offpeak_multiplier: float = 0.70   # Off-peak hours: 23-07
    
    # ---------------------------------------------------------------------------
    # Carbon-Aware Pricing Parameters
    # Source: Nilsson et al. (2017), Holland & Mansur (2008)
    # ---------------------------------------------------------------------------
    carbon_weight_alpha: float = 0.40   # Carbon sensitivity coefficient
    carbon_reference: float = None      # Will be calculated from data (mean CI)
    price_floor_ratio: float = 0.50     # Minimum price = 50% of base
    price_ceiling_ratio: float = 2.00   # Maximum price = 200% of base
    
    # ---------------------------------------------------------------------------
    # Critical Peak Pricing (CPP) Parameters
    # Source: Faruqui & Sergici (2010, 2011)
    # ---------------------------------------------------------------------------
    cpp_multiplier: float = 3.0              # Critical event price multiplier
    cpp_threshold_percentile: float = 90.0   # Top 10% CI triggers critical event
    
    # ---------------------------------------------------------------------------
    # Behavioral Response Parameters
    # Sources:
    #   - Filippini (2011): Swiss residential price elasticity
    #   - Faruqui & Sergici (2010): Technology amplification
    #   - Kahneman & Tversky (1979), Spurlock (2020): Loss aversion
    #   - Chen et al. (2023): Thermal comfort constraints
    # ---------------------------------------------------------------------------
    base_elasticity: float = -0.35           # Swiss residential price elasticity
    smart_meter_penetration: float = 0.80    # 80% of Swiss households
    smart_meter_amplification: float = 0.40  # +40% response with smart meter
    thermostat_penetration: float = 0.15     # 15% with smart thermostat
    thermostat_amplification: float = 1.00   # +100% response with thermostat
    loss_aversion_factor: float = 2.5        # Response multiplier for price increases
    thermal_comfort_floor: float = 0.60      # Minimum demand ratio (heating/cooling)
    
    # ---------------------------------------------------------------------------
    # Hypothesis Test Thresholds
    # Source: Hao et al. (2024) meta-analysis range: -0.10 to -0.25
    # ---------------------------------------------------------------------------
    h1_threshold: float = -0.15  # Required Δρ for H1 support
    
    def calculate_tech_amplification(self) -> float:
        """
        Calculate combined technology amplification factor.
        
        Formula:
            τ = (penetration_meters × amplification_meters) + 
                (penetration_thermostat × amplification_thermostat)
        
        Returns:
            Combined amplification factor (τ)
        """
        return (self.smart_meter_penetration * self.smart_meter_amplification +
                self.thermostat_penetration * self.thermostat_amplification)