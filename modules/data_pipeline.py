import pandas as pd
import numpy as np
from pathlib import Path
from typing import List


class DataPipeline:
    """
    Data loading and cleaning pipeline for Swiss electricity analysis.
    
    Responsibilities:
    - Load carbon intensity data (Electricity Maps CSV)
    - Load consumption data (SFOE Excel)
    - Clean and validate both datasets
    - Merge on hourly timestamps
    - Generate cleaning report
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data pipeline.
        
        Args:
            data_dir: Path to the data directory
        """
        self.data_dir = Path(data_dir)
        self.carbon_raw = None
        self.carbon_clean = None
        self.consumption_raw = None
        self.consumption_clean = None
        self.merged_data = None
        self.cleaning_report = {}
    
    def load_carbon_intensity_data(self, 
                                   years: List[int] = [2021, 2022, 2023, 2024]
                                   ) -> pd.DataFrame:
        """
        Load hourly carbon intensity data from Electricity Maps CSV files.
        
        Data source: Electricity Maps (https://www.electricitymaps.com/)
        Files: CH_YYYY_hourly.csv
        
        Args:
            years: List of years to load
            
        Returns:
            DataFrame with raw carbon intensity data
        """
        print("\n" + "="*70)
        print("STEP 1: LOADING CARBON INTENSITY DATA")
        print("="*70)
        
        dataframes = []
        
        for year in years:
            filepath = self.data_dir / f"CH_{year}_hourly.csv"
            
            if filepath.exists():
                df = pd.read_csv(filepath)
                df['source_year'] = year
                dataframes.append(df)
                print(f"  [OK] Loaded {year}: {len(df):,} hourly records")
            else:
                print(f"  [WARN] File not found: {filepath}")
        
        if not dataframes:
            raise FileNotFoundError("No carbon intensity data files found")
        
        self.carbon_raw = pd.concat(dataframes, ignore_index=True)
        print(f"\n  Total raw records: {len(self.carbon_raw):,}")
        
        return self.carbon_raw
    
    def load_consumption_data(self, 
                              years: List[int] = [2021, 2022, 2023, 2024]
                              ) -> pd.DataFrame:
        """
        Load 15-minute consumption data from SFOE Excel files.
        
        Data source: Swiss Federal Office of Energy (SFOE) via Swissgrid
        Files: Energy_Statistic_CH_YYYY.xlsx
        Sheet: Zeitreihen0h15 (15-minute time series)
        
        Args:
            years: List of years to load
            
        Returns:
            DataFrame with raw consumption data
        """
        print("\n" + "="*70)
        print("STEP 2: LOADING CONSUMPTION DATA")
        print("="*70)
        
        dataframes = []
        
        for year in years:
            filepath = self.data_dir / f"Energy_Statistic_CH_{year}.xlsx"
            
            if filepath.exists():
                try:
                    df = pd.read_excel(filepath, 
                                      sheet_name='Zeitreihen0h15', 
                                      skiprows=1)
                    df['source_year'] = year
                    dataframes.append(df)
                    print(f"  [OK] Loaded {year}: {len(df):,} quarter-hourly records")
                except Exception as e:
                    print(f"  [ERROR] Failed to load {year}: {str(e)}")
            else:
                print(f"  [SKIP] File not found: {filepath}")
        
        if not dataframes:
            raise FileNotFoundError("No consumption data files found")
        
        self.consumption_raw = pd.concat(dataframes, ignore_index=True)
        print(f"\n  Total raw records: {len(self.consumption_raw):,}")
        
        return self.consumption_raw
    
    def clean_carbon_data(self) -> pd.DataFrame:
        """
        Clean and standardize carbon intensity data.
        
        Cleaning steps:
        1. Standardize column names
        2. Parse datetime
        3. Handle missing values
        4. Remove duplicates
        5. Add temporal features
        
        Returns:
            Cleaned carbon intensity DataFrame
        """
        print("\n" + "="*70)
        print("STEP 3: CLEANING CARBON INTENSITY DATA")
        print("="*70)
        
        if self.carbon_raw is None:
            raise ValueError("Raw carbon data not loaded")
        
        df = self.carbon_raw.copy()
        initial_rows = len(df)
        
        # Standardize column names
        column_mapping = {
            'Datetime (UTC)': 'datetime',
            'Carbon intensity gCO₂eq/kWh (direct)': 'carbon_intensity',
            'Carbon intensity gCO₂eq/kWh (Life cycle)': 'carbon_intensity_lifecycle',
            'Carbon-free energy percentage (CFE%)': 'cfe_percent',
            'Renewable energy percentage (RE%)': 'renewable_percent'
        }
        df = df.rename(columns=column_mapping)
        print(f"  [3.1] Standardized column names")
        
        # Parse datetime
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df['datetime'] = df['datetime'].dt.tz_localize(None)
        print(f"  [3.2] Parsed datetime column")
        
        # Handle missing values
        missing_ci = df['carbon_intensity'].isna().sum()
        if missing_ci > 0:
            df['carbon_intensity'] = df['carbon_intensity'].interpolate(method='linear')
            print(f"  [3.3] Interpolated {missing_ci} missing values")
        else:
            print(f"  [3.3] No missing carbon intensity values")
        
        # Remove duplicates
        duplicates = df.duplicated(subset=['datetime']).sum()
        df = df.drop_duplicates(subset=['datetime'], keep='first')
        print(f"  [3.4] Removed {duplicates} duplicate timestamps")
        
        # Add temporal features
        df['hour'] = df['datetime'].dt.hour
        df['month'] = df['datetime'].dt.month
        df['year'] = df['datetime'].dt.year
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        df['date_hour'] = df['datetime'].dt.floor('h')
        
        # Add season
        season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                      3: 'Spring', 4: 'Spring', 5: 'Spring',
                      6: 'Summer', 7: 'Summer', 8: 'Summer',
                      9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}
        df['season'] = df['month'].map(season_map)
        print(f"  [3.5] Added temporal features")
        
        final_rows = len(df)
        self.cleaning_report['carbon_initial'] = initial_rows
        self.cleaning_report['carbon_final'] = final_rows
        self.cleaning_report['carbon_removed'] = initial_rows - final_rows
        
        print(f"\n  Cleaning summary: {initial_rows:,} → {final_rows:,} rows")
        
        self.carbon_clean = df
        return df
    
    def clean_consumption_data(self) -> pd.DataFrame:
        """
        Clean consumption data and aggregate from 15-minute to hourly.
        
        Returns:
            Cleaned and aggregated consumption DataFrame
        """
        print("\n" + "="*70)
        print("STEP 4: CLEANING CONSUMPTION DATA")
        print("="*70)
        
        if self.consumption_raw is None:
            raise ValueError("Raw consumption data not loaded")
        
        df = self.consumption_raw.copy()
        initial_rows = len(df)
        
        # Identify and rename columns
        df = df.rename(columns={
            df.columns[0]: 'datetime',
            df.columns[1]: 'consumption_kwh'
        })
        print(f"  [4.1] Identified datetime and consumption columns")
        
        # Parse datetime
        df['datetime'] = pd.to_datetime(df['datetime'], format='%d.%m.%Y %H:%M', errors='coerce')
        invalid_dates = df['datetime'].isna().sum()
        df = df.dropna(subset=['datetime'])
        print(f"  [4.2] Parsed datetime, removed {invalid_dates} invalid timestamps")
        
        # Convert consumption to numeric
        df['consumption_kwh'] = pd.to_numeric(df['consumption_kwh'], errors='coerce')
        missing_consumption = df['consumption_kwh'].isna().sum()
        df = df.dropna(subset=['consumption_kwh'])
        print(f"  [4.3] Removed {missing_consumption} rows with missing consumption")
        
        # Aggregate to hourly
        df['date_hour'] = df['datetime'].dt.floor('h')
        
        hourly = df.groupby('date_hour').agg({
            'consumption_kwh': 'sum'
        }).reset_index()
        
        print(f"  [4.4] Aggregated to hourly: {len(df):,} → {len(hourly):,} rows")
        
        # Add temporal features
        hourly['hour'] = hourly['date_hour'].dt.hour
        hourly['month'] = hourly['date_hour'].dt.month
        hourly['year'] = hourly['date_hour'].dt.year
        print(f"  [4.5] Added temporal features")
        
        final_rows = len(hourly)
        self.cleaning_report['consumption_initial'] = initial_rows
        self.cleaning_report['consumption_hourly'] = final_rows
        
        print(f"\n  Cleaning summary: {initial_rows:,} quarter-hourly → {final_rows:,} hourly rows")
        
        self.consumption_clean = hourly
        return hourly
    
    def merge_datasets(self, analysis_year: int = 2023) -> pd.DataFrame:
        """
        Merge carbon intensity and consumption data for analysis.
        
        Args:
            analysis_year: Year to use for the main analysis
            
        Returns:
            Merged DataFrame ready for simulation
        """
        print("\n" + "="*70)
        print("STEP 5: MERGING DATASETS")
        print("="*70)
        
        if self.carbon_clean is None or self.consumption_clean is None:
            raise ValueError("Clean data not available")
        
        # Filter to analysis year
        carbon_year = self.carbon_clean[self.carbon_clean['year'] == analysis_year].copy()
        consumption_year = self.consumption_clean[
            self.consumption_clean['year'] == analysis_year
        ].copy()
        
        print(f"  Filtering to {analysis_year}:")
        print(f"    Carbon intensity records: {len(carbon_year):,}")
        print(f"    Consumption records: {len(consumption_year):,}")
        
        # Perform inner join
        merged = pd.merge(
            carbon_year,
            consumption_year[['date_hour', 'consumption_kwh']],
            on='date_hour',
            how='inner'
        )
        
        merged = merged.sort_values('datetime').reset_index(drop=True)
        
        self.cleaning_report['merged_rows'] = len(merged)
        self.cleaning_report['analysis_year'] = analysis_year
        
        print(f"\n  Merged dataset: {len(merged):,} hourly observations")
        print(f"  Date range: {merged['datetime'].min()} to {merged['datetime'].max()}")
        
        self.merged_data = merged
        return merged
    
    def generate_cleaning_report(self) -> str:
        """Generate a summary report of the data cleaning process."""
        
        report = """
DATA CLEANING REPORT
====================

Carbon Intensity Data:
  Initial records: {carbon_initial:,}
  After cleaning: {carbon_final:,}
  Records removed: {carbon_removed:,}

Consumption Data:
  Initial records (15-min): {consumption_initial:,}
  After aggregation (hourly): {consumption_hourly:,}

Merged Dataset:
  Analysis year: {analysis_year}
  Final observations: {merged_rows:,}
""".format(**self.cleaning_report)
        
        return report
    
    def run_full_pipeline(self, 
                          carbon_years: List[int] = [2021, 2022, 2023, 2024],
                          consumption_years: List[int] = [2021, 2022, 2023],
                          analysis_year: int = 2023) -> pd.DataFrame:
        """
        Execute the complete data pipeline.
        
        Args:
            carbon_years: Years to load for carbon data
            consumption_years: Years to load for consumption
            analysis_year: Year for main analysis
            
        Returns:
            Cleaned and merged DataFrame
        """
        self.load_carbon_intensity_data(carbon_years)
        self.load_consumption_data(consumption_years)
        self.clean_carbon_data()
        self.clean_consumption_data()
        self.merge_datasets(analysis_year)
        
        print(self.generate_cleaning_report())
        
        return self.merged_data