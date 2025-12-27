import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3
})


class ThesisFigures:
    """
    Publication-quality figure generation for thesis.
    
    All figures are saved as PDF for LaTeX inclusion.
    Color palette is consistent across all figures.
    """
    
    def __init__(self, output_dir: str = "results/figures"):
        """Initialize with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Consistent color palette
        self.colors = {
            'tou': '#1f77b4',       # Blue
            'carbon': '#2ca02c',    # Green
            'cpp': '#ff7f0e',       # Orange
            'baseline': '#7f7f7f',  # Gray
            'winter': '#d62728',    # Red
            'summer': '#9467bd',    # Purple
            'clean': '#2ca02c',     # Green
            'dirty': '#d62728'      # Red
        }
    
    def generate_all_figures(self, data: pd.DataFrame, 
                            sensitivity_data: Optional[pd.DataFrame] = None):
        """
        Generate all thesis figures.
        
        Args:
            data: Simulation results DataFrame
            sensitivity_data: Optional sensitivity analysis results
        """
        print("\n" + "="*70)
        print("GENERATING THESIS FIGURES")
        print("="*70)
        
        print("\nSection 4.1: Data Exploration")
        self._section_4_exploration(data)
        
        print("\nSection 4.2: Pricing Schemes")
        self._section_4_2_pricing(data)
        
        print("\nSection 4.3: Main Results")
        self._section_4_3_results(data)
        
        if sensitivity_data is not None:
            print("\nSection 4.4: Sensitivity Analysis")
            self._section_4_4_sensitivity(sensitivity_data)
        
        print(f"\nAll figures saved to: {self.output_dir}")
    
    # =========================================================================
    # SECTION 4.1: DATA EXPLORATION
    # =========================================================================
    
    def _section_4_exploration(self, data: pd.DataFrame):
        """Generate all data exploration figures."""
        self._fig_carbon_timeseries(data)
        self._fig_carbon_distribution(data)
        self._fig_hourly_pattern(data)
        self._fig_seasonal_pattern(data)
        self._fig_scenario_validation(data)
        self._fig_baseline_correlation(data)
    
    def _fig_carbon_timeseries(self, data: pd.DataFrame):
        """
        Figure 4.1: Carbon intensity time series.
        """
        fig, ax = plt.subplots(figsize=(14, 5))

        ax.plot(data['datetime'], data['carbon_intensity'], 
                linewidth=0.3, alpha=0.6, color='gray', label='Hourly')

        rolling = data.set_index('datetime')['carbon_intensity'].rolling('7D').mean()
        ax.plot(rolling.index, rolling.values, 
                linewidth=2, color=self.colors['carbon'], label='7-day average')

        ax.set_xlabel('Date')
        ax.set_ylabel('Carbon Intensity (gCO₂/kWh)')

        # Determine year range for title
        years = data['datetime'].dt.year.unique()
        if len(years) == 1:
            title_years = str(years[0])
        else:
            title_years = f"{years.min()}–{years.max()}"
        ax.set_title(f'Swiss Grid Carbon Intensity ({title_years})')

        ax.legend(loc='upper right')

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        ax.set_ylim(0, None)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "fig_4_1_carbon_timeseries.pdf")
        plt.close()
        print(f"  Saved: fig_4_1_carbon_timeseries.pdf")
    
    def _fig_carbon_distribution(self, data: pd.DataFrame):
        """
        Figure 4.2: Distribution of carbon intensity.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ci = data['carbon_intensity']
        
        n, bins, patches = ax.hist(ci, bins=50, edgecolor='white', 
                                   color=self.colors['carbon'], alpha=0.7)
        
        # Vertical reference lines
        ax.axvline(ci.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean = {ci.mean():.1f}')
        ax.axvline(ci.median(), color='blue', linestyle='--', linewidth=2,
                   label=f'Median = {ci.median():.1f}')
        ax.axvline(ci.quantile(0.90), color='orange', linestyle=':', linewidth=2,
                   label=f'90th pctl = {ci.quantile(0.90):.1f}')
        
        # FIX: Stats box positioned at upper right, outside legend area
        stats_text = f'CV = {ci.std()/ci.mean()*100:.1f}%\nn = {len(ci):,}'
        ax.text(0.97, 0.75, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('Carbon Intensity (gCO₂/kWh)')
        ax.set_ylabel('Frequency (hours)')
        ax.set_title('Distribution of Hourly Carbon Intensity')
        
        # FIX: Legend at upper left to avoid overlap with stats box
        ax.legend(loc='upper center', framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "fig_4_2_carbon_distribution.pdf")
        plt.close()
        print(f"  Saved: fig_4_2_carbon_distribution.pdf")
    
    def _fig_hourly_pattern(self, data: pd.DataFrame):
        """
        Figure 4.3: Hourly pattern of carbon intensity.
        
        FIX: Legend moved to lower left corner
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        
        hourly = data.groupby('hour')['carbon_intensity'].agg(['mean', 'std', 'count'])
        hourly['se'] = hourly['std'] / np.sqrt(hourly['count'])
        hourly['ci95'] = 1.96 * hourly['se']
        
        hours = hourly.index
        means = hourly['mean']
        ci95 = hourly['ci95']
        
        ax.fill_between(hours, means - ci95, means + ci95, 
                        alpha=0.3, color=self.colors['carbon'])
        ax.plot(hours, means, 'o-', color=self.colors['carbon'], 
                linewidth=2, markersize=6, label='Mean ± 95% CI')
        
        peak_hour = means.idxmax()
        valley_hour = means.idxmin()
        ax.annotate(f'Peak: {means[peak_hour]:.1f}', 
                    xy=(peak_hour, means[peak_hour]),
                    xytext=(peak_hour+1.5, means[peak_hour]+5),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
        ax.annotate(f'Valley: {means[valley_hour]:.1f}',
                    xy=(valley_hour, means[valley_hour]),
                    xytext=(valley_hour+2, means[valley_hour]+10),
                    arrowprops=dict(arrowstyle='->', color='blue'),
                    fontsize=10, color='blue')
        
        # TOU peak shading
        ax.axvspan(7, 9, alpha=0.15, color='red', label='TOU Peak')
        ax.axvspan(18, 21, alpha=0.15, color='red')
        
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Carbon Intensity (gCO₂/kWh)')
        ax.set_title('Daily Pattern of Grid Carbon Intensity')
        ax.set_xticks(range(0, 24, 2))
        ax.set_xlim(-0.5, 23.5)
        
        # FIX: Legend at lower left
        ax.legend(loc='lower left', framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "fig_4_3_hourly_pattern.pdf")
        plt.close()
        print(f"  Saved: fig_4_3_hourly_pattern.pdf")
    
    def _fig_seasonal_pattern(self, data: pd.DataFrame):
        """Figure 4.4: Seasonal variation."""
        fig, ax = plt.subplots(figsize=(8, 5))
        
        seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
        season_data = [data[data['season'] == s]['carbon_intensity'].values for s in seasons]
        
        bp = ax.boxplot(season_data, labels=seasons, patch_artist=True)
        
        colors = ['#a6cee3', '#b2df8a', '#fdbf6f', '#fb9a99']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        means = [np.mean(d) for d in season_data]
        ax.scatter(range(1, 5), means, marker='D', color='red', 
                   s=50, zorder=5, label='Mean')
        
        ax.set_ylabel('Carbon Intensity (gCO₂/kWh)')
        ax.set_title('Seasonal Variation in Carbon Intensity')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "fig_4_4_seasonal_pattern.pdf")
        plt.close()
        print(f"  Saved: fig_4_4_seasonal_pattern.pdf")
    
    def _fig_scenario_validation(self, data: pd.DataFrame):
        """Figure 4.5: Research scenario validation."""
        fig, ax = plt.subplots(figsize=(8, 5))
        
        winter_evening = data[(data['month'] == 1) & 
                             (data['hour'].isin([18, 19, 20]))]['carbon_intensity']
        summer_midday = data[(data['month'] == 7) & 
                            (data['hour'].isin([11, 12, 13, 14]))]['carbon_intensity']
        
        parts = ax.violinplot([winter_evening, summer_midday], positions=[1, 2],
                              showmeans=True, showmedians=True)
        
        parts['bodies'][0].set_facecolor(self.colors['winter'])
        parts['bodies'][0].set_alpha(0.7)
        parts['bodies'][1].set_facecolor(self.colors['summer'])
        parts['bodies'][1].set_alpha(0.7)
        
        ax.scatter(np.random.normal(1, 0.05, len(winter_evening)), winter_evening,
                   alpha=0.3, s=10, color=self.colors['winter'])
        ax.scatter(np.random.normal(2, 0.05, len(summer_midday)), summer_midday,
                   alpha=0.3, s=10, color=self.colors['summer'])
        
        ratio = winter_evening.mean() / summer_midday.mean()
        ax.annotate(f'{ratio:.1f}× higher', xy=(1.5, winter_evening.mean() * 0.9), 
                    fontsize=14, ha='center', weight='bold')
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Winter Evening\n(Jan 18-21h)', 
                           'Summer Midday\n(Jul 11-14h)'])
        ax.set_ylabel('Carbon Intensity (gCO₂/kWh)')
        ax.set_title('Research Scenario Validation')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "fig_4_5_scenario_validation.pdf")
        plt.close()
        print(f"  Saved: fig_4_5_scenario_validation.pdf")
    
    def _fig_baseline_correlation(self, data: pd.DataFrame):
        """Figure 4.6: Baseline correlation problem."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sample = data.sample(min(2000, len(data)), random_state=42)
        
        scatter = ax.scatter(sample['carbon_intensity'], 
                            sample['consumption_kwh'] / 1e6,
                            c=sample['hour'], cmap='twilight',
                            alpha=0.5, s=20)
        
        z = np.polyfit(data['carbon_intensity'], data['consumption_kwh'] / 1e6, 1)
        p = np.poly1d(z)
        x_line = np.linspace(data['carbon_intensity'].min(), 
                            data['carbon_intensity'].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, 
                label=f'ρ = {data["carbon_intensity"].corr(data["consumption_kwh"]):.3f}')
        
        cbar = plt.colorbar(scatter, ax=ax, label='Hour of Day')
        
        ax.set_xlabel('Carbon Intensity (gCO₂/kWh)')
        ax.set_ylabel('Electricity Demand (GWh)')
        ax.set_title('Baseline: Demand vs Carbon Intensity')
        ax.legend(loc='upper right', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "fig_4_6_baseline_correlation.pdf")
        plt.close()
        print(f"  Saved: fig_4_6_baseline_correlation.pdf")
    
    # =========================================================================
    # SECTION 4.2: PRICING SCHEMES
    # =========================================================================
    
    def _section_4_2_pricing(self, data: pd.DataFrame):
        """Generate pricing scheme figures."""
        self._fig_price_comparison_day(data)
        self._fig_price_vs_carbon(data)
    
    def _fig_price_comparison_day(self, data: pd.DataFrame):
        """Figure 4.7: Price comparison for sample day."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        jan_data = data[data['month'] == 1].copy()
        daily_ci = jan_data.groupby(jan_data['datetime'].dt.date)['carbon_intensity'].mean()
        high_carbon_day = daily_ci.idxmax()
        
        day_data = data[data['datetime'].dt.date == high_carbon_day].copy()
        hours = day_data['hour'].values
        
        ax1.fill_between(hours, 0, day_data['carbon_intensity'], 
                         alpha=0.3, color=self.colors['dirty'])
        ax1.plot(hours, day_data['carbon_intensity'], 'o-',
                 color=self.colors['dirty'], linewidth=2, label='Carbon Intensity')
        ax1.set_ylabel('Carbon Intensity (gCO₂/kWh)')
        ax1.set_title(f'Price Comparison: High-Carbon Day ({high_carbon_day})')
        ax1.legend(loc='upper right')
        
        ax2.plot(hours, day_data['price_tou'], 's-', 
                 color=self.colors['tou'], linewidth=2, markersize=6, label='TOU')
        ax2.plot(hours, day_data['price_carbon'], 'o-',
                 color=self.colors['carbon'], linewidth=2, markersize=6, label='Carbon-Aware')
        ax2.plot(hours, day_data['price_cpp'], '^-',
                 color=self.colors['cpp'], linewidth=2, markersize=6, label='CPP')
        
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Price (CHF/kWh)')
        ax2.set_xticks(range(0, 24, 2))
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "fig_4_7_price_comparison_day.pdf")
        plt.close()
        print(f"  Saved: fig_4_7_price_comparison_day.pdf")
    
    def _fig_price_vs_carbon(self, data: pd.DataFrame):
        """Figure 4.8: Price-carbon relationship."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sorted_data = data.sort_values('carbon_intensity')
        
        ax.scatter(sorted_data['carbon_intensity'], sorted_data['price_tou'],
                   alpha=0.1, s=5, color=self.colors['tou'])
        ax.scatter(sorted_data['carbon_intensity'], sorted_data['price_carbon'],
                   alpha=0.3, s=5, color=self.colors['carbon'])
        ax.scatter(sorted_data['carbon_intensity'], sorted_data['price_cpp'],
                   alpha=0.1, s=5, color=self.colors['cpp'])
        
        z = np.polyfit(data['carbon_intensity'], data['price_carbon'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, data['carbon_intensity'].max(), 100)
        ax.plot(x_line, p(x_line), '--', color=self.colors['carbon'], 
                linewidth=2, label='Carbon-Aware (trend)')
        
        ax.axhline(data['price_tou'].mean(), linestyle='--', 
                   color=self.colors['tou'], linewidth=2, label='TOU (mean)')
        
        ci_90 = data['carbon_intensity'].quantile(0.90)
        ax.axvline(ci_90, linestyle=':', color=self.colors['cpp'], 
                   linewidth=2, label=f'CPP threshold ({ci_90:.0f})')
        
        ax.set_xlabel('Carbon Intensity (gCO₂/kWh)')
        ax.set_ylabel('Price (CHF/kWh)')
        ax.set_title('Price Response to Carbon Intensity by Scheme')
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "fig_4_8_price_vs_carbon.pdf")
        plt.close()
        print(f"  Saved: fig_4_8_price_vs_carbon.pdf")
    
    # =========================================================================
    # SECTION 4.3: MAIN RESULTS
    # =========================================================================
    
    def _section_4_3_results(self, data: pd.DataFrame):
        """Generate main results figures."""
        self._fig_correlation_comparison(data)
        self._fig_emission_comparison(data)
        self._fig_demand_shift_heatmap(data)
        self._fig_winter_evening_detail(data)
    
    def _fig_correlation_comparison(self, data: pd.DataFrame):
        """Figure 4.9: Correlation comparison (KEY RESULT)."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        rho_tou = data['demand_tou'].corr(data['carbon_intensity'])
        rho_carbon = data['demand_carbon'].corr(data['carbon_intensity'])
        rho_cpp = data['demand_cpp'].corr(data['carbon_intensity'])
        
        schemes = ['TOU\n(Baseline)', 'Carbon-Aware\n(Intervention)', 'CPP\n(Alternative)']
        correlations = [rho_tou, rho_carbon, rho_cpp]
        colors = [self.colors['tou'], self.colors['carbon'], self.colors['cpp']]
        
        bars = ax.bar(schemes, correlations, color=colors, edgecolor='black', linewidth=1.5)
        
        for bar, val in zip(bars, correlations):
            height = bar.get_height()
            ax.annotate(f'{val:+.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5 if height > 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if height > 0 else 'top',
                        fontsize=14, fontweight='bold')
        
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.axhline(-0.15, color='red', linestyle='--', linewidth=2, 
                   label='H1 threshold (Δρ ≤ -0.15 from baseline)')
        
        ax.set_ylabel('Correlation ρ(Demand, Carbon Intensity)')
        ax.set_title('Temporal Alignment: Demand-Carbon Correlation')
        ax.set_ylim(-0.6, 0.5)
        
        ax.text(0.02, 0.98, 'Positive ρ = Problem\n(demand ↑ when dirty)',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))
        ax.text(0.02, 0.02, 'Negative ρ = Goal\n(demand ↑ when clean)',
                transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='honeydew', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "fig_4_9_correlation_comparison.pdf")
        plt.close()
        print(f"  Saved: fig_4_9_correlation_comparison.pdf")
    
    def _fig_emission_comparison(self, data: pd.DataFrame):
        """Figure 4.10: Emission reduction."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        e_tou = (data['demand_tou'] * data['carbon_intensity']).sum() / 1e9
        e_carbon = (data['demand_carbon'] * data['carbon_intensity']).sum() / 1e9
        e_cpp = (data['demand_cpp'] * data['carbon_intensity']).sum() / 1e9
        
        reduction_carbon = (e_tou - e_carbon) / e_tou * 100
        reduction_cpp = (e_tou - e_cpp) / e_tou * 100
        
        schemes = ['TOU\n(Baseline)', 'Carbon-Aware', 'CPP']
        emissions = [e_tou, e_carbon, e_cpp]
        colors = [self.colors['tou'], self.colors['carbon'], self.colors['cpp']]
        
        bars = ax.bar(schemes, emissions, color=colors, edgecolor='black', linewidth=1.5)
        
        ax.annotate(f'{e_tou:.0f} ktCO₂',
                    xy=(bars[0].get_x() + bars[0].get_width()/2, e_tou),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', fontsize=12, fontweight='bold')
        
        ax.annotate(f'{e_carbon:.0f} ktCO₂\n(-{reduction_carbon:.1f}%)',
                    xy=(bars[1].get_x() + bars[1].get_width()/2, e_carbon),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', fontsize=12, fontweight='bold', color='green')
        
        ax.annotate(f'{e_cpp:.0f} ktCO₂\n(-{reduction_cpp:.1f}%)',
                    xy=(bars[2].get_x() + bars[2].get_width()/2, e_cpp),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', fontsize=12, fontweight='bold', color='orange')
        
        ax.set_ylabel('Total Emissions (ktCO₂)')
        ax.set_title('Annual Emissions by Pricing Scheme')
        ax.set_ylim(0, e_tou * 1.15)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "fig_4_10_emission_comparison.pdf")
        plt.close()
        print(f"  Saved: fig_4_10_emission_comparison.pdf")
    
    def _fig_demand_shift_heatmap(self, data: pd.DataFrame):
        """Figure 4.11: Demand shift heatmap."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        data['demand_change_pct'] = (data['demand_carbon'] - data['demand_tou']) / data['demand_tou'] * 100
        
        pivot = data.pivot_table(values='demand_change_pct', 
                                 index='hour', 
                                 columns='month',
                                 aggfunc='mean')
        
        im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn',
                       vmin=-15, vmax=15)
        
        cbar = plt.colorbar(im, ax=ax, label='Demand Change (%)')
        
        ax.set_xticks(range(12))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.set_yticks(range(0, 24, 2))
        ax.set_yticklabels(range(0, 24, 2))
        
        ax.set_xlabel('Month')
        ax.set_ylabel('Hour of Day')
        ax.set_title('Demand Response: Carbon-Aware vs TOU')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "fig_4_11_demand_shift.pdf")
        plt.close()
        print(f"  Saved: fig_4_11_demand_shift.pdf")
    
    def _fig_winter_evening_detail(self, data: pd.DataFrame):
        """
        Figure 4.12: Winter evening H2 analysis.
        
        FIX: Right panel y-axis extended to prevent labels touching frame
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        winter_eve = data[(data['month'] == 1) & (data['hour'].isin([18, 19, 20]))]
        
        ax1.scatter(winter_eve['carbon_intensity'], 
                    winter_eve['demand_tou'] / 1e6,
                    alpha=0.5, label='TOU', color=self.colors['tou'])
        ax1.scatter(winter_eve['carbon_intensity'],
                    winter_eve['demand_carbon'] / 1e6,
                    alpha=0.5, label='Carbon-Aware', color=self.colors['carbon'])
        
        for demand_col, color in [('demand_tou', self.colors['tou']),
                                  ('demand_carbon', self.colors['carbon'])]:
            z = np.polyfit(winter_eve['carbon_intensity'], winter_eve[demand_col] / 1e6, 1)
            p = np.poly1d(z)
            x_line = np.linspace(winter_eve['carbon_intensity'].min(),
                                winter_eve['carbon_intensity'].max(), 100)
            ax1.plot(x_line, p(x_line), '--', color=color, linewidth=2)
        
        rho_tou = winter_eve['demand_tou'].corr(winter_eve['carbon_intensity'])
        rho_carbon = winter_eve['demand_carbon'].corr(winter_eve['carbon_intensity'])
        
        ax1.set_xlabel('Carbon Intensity (gCO₂/kWh)')
        ax1.set_ylabel('Demand (GWh)')
        ax1.set_title(f'Winter Evening (Jan 18-21h)\nρ: TOU={rho_tou:.3f}, Carbon={rho_carbon:.3f}')
        ax1.legend()
        
        # Right panel: Effect strength comparison
        overall_delta = data['demand_carbon'].corr(data['carbon_intensity']) - \
                        data['demand_tou'].corr(data['carbon_intensity'])
        winter_delta = rho_carbon - rho_tou
        
        bars = ax2.bar(['Overall', 'Winter Evening'], 
                       [overall_delta, winter_delta],
                       color=[self.colors['baseline'], self.colors['winter']],
                       edgecolor='black', linewidth=1.5)
        
        # FIX: Labels positioned inside bars to avoid touching frame
        for bar, val in zip(bars, [overall_delta, winter_delta]):
            ax2.annotate(f'{val:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, val * 0.85),
                        ha='center', va='top',
                        fontsize=14, fontweight='bold', color='white')
        
        ax2.axhline(-0.15, color='red', linestyle='--', linewidth=2, label='H1 threshold')
        ax2.set_ylabel('Δρ (Correlation Change)')
        ax2.set_title('Effect Strength Comparison')
        ax2.legend(loc='lower right')
        
        # FIX: Extend y-axis to give more room at bottom
        ymin = min(overall_delta, winter_delta) * 1.15
        ax2.set_ylim(ymin, 0.05)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "fig_4_12_winter_evening.pdf")
        plt.close()
        print(f"  Saved: fig_4_12_winter_evening.pdf")
    
    # =========================================================================
    # SECTION 4.4: SENSITIVITY ANALYSIS
    # =========================================================================
    
    def _section_4_4_sensitivity(self, sensitivity_data: pd.DataFrame):
        """Generate sensitivity analysis figures."""
        # Check format and generate appropriate figures
        if 'parameter' in sensitivity_data.columns:
            self._fig_sensitivity_main_effects(sensitivity_data)
        else:
            print("  [WARN] Unrecognized sensitivity data format, skipping figures")
    
    def _fig_sensitivity_main_effects(self, sensitivity_data: pd.DataFrame):
        """
        Figure 4.13: Main effects sensitivity analysis.
        
        Shows how each parameter (α, τ, λ) affects Δρ and emission reduction.
        """
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        
        params = ['alpha', 'tau', 'lambda']
        titles = ['Carbon Weight (α)', 'Tech Amplification (τ)', 'Loss Aversion (λ)']
        
        # Top row: Delta_rho
        for ax, param, title in zip(axes[0], params, titles):
            param_data = sensitivity_data[sensitivity_data['parameter'] == param].copy()
            if len(param_data) == 0:
                ax.text(0.5, 0.5, f'No data for {param}', ha='center', va='center')
                continue
            
            param_data = param_data.sort_values('value')
            ax.plot(param_data['value'], param_data['delta_rho'], 'o-',
                   color=self.colors['carbon'], linewidth=2, markersize=10)
            ax.axhline(-0.15, color='red', linestyle='--', linewidth=2, label='Threshold')
            ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xlabel(title)
            ax.set_ylabel('Δρ')
            ax.set_title(f'Effect of {title} on Δρ')
            ax.legend(loc='best', fontsize=9)
            ax.set_ylim(-0.85, 0.1)
        
        # Bottom row: Emission reduction
        for ax, param, title in zip(axes[1], params, titles):
            param_data = sensitivity_data[sensitivity_data['parameter'] == param].copy()
            if len(param_data) == 0:
                continue
            
            param_data = param_data.sort_values('value')
            ax.plot(param_data['value'], param_data['emission_reduction_pct'], 'o-',
                   color=self.colors['carbon'], linewidth=2, markersize=10)
            ax.set_xlabel(title)
            ax.set_ylabel('Emission Reduction (%)')
            ax.set_title(f'Effect of {title} on Emissions')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "fig_4_13_sensitivity_main_effects.pdf")
        plt.close()
        print(f"  Saved: fig_4_13_sensitivity_main_effects.pdf")
