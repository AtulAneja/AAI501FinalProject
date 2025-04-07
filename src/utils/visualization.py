import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from src.data.data_loader import load_tariffs

def apply_tariff_impact(value, tariff_rate, year, implementation_year=2025):
    """Calculate the impact of tariffs on trade value with gradual implementation."""
    if year < implementation_year:
        return value
    
    # Calculate years since implementation
    years_since_implementation = year - implementation_year
    
    # Maximum impact is 30% of the tariff rate, reached gradually over 3 years
    max_impact = (tariff_rate / 100) * 0.3
    impact_factor = min(max_impact * (years_since_implementation / 3), max_impact)
    
    return value * (1 - impact_factor)

def visualize_trade_data(exports_df, imports_df, predictions=None):
    """Visualize average trade data and predictions across all countries."""
    print("Starting visualization...")
    plt.figure(figsize=(15, 8))
    
    # Get years for x-axis
    historical_years = [str(year) for year in range(1992, 2024)]
    future_years = [str(year) for year in range(2024, 2036)]
    all_years = historical_years + future_years
    print(f"Years range: {historical_years[0]} to {future_years[-1]}")
    
    # Calculate average trade values across all countries
    exports_values = exports_df[historical_years].mean()
    imports_values = imports_df[historical_years].mean()
    print("Calculated average trade values")
    
    # Convert from log scale back to original scale
    exports_values = np.exp(exports_values) - 1
    imports_values = np.exp(imports_values) - 1
    print(f"Value ranges - Exports: {exports_values.min():.2f} to {exports_values.max():.2f}")
    print(f"Value ranges - Imports: {imports_values.min():.2f} to {imports_values.max():.2f}")
    
    # Plot historical data
    plt.plot(historical_years, exports_values, 'b-', label='Average Exports', linewidth=2)
    plt.plot(historical_years, imports_values, 'r-', label='Average Imports', linewidth=2)
    print("Plotted historical data")
    
    if predictions is not None:
        print("Processing predictions...")
        # Load tariff data
        tariffs = load_tariffs()
        avg_tariff = sum(tariffs.values()) / len(tariffs)
        print(f"Average tariff rate: {avg_tariff:.2f}%")
        
        # Convert predictions from log scale back to original scale
        pred_exports = np.exp(predictions[0]) - 1
        pred_imports = np.exp(predictions[1]) - 1
        
        # Calculate growth rates based on recent historical data
        last_5_exports = exports_values.values[-5:]
        last_5_imports = imports_values.values[-5:]
        exports_growth = (last_5_exports[-1] - last_5_exports[0]) / last_5_exports[0] / 5
        imports_growth = (last_5_imports[-1] - last_5_imports[0]) / last_5_imports[0] / 5
        
        # Adjust initial predictions based on last historical values if they seem too low
        if pred_exports < exports_values.iloc[-1]:
            pred_exports = exports_values.iloc[-1] * (1 + exports_growth)
        if pred_imports < imports_values.iloc[-1]:
            pred_imports = imports_values.iloc[-1] * (1 + imports_growth)
        
        print(f"Initial predictions - Exports: {pred_exports:.2f}, Imports: {pred_imports:.2f}")
        print(f"Growth rates - Exports: {exports_growth:.2%}, Imports: {imports_growth:.2%}")
        
        # Generate future predictions
        future_exports = [pred_exports]
        future_imports = [pred_imports]
        
        for i, year in enumerate(range(2024, 2036)):
            if i > 0:  # Skip first year as it's already set
                next_export = future_exports[-1] * (1 + exports_growth)
                next_import = future_imports[-1] * (1 + imports_growth)
                
                # Apply tariff impact after April 3, 2025
                if year >= 2025:
                    next_export = apply_tariff_impact(next_export, avg_tariff, year)
                    next_import = apply_tariff_impact(next_import, avg_tariff, year)
                
                future_exports.append(next_export)
                future_imports.append(next_import)
        
        print(f"Generated {len(future_exports)} future predictions")
        print(f"Final predictions - Exports: {future_exports[-1]:.2f}, Imports: {future_imports[-1]:.2f}")
        
        # Plot predictions
        plt.plot(['2023'] + future_years[:future_years.index('2025')+1], 
                [exports_values.iloc[-1]] + future_exports[:future_years.index('2025')+1],
                'b--', label='Predicted Exports (Pre-Tariffs)', linewidth=2, alpha=0.7)
        plt.plot(future_years[future_years.index('2025'):], 
                future_exports[future_years.index('2025'):],
                'b:', label='Predicted Exports (Post-Tariffs)', linewidth=2, alpha=0.7)
        
        plt.plot(['2023'] + future_years[:future_years.index('2025')+1], 
                [imports_values.iloc[-1]] + future_imports[:future_years.index('2025')+1],
                'r--', label='Predicted Imports (Pre-Tariffs)', linewidth=2, alpha=0.7)
        plt.plot(future_years[future_years.index('2025'):], 
                future_imports[future_years.index('2025'):],
                'r:', label='Predicted Imports (Post-Tariffs)', linewidth=2, alpha=0.7)
        print("Plotted prediction lines")
        
        # Add vertical line for tariff implementation
        plt.axvline(x='2025', color='orange', linestyle='--', alpha=0.5)
        plt.text('2025', plt.ylim()[1]*0.9, 'New Tariffs\nImplemented', 
                ha='right', va='top', fontsize=10, color='orange')
        
        # Plot prediction markers
        plt.plot(future_years, future_exports, 'b*', markersize=8, alpha=0.7)
        plt.plot(future_years, future_imports, 'r*', markersize=8, alpha=0.7)
        print("Added markers and annotations")
        
        # Add prediction values as text for key years
        plt.text('2024', future_exports[0], f'{future_exports[0]/1e9:.1f}B', 
                ha='right', va='bottom', fontsize=10, color='blue')
        plt.text('2024', future_imports[0], f'{future_imports[0]/1e9:.1f}B', 
                ha='right', va='top', fontsize=10, color='red')
        
        plt.text('2025', future_exports[1], f'{future_exports[1]/1e9:.1f}B', 
                ha='right', va='bottom', fontsize=10, color='blue')
        plt.text('2025', future_imports[1], f'{future_imports[1]/1e9:.1f}B', 
                ha='right', va='top', fontsize=10, color='red')
        
        plt.text('2035', future_exports[-1], f'{future_exports[-1]/1e9:.1f}B', 
                ha='right', va='bottom', fontsize=10, color='blue')
        plt.text('2035', future_imports[-1], f'{future_imports[-1]/1e9:.1f}B', 
                ha='right', va='top', fontsize=10, color='red')
    
    # Add vertical line to separate historical and predicted data
    plt.axvline(x='2023', color='gray', linestyle='--', alpha=0.5)
    
    # Add labels and title
    plt.title('Global Trade Trends: Historical Data and Future Predictions (1992-2035)\nIncluding Tariff Impact from April 2025', 
              fontsize=16, pad=20)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Trade Value (USD)', fontsize=14)
    
    # Format y-axis to show values in billions
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e9:.0f}B'))
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add text annotation for the prediction period
    plt.text('2023.5', plt.ylim()[1]*0.95, 'Predictions', 
            ha='center', va='center', fontsize=12, color='gray')
    
    print("Finalizing plot...")
    plt.tight_layout()
    
    # Save with explicit path and print confirmation
    save_path = 'trade_visualization.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {save_path}")
    plt.close()
    print("Visualization complete") 