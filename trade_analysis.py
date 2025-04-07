import numpy as np
import pandas as pd
import xmltodict
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime

def load_xml_data(file_path):
    """Load and parse XML data into a pandas DataFrame."""
    with open(file_path, 'r', encoding='utf-8') as file:
        xml_data = xmltodict.parse(file.read())
    
    # Find the REPORT worksheet
    workbook = xml_data['ss:Workbook']
    worksheets = workbook['Worksheet']
    report_worksheet = None
    
    for worksheet in worksheets:
        if worksheet['@ss:Name'] == 'REPORT':
            report_worksheet = worksheet
            break
    
    if not report_worksheet:
        raise ValueError("REPORT worksheet not found")
    
    table = report_worksheet['Table']
    rows = table['Row']
    
    # Skip the first 4 rows (title and metadata)
    data_rows = rows[4:]
    
    # Extract headers (first row after metadata)
    headers = []
    header_cells = data_rows[0]['Cell']
    for cell in header_cells:
        if 'Data' in cell and '#text' in cell['Data']:
            headers.append(cell['Data']['#text'])
        else:
            headers.append(f'Column_{len(headers)}')
    
    # Process data rows
    data = []
    for row in data_rows[1:]:  # Skip header row
        if 'Cell' not in row:
            continue
            
        row_data = [''] * len(headers)  # Initialize with empty strings
        cells = row['Cell']
        if not isinstance(cells, list):
            cells = [cells]  # Handle single cell case
        
        for i, cell in enumerate(cells):
            if 'Data' in cell and '#text' in cell['Data']:
                # Get the index, accounting for merged cells
                if 'ss:Index' in cell:
                    idx = int(cell['ss:Index']) - 1
                else:
                    idx = i
                
                if idx < len(row_data):
                    row_data[idx] = cell['Data']['#text']
        
        # Only add rows that have some non-empty values
        if any(val != '' for val in row_data):
            data.append(row_data)
    
    df = pd.DataFrame(data, columns=headers)
    print(f"\nSample of data from {file_path}:")
    print(df.head())
    return df

def preprocess_data(df):
    """Preprocess the data for deep learning."""
    # Remove any non-numeric characters and convert to float
    for col in df.columns:
        if col != 'Time':  # Skip the Time column
            # Remove commas and other non-numeric characters
            df[col] = df[col].str.replace(r'[^\d.-]', '', regex=True)
            # Convert to float and handle invalid values
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill missing values with median of the column
    for col in df.columns:
        if col != 'Time':
            df[col] = df[col].fillna(df[col].median())
    
    # Remove rows where all values are 0 or missing
    numeric_cols = df.columns[df.columns != 'Time']
    df = df[~(df[numeric_cols] == 0).all(axis=1)]
    
    # Apply log transformation to handle large values
    for col in df.columns:
        if col != 'Time':
            # Add a small constant to handle zeros
            min_positive = df[col][df[col] > 0].min()
            df[col] = np.log1p(df[col].clip(lower=min_positive/10))
    
    return df

def prepare_data(exports_df, imports_df, target_year='2024'):
    """Prepare data for training by using previous years to predict target year."""
    # Drop the Time column and the target year
    X_exports = exports_df.drop(['Time', target_year], axis=1)
    X_imports = imports_df.drop(['Time', target_year], axis=1)
    
    # Get target values
    y_exports = exports_df[target_year]
    y_imports = imports_df[target_year]
    
    # Convert to numeric, handling any non-numeric values
    X_exports = X_exports.apply(pd.to_numeric, errors='coerce')
    X_imports = X_imports.apply(pd.to_numeric, errors='coerce')
    y_exports = pd.to_numeric(y_exports, errors='coerce')
    y_imports = pd.to_numeric(y_imports, errors='coerce')
    
    # Fill NaN values with the mean of the column
    for col in X_exports.columns:
        X_exports[col] = X_exports[col].fillna(X_exports[col].mean())
    for col in X_imports.columns:
        X_imports[col] = X_imports[col].fillna(X_imports[col].mean())
    
    y_exports = y_exports.fillna(y_exports.mean())
    y_imports = y_imports.fillna(y_imports.mean())
    
    # Create separate datasets for exports and imports
    X = pd.concat([X_exports, X_imports], axis=0)
    y = pd.concat([y_exports, y_imports], axis=0)
    
    return X, y

def build_model(input_shape):
    """Build a deep learning model."""
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),  # Properly specify input shape as tuple
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    
    # Use gradient clipping to prevent exploding gradients
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss='huber',
        metrics=['mae']
    )
    return model

def load_tariffs(file_path='tariffs.csv'):
    """Load tariff data from CSV file."""
    tariffs_df = pd.read_csv(file_path)
    # Convert to dictionary for easier lookup
    tariffs = dict(zip(tariffs_df['Country'], tariffs_df['Tariff Rate']))
    return tariffs

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

def main():
    # Load data
    print("Loading data...")
    exports_df = load_xml_data('ExportsPerCountry.xml')
    imports_df = load_xml_data('ImportsPerCountry.xml')
    
    # Print data info
    print("\nExports DataFrame Info:")
    print(exports_df.info())
    print("\nImports DataFrame Info:")
    print(imports_df.info())
    
    # Preprocess data
    print("\nPreprocessing data...")
    exports_df = preprocess_data(exports_df)
    imports_df = preprocess_data(imports_df)
    
    # Store the last 5 years of actual values for trend calculation
    historical_years = [str(year) for year in range(2019, 2024)]
    last_5_exports = exports_df[historical_years].mean()
    last_5_imports = imports_df[historical_years].mean()
    
    # Prepare data for training
    X, y = prepare_data(exports_df, imports_df, target_year='2024')
    
    # Scale the features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Build and train model
    print("\nTraining model...")
    model = build_model(X_train.shape[1])
    
    # Add callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss = model.evaluate(X_test, y_test)
    print(f"Test loss (Huber): {test_loss[0]:.4f}")
    print(f"Test MAE: {test_loss[1]:.4f}")
    
    # Make predictions for visualization
    print("\nMaking predictions for visualization...")
    
    # Calculate the average growth rate from the last 5 years
    exports_growth = (np.exp(last_5_exports.iloc[-1]) - np.exp(last_5_exports.iloc[0])) / np.exp(last_5_exports.iloc[0]) / 5
    imports_growth = (np.exp(last_5_imports.iloc[-1]) - np.exp(last_5_imports.iloc[0])) / np.exp(last_5_imports.iloc[0]) / 5
    
    # Use the last actual values and growth rates for initial predictions
    pred_exports = np.log1p(np.exp(last_5_exports.iloc[-1]) * (1 + exports_growth))
    pred_imports = np.log1p(np.exp(last_5_imports.iloc[-1]) * (1 + imports_growth))
    
    # Visualize data and predictions
    print("\nGenerating visualization...")
    visualize_trade_data(exports_df, imports_df, predictions=[pred_exports, pred_imports])

if __name__ == "__main__":
    main() 