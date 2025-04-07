import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import load_xml_data, load_tariffs
from src.data.preprocessor import preprocess_data, prepare_data, scale_data
from src.models.model import build_model, train_model, evaluate_model
from src.utils.visualization import visualize_trade_data
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    # Load data
    print("Loading data...")
    exports_df = load_xml_data('data/ExportsPerCountry.xml')
    imports_df = load_xml_data('data/ImportsPerCountry.xml')
    
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
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    
    # Build and train model
    print("\nTraining model...")
    model = build_model(X_train_scaled.shape[1])
    
    # Train the model
    history = train_model(
        model,
        X_train_scaled, y_train,
        X_test_scaled, y_test
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss = evaluate_model(model, X_test_scaled, y_test)
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