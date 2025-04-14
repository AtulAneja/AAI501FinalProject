import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error

# Model metrics libraries
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.data_loader import load_normalized_annual_world_trade_data
import matplotlib.pyplot as plt

def run_catboost_gradient_regression_example():
    # Load core table
    df = load_normalized_annual_world_trade_data()

    def create_lag_features(df, lags=[1, 2]):
        for lag in lags:
            df[f'GDP_lag{lag}'] = df.groupby('Country')['GDP'].shift(lag)
            df[f'CPI_lag{lag}'] = df.groupby('Country')['CPI'].shift(lag)
            df[f'Imports_lag{lag}'] = df.groupby('Country')['Exports'].shift(lag)
            df[f'Exports_lag{lag}'] = df.groupby('Country')['Imports'].shift(lag)
            df[f'Tariff_lag{lag}'] = df.groupby('Country')['Tariff_Rate'].shift(lag)
        return df
    
    df = create_lag_features(df)
    
    df.dropna(inplace=True)
    
    # Train: up to 2020, Test: 2021–2023
    train_df = df[df['Year'] <= 2020]
    test_df = df[df['Year'] > 2020]
    
    features = [col for col in df.columns if 'lag' in col]
    target = 'GDP'
    
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]
    
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function='MAE',
        verbose=100
    )
    
    model.fit(X_train, y_train)
    
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate model prediction accuracy
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)    
    print(f"\nMAE on test set: {mae:.2f}")
    print(f"R² Score: {r2:.4f}\n")
    
    # Optional: add predictions to test_df
    test_df['GDP_pred'] = y_pred
    
    test_cont = ["Australia", 'Mexico', 'Switzerland', 'China', 'United States']
    
    def walk_forward_plots(countries):
        for target_country in countries:
            # Parameters
            start_year = 2000
            end_year = 2021
            lag_window = 2  # depending on how many lags you trained with
            
            # Get all data for country
            country_df = df[df['Country'] == target_country].sort_values('Year').reset_index(drop=True)
            
            # Create empty list for predictions
            predictions = []
            
            # Walk-forward loop
            for year in range(start_year + lag_window, end_year + 1):
                # Get current row for prediction
                current_row = country_df[country_df['Year'] == year].copy()
            
                # Build feature row using lags from past rows
                for lag in range(1, lag_window + 1):
                    past_row = country_df[country_df['Year'] == year - lag]
                    if not past_row.empty:
                        current_row[f'GDP_lag{lag}'] = past_row['GDP'].values[0]
                        current_row[f'CPI_lag{lag}'] = past_row['CPI'].values[0]
                        current_row[f'Imports_lag{lag}'] = past_row['Imports'].values[0]
                        current_row[f'Exports_lag{lag}'] = past_row['Exports'].values[0]
                        current_row[f'Tariff_lag{lag}'] = past_row['Tariff_Rate'].values[0]
                    else:
                        current_row[f'GDP_lag{lag}'] = np.nan
            
                # Skip if we don't have all lag features
                if current_row[features].isnull().any().any():
                    continue
            
                # Predict using CatBoost
                pred = model.predict(Pool(current_row[features]))[0]
            
                # Store year, prediction, and actual
                predictions.append({
                    'Year': year,
                    'Predicted_GDP': pred,
                    'Actual_GDP': current_row['GDP'].values[0]
                })
                
            pred_df = pd.DataFrame(predictions)
            
            plt.figure(figsize=(12, 6))
            plt.plot(pred_df['Year'], pred_df['Actual_GDP'], label='Actual GDP', marker='o', linewidth=2)
            plt.plot(pred_df['Year'], pred_df['Predicted_GDP'], label='Predicted GDP (Walk-Forward)', marker='x', linestyle='--', linewidth=2)
            plt.title(f"{target_country} Walk-Forward GDP Forecast (CatBoost)")
            plt.xlabel("Year")
            plt.ylabel("GDP")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
            
    walk_forward_plots(test_cont)

if __name__ == "__main__":
    run_catboost_gradient_regression_example() 