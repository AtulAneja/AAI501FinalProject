
###############################################################################
# LMST and Catboost Model Examples
###############################################################################
import numpy as np  # Import NumPy for numerical operations
import pandas as pd  # Import Pandas for data manipulation (e.g., reading CSV files)
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from load_normalized_annual_trade_dataset import load_normalized_annual_world_trade_data

# Required Modeling libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def run_lstm_model_example():
    df = load_normalized_annual_world_trade_data()


    features = ['Exports', 'Imports', 'CPI', 'GDP', 'Tariff_Rate']

    # ERROR: KNNInputer lib wasn't working properly, so I will leave the cod
    #        potentially  fix in future
    # Impute missing values
    #imputer = KNNImputer(n_neighbors=5)
    #df[features] = imputer.fit_transform(df[features])
    
    print("****** Shape of dataset before dropping bad rows:", df.shape[0])
    df = df.dropna()
    print("****** Shape of dataset AFTER dropping bad rows:", df.shape[0])
    
    # Normalize features
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    
    
    sequence_length = 5
    X, y = [], []
    
    grouped = df.groupby('Country')
    
    for country, group in grouped:
        group = group.sort_values('Year').reset_index(drop=True)
        if len(group) < sequence_length + 1:
            continue  # Skip short series
        for i in range(len(group) - sequence_length):
            X.append(group.loc[i:i+sequence_length-1, ['Imports', 'Exports', 'CPI', 'Tariff_Rate']].values)
            y.append(group.loc[i + sequence_length, 'GDP'])
    
    X = np.array(X)
    y = np.array(y) 
    
    print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42 )
    
    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.1)
    
    # Evaluate model
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")
    
    # Predict
    y_pred = model.predict(X_test)
    print("Num of X values = ", X_test.shape)
    print("Num of y values = ", y_pred.shape)
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label='True GDP')
    plt.plot(y_pred, label='Predicted GDP')
    plt.legend()
    plt.title("True vs Predicted GDP")
    plt.show()
    
    def invert_scaling(scaler, features, y):
        # Extract min and max for GDP only
        gdp_index = features.index('GDP')
        gdp_min = scaler.data_min_[gdp_index]
        gdp_max = scaler.data_max_[gdp_index]
        
        return  y * (gdp_max - gdp_min) + gdp_min

    y_test_actual = invert_scaling(scaler, features, y_test)
    y_pred_actual = invert_scaling(scaler, features, y_pred)
    
    # Calculate errors
    errors = y_test_actual - y_pred_actual
    
    # Accuracy metrics
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    r2 = r2_score(y_test_actual, y_pred_actual)
    mape = np.mean(np.abs(errors / y_test_actual)) * 100
    
    # Display metrics
    print("Model Accuracy Metrics:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_actual[:150], label='True GDP')
    plt.plot(y_pred_actual[:150], label='Predicted GDP')
    plt.title("True vs Predicted GDP (Original Scale)")
    plt.xlabel("Sample")
    plt.ylabel("GDP")
    plt.legend()
    plt.show()
    
    
    def make_graph(model, scaler, df):#, test):
        # Select country
        test_cont = ["Australia", 'Mexico', 'Switzerland', 'China', 'United States']  # or any country from df['Country'].unique()
        
        for target_country in test_cont:    
            # Get country data
            country_df = df[df['Country'] == target_country].sort_values('Year').reset_index(drop=True)
            
            # Prepare sequences
            seq_len = 5
            X_country = []
            years = []
            
            for i in range(len(country_df) - seq_len):
                sequence = country_df.loc[i:i+seq_len-1, ['Imports', 'Exports', 'CPI', 'Tariff_Rate']].values
                X_country.append(sequence)
                years.append(country_df.loc[i + seq_len, 'Year'])  # year being predicted
            
            X_country = np.array(X_country)
            
            # Predict GDP (scaled)
            y_pred_scaled = model.predict(X_country).flatten()
            
            # Invert scaling for GDP
            gdp_index = features.index('GDP')
            gdp_min = scaler.data_min_[gdp_index]
            gdp_max = scaler.data_max_[gdp_index]
            
            y_pred = y_pred_scaled * (gdp_max - gdp_min) + gdp_min
        
            actual_gdp = country_df.loc[seq_len:, 'GDP'].values
            actual_gdp = actual_gdp * (gdp_max - gdp_min) + gdp_min  # unscale GDP if still scaled
            plt.figure(figsize=(12, 6))
            plt.plot(years, actual_gdp, label='Actual GDP', marker='o')
            plt.plot(years, y_pred, label='Predicted GDP', marker='x')
            
            # Highlight the last prediction
            plt.scatter(years[-1], y_pred[-1], color='red', label='Last Predicted GDP', zorder=5)
            plt.scatter(years[-1], actual_gdp[-1], color='green', label='Actual GDP (Last Year)', zorder=5)
            
            plt.title(f"{target_country} GDP Over Time (Actual vs Predicted)")
            plt.xlabel("Year")
            plt.ylabel("GDP")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
    
    
    make_graph(model, scaler, df)

if __name__ == "__main__":
    run_lstm_model_example() 