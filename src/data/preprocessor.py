import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

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

def scale_data(X_train, X_test):
    """Scale the data using RobustScaler."""
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler 