# Global Trade Analysis and Prediction System

## Project Overview
This project implements a comprehensive system for analyzing and predicting global trade patterns, economic growth, and trade relationships between countries. The system uses various machine learning models and statistical analysis to predict future trade patterns, GDP growth, and assess economic indicators.

## Project Structure
```
project/
├── src/                    # Source code directory
├── data/                   # Raw data files
├── docs/                   # Documentation and reports
├── cpi_per_country.csv     # Consumer Price Index data
├── GDP1960-2023.csv        # Historical GDP data
├── 34_years_world_export_import_dataset.csv  # Trade data
├── World Census.xls        # Population and demographic data
├── Trade Growth Prediction_using CAGR indicator.ipynb  # CAGR analysis
├── AAI_501_ClassificationModel_Predictive_GDP_Growth_using_CAGR_2004-2023.ipynb  # GDP growth prediction
├── load_normalized_annual_trade_dataset.py   # Data preprocessing
├── lstm_mult_feature_model.py                # LSTM model implementation
├── catboost_gradient_boost_model.py          # CatBoost model implementation
├── trade_analysis.py                         # Trade analysis utilities
└── requirements.txt                          # Project dependencies
```

## Key Components

### Data Analysis and Processing
- **Trade Data Analysis**: Analysis of 34 years of world export/import data
- **GDP Growth Prediction**: Using CAGR indicators and machine learning models
- **Economic Indicators**: Integration of CPI, GDP, and census data
- **Data Normalization**: Preprocessing and normalization of trade datasets

### Machine Learning Models
1. **LSTM Model** (`lstm_mult_feature_model.py`)
   - Multi-feature time series prediction
   - Long-term trade pattern analysis
   - Sequence modeling for economic trends

2. **CatBoost Model** (`catboost_gradient_boost_model.py`)
   - Gradient boosting for trade prediction
   - Feature importance analysis
   - Robust prediction handling

3. **CAGR-based Analysis** (`Trade Growth Prediction_using CAGR indicator.ipynb`)
   - Compound Annual Growth Rate calculations
   - Growth trend analysis
   - Future projections

### Visualization and Analysis
- Trade pattern visualization
- Economic growth analysis
- Country-specific trade relationships
- Interactive data exploration

## Requirements
- Python 3.8+
- pandas
- numpy
- matplotlib
- scikit-learn
- tensorflow
- catboost
- jupyter
- xlrd (for Excel file support)

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run data preprocessing:
   ```bash
   python load_normalized_annual_trade_dataset.py
   ```

3. Train models:
   ```bash
   python lstm_mult_feature_model.py
   python catboost_gradient_boost_model.py
   ```

4. Explore analysis notebooks:
   - Open `Trade Growth Prediction_using CAGR indicator.ipynb`
   - Open `AAI_501_ClassificationModel_Predictive_GDP_Growth_using_CAGR_2004-2023.ipynb`

## Results
The system provides:
- Trade pattern predictions
- GDP growth forecasts
- Economic indicator analysis
- Country-specific trade relationship insights
- Visualizations of economic trends

## Future Improvements
1. Enhanced feature engineering
2. Integration of more economic indicators
3. Real-time data updates
4. Advanced visualization dashboards
5. API endpoints for predictions
6. Automated report generation

## Documentation
See the `docs/` directory for detailed documentation on:
- Data processing pipeline
- Model architectures
- Analysis methodologies
- Visualization techniques 