# Global Trade Analysis and Prediction System

## Project Overview
This project implements a machine learning-based system for analyzing and predicting global trade patterns, with a specific focus on the impact of tariffs on international trade flows. The system uses historical trade data to train a deep learning model that can predict future trade patterns and assess the potential impact of tariff changes.

## Project Structure
```
project/
├── src/
│   ├── data/           # Data processing and loading
│   ├── models/         # Model architecture and training
│   └── utils/          # Utility functions and visualization
├── docs/               # Documentation and reports
└── data/              # Raw data files
```

## Methodology

### Data Processing
1. **Data Collection**: Historical trade data from 1992-2023
2. **Preprocessing**:
   - Handling missing values
   - Log transformation for large values
   - Robust scaling for model stability
   - Country-specific tariff integration

### Model Architecture
The project uses a deep neural network with the following architecture:
- Input Layer: Size based on historical trade features
- Hidden Layers: 
  - Dense layer (64 units) with ReLU activation
  - Dropout (0.2) for regularization
  - Dense layer (32 units) with ReLU activation
  - Dropout (0.2) for regularization
- Output Layer: Single unit with linear activation

### Training Process
- Loss Function: Huber loss for robustness against outliers
- Optimizer: Adam with gradient clipping
- Training Strategy:
  - Early stopping to prevent overfitting
  - Learning rate reduction on plateau
  - 50 epochs with batch size of 32

### Prediction Methodology
1. **Initial Predictions**: Based on historical growth rates
2. **Tariff Impact**: 
   - Gradual implementation of tariff effects
   - Country-specific tariff rates
   - Maximum impact capped at 30% of tariff rate
3. **Growth Rate Calculation**: 
   - Linear growth rates from last 5 years
   - Adjusted for tariff impacts post-2025

## Why This Model?

### 1. Deep Learning Approach
- **Complex Patterns**: Deep learning can capture complex, non-linear relationships in trade data
- **Feature Learning**: Automatically learns relevant features from historical data
- **Scalability**: Can handle large datasets and multiple features effectively

### 2. Model Architecture Choices
- **Dropout Layers**: Prevent overfitting in the presence of noisy trade data
- **Huber Loss**: Robust against outliers in trade values
- **Gradient Clipping**: Stabilizes training with large value ranges

### 3. Data Processing Decisions
- **Log Transformation**: Handles the large range of trade values
- **Robust Scaling**: Reduces impact of outliers
- **Mean Imputation**: Preserves data distribution for missing values

### 4. Prediction Strategy
- **Gradual Tariff Impact**: More realistic than immediate effects
- **Country-Specific Rates**: Accounts for different trade relationships
- **Growth Rate Adjustment**: Balances historical trends with tariff impacts

## Results
The model achieves:
- Test Loss (Huber): 0.68
- Test MAE: 1.09
- Realistic predictions aligned with historical trends
- Smooth transition in tariff impact predictions

## Future Improvements
1. Incorporate more economic indicators
2. Add seasonal patterns in trade data
3. Implement ensemble methods for more robust predictions
4. Add confidence intervals to predictions
5. Include more sophisticated tariff impact modeling

## Requirements
- Python 3.8+
- TensorFlow 2.x
- pandas
- numpy
- matplotlib
- scikit-learn
- xmltodict

## Usage
1. Place trade data XML files in the data directory
2. Run `python src/main.py`
3. View generated visualizations and predictions

## Documentation
See the `docs/` directory for detailed documentation on:
- Data processing pipeline
- Model architecture
- Training process
- Prediction methodology
- Visualization techniques 