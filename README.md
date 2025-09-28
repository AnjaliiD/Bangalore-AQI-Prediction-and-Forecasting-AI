# Air Quality Index Forecasting and Prediction for Bangalore

A deep learning project that forecasts Air Quality Index (AQI) for Bangalore using GRU (Gated Recurrent Unit) neural networks. The model analyzes historical air quality data including multiple pollutants to predict future AQI values with high accuracy.

## Overview

This project implements a time series forecasting model to predict air quality in Bangalore for the next 7 days. By leveraging historical data on various air pollutants, the model achieves strong predictive performance with an R² score of 0.70 and RMSE of 13.41.

## Features

- **Multi-feature Time Series Analysis**: Incorporates 7 air quality parameters (AQI, PM2.5, PM10, NO2, SO2, CO, O3)
- **GRU Neural Network Architecture**: Uses stacked GRU layers with L2 regularization to prevent overfitting
- **7-Day Forecasting**: Generates predictions for the next week with confidence
- **Comprehensive Visualizations**: Includes actual vs predicted comparisons, forecast plots, and training metrics
- **Robust Training Pipeline**: Implements early stopping and learning rate scheduling for optimal model performance

## Model Architecture

The model consists of:
- Two stacked GRU layers (128 and 64 units)
- L2 regularization for improved generalization
- Adam optimizer with adaptive learning rate
- 30-day sliding window for sequential prediction

## Performance Metrics

- **MAE (Mean Absolute Error)**: 9.46
- **RMSE (Root Mean Square Error)**: 13.41
- **R² Score**: 0.7006

The model successfully captures air quality patterns and trends, achieving convergence after 40 epochs with validation loss of 0.0062.

## Installation

### Prerequisites

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

### Setup

```bash
# Clone the repository
git clone https://github.com/AnjaliiD/Bangalore-AQI-Prediction-and-Forecasting-AI.git
cd Bangalore-AQI-Prediction-and-Forecasting-AI

# Install required packages
pip install -r requirements.txt
```

## Usage

### Training the Model

```python
# Load and preprocess data
df = pd.read_csv('Bangalore_AQI_Dataset.csv')

# Train the model
model = Sequential()
model.add(GRU(128, return_sequences=True, input_shape=(window_size, len(features)), kernel_regularizer=l2(0.001)))
model.add(GRU(64, kernel_regularizer=l2(0.00005)))
model.add(Dense(1))

# Fit the model
history = model.fit(X_train, y_train, epochs=40, batch_size=32, validation_split=0.1, callbacks=[early_stop, lr_scheduler])
```

### Making Predictions

```python
# Forecast next 7 days
forecasted_values = forecast_future(model, last_sequence, 7, scaler, aqi_scaler)
```

## Dataset

The model expects a CSV file with the following columns:
- `Date`: Date of measurement (format: DD/MM/YY)
- `AQI`: Air Quality Index
- `PM2.5`: Particulate Matter 2.5
- `PM10`: Particulate Matter 10
- `NO2`: Nitrogen Dioxide
- `SO2`: Sulfur Dioxide
- `CO`: Carbon Monoxide
- `O3`: Ozone

## Model Training Details

- **Window Size**: 30 days
- **Training Split**: 80-20 train-test split
- **Batch Size**: 32
- **Initial Learning Rate**: 0.001
- **Learning Rate Reduction**: 0.5x every 3 epochs without improvement
- **Early Stopping**: Patience of 5 epochs

## Visualizations

The project generates three key visualizations:

1. **Actual vs Predicted AQI**: Comparison of model predictions against actual values on test data
2. **7-Day Forecast**: Predicted AQI values for the upcoming week
3. **Training Loss Curve**: Training and validation loss progression over epochs

## Results

The model demonstrates strong predictive capability with validation loss converging to 0.0062. The forecasting system provides reliable short-term air quality predictions that can be used for:

- Public health advisories
- Environmental planning
- Pollution control measures
- Citizen awareness initiatives

## Future Improvements

- Incorporate weather data (temperature, humidity, wind speed) as additional features
- Experiment with hybrid models (GRU + CNN)
- Extend forecasting window to 14-30 days
- Implement ensemble methods for improved accuracy
- Add real-time data integration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Air quality data sourced from Bangalore monitoring stations and AQI.in
- Built with TensorFlow and Keras
- Inspired by environmental health research

## Contact

For questions or feedback, please open an issue on GitHub or email me at anjalidesai0111@gmail.com.
