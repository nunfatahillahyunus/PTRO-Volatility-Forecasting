# PTRO Volatility Forecasting using LSTM

Web application for analyzing and forecasting the volatility of **PT Petrosea Tbk (PTRO)** stock using **Long Short-Term Memory (LSTM)**.

## Overview

This project was developed to support stock risk analysis by modeling historical volatility patterns of PTRO.  
The application includes:

- Stock data exploration
- Return and volatility analysis
- Volatility distribution visualization
- LSTM modeling
- Model evaluation
- Volatility forecasting
- Risk classification

## Objectives

The main objective of this project is to build a machine learning model that can predict **next-day volatility** of PTRO stock based on historical market behavior.

## Why PTRO?

PTRO was selected because it is a mining-sector stock with relatively dynamic price movements and volatility characteristics that make it suitable for time series risk analysis.

## Dataset

Data is collected from **Yahoo Finance** using the ticker:

- `PTRO.JK`

Data period used in the project:

- **2015 to present**

Features used:

- `Close`
- `Volume`
- `Log Return`
- `21-day Rolling Volatility`

Target variable:

- `Y` = volatility at **t+1**

## Methodology

The workflow in this project is:

1. Data collection from Yahoo Finance
2. Exploratory Data Analysis (EDA)
3. Feature engineering
4. Dataset splitting
5. Scaling
6. Sequence creation for LSTM
7. Model training
8. Evaluation
9. Forecasting
10. Risk classification

## Model Architecture

The model uses:

- **LSTM layer**
- **Dropout layer**
- **Dense output layer**

Main configuration:

- Window size: **21 days**
- Loss function: **MSE**
- Optimizer: **Adam**

## Evaluation Metrics

The model is evaluated using:

- **MAE**
- **RMSE**

Example results from the latest experiment:

- **MAE**: 0.0481
- **RMSE**: 0.0707

## Web Application Features

### 1. Home
- Project overview
- PTRO profile
- Analysis workflow
- Application navigation

### 2. Stock Data & EDA
- Historical stock table
- Closing price and moving average chart
- Log return visualization
- Volatility distribution
- Correlation matrix
- Dataset split information

### 3. LSTM Modeling
- Sequence configuration
- Model architecture
- Training vs validation loss
- Actual vs predicted volatility
- Evaluation metrics

### 4. Forecasting
- Latest volatility
- Predicted volatility for t+1
- Risk classification

## Risk Classification

Volatility risk is grouped based on distribution thresholds:

- **Low Risk**
- **Medium Risk**
- **High Risk**
- **Extreme Risk**

## Project Structure

```text
PTRO-Volatility-Forecasting/
├── Home.py
├── pages/
│   ├── 1_Stock_Data.py
│   ├── 2_EDA.py
│   ├── 3_LSTM.py
│   └── 4_Forecasting.py
├── CSV/
│   ├── data_train_scaled.csv
│   ├── data_val_scaled.csv
│   └── data_test_scaled.csv
├── Stat/
│   ├── time_series_plot.png
│   ├── training_validation_loss.png
│   └── actual_vs_predicted.png
├── PTRO_LSTM_Volatility.h5
├── PTRO_scaler_X.pkl
├── PTRO_scaler_Y.pkl
├── logo.png
├── VINIX7.png
├── requirements.txt
└── README.md
