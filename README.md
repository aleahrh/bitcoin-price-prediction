# Bitcoin Price Prediction using RNN

An RNN-based regression model that predicts Bitcoin closing prices from historical market data using PyTorch.

## Overview

Using historical Open, High, and Low prices as input features, this model predicts the Bitcoin closing price. The model achieved an **R² score of 0.9978** on a held-out test set, indicating near-perfect predictive accuracy.

## Tech Stack

- Python
- PyTorch
- scikit-learn
- Pandas
- NumPy

## Model Architecture

- RNN with 2 layers and hidden size of 128
- Fully connected output layer
- Adam optimizer with MSE loss
- 80/20 train/test split with StandardScaler normalization

## Results

| Metric | Value |
|--------|-------|
| R² Score | 0.9978 |

## How to Run

```bash
pip install torch scikit-learn pandas numpy
python rnn.py
```

## Author

Aleah Hassabo — [LinkedIn](https://www.linkedin.com/in/aleah-hassabo)
