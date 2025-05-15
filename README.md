# Risk-Return Optimizer

This project performs **portfolio optimization** using historical stock data. It computes the **Efficient Frontier**, identifies the **Maximum Sharpe Ratio** and **Minimum Volatility** portfolios, and visualizes results with an interactive plot. Ideal for understanding modern portfolio theory in practice.

## 🚀 Features

- Downloads stock price data via Yahoo Finance
- Calculates annualized returns, volatility, and Sharpe Ratio
- Simulates thousands of random portfolios
- Plots the Efficient Frontier
- Highlights optimal portfolios for different risk preferences
- Saves plots and portfolio data

## 📊 Visualization

- Efficient Frontier plotted with Sharpe Ratios
- Clear visual markers for:
  - ✅ Max Sharpe portfolio
  - 🔵 Min Volatility portfolio
- Color bar shows Sharpe Ratios

## 🧠 Analysis Included

- Evaluation of return realism
- Stress-testing assumptions
- Portfolio fit for conservative vs aggressive risk profiles

## 📁 Output

- Efficient frontier plot (`output_efficient_frontier.png`)
- Console output of portfolio compositions and metrics

## 📦 Requirements

- Python 3.9+
- `yfinance`
- `numpy`
- `pandas`
- `matplotlib`

### ✅ Install Dependencies

Use the provided `environment.yml` file to set up the environment:

```bash
conda env create -f environment.yml
conda activate stock-prediction-env
```
