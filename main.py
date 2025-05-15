#!/usr/bin/env python3
"""
Portfolio Optimization with Modern Portfolio Theory (MPT)

Features:
 - Single-call data download via yfinance (threads=False, auto_adjust=True)
 - Monte Carlo simulation (10,000 portfolios) for the Efficient Frontier
 - Identification of:
     • Maximum Sharpe Ratio portfolio
     • Minimum Volatility portfolio
 - Stress-test analysis over a custom date range
 - Risk-profile selection (conservative, moderate, aggressive)
 - All visualizations are titled, labeled, and saved as PNGs
"""

import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ------------------------- Main Functions --------------------------

def download_data(tickers, start, end):
    """
    Download adjusted close prices for all tickers in one call,
    with threading disabled to avoid SQLite locks.
    Returns a DataFrame of adjusted-close prices (columns=ticker symbols).
    """
    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,  # fetch fully-adjusted prices directly
        threads=False      # disable threading to prevent cache locks
    )
    # With auto_adjust=True, yfinance returns a DataFrame with columns = tickers
    df.dropna(how='any', inplace=True)
    return df

def compute_statistics(prices):
    """
    From daily price data, compute:
      - daily returns
      - annualized mean returns
      - annualized covariance matrix
    Assumes 252 trading days per year.
    """
    returns = prices.pct_change().dropna()
    mean_daily = returns.mean()
    cov_daily = returns.cov()
    annual_factor = 252
    return returns, mean_daily * annual_factor, cov_daily * annual_factor

def simulate_portfolios(mean_ret, cov_mat, n_portfolios, rf_rate):
    """
    Monte Carlo simulation of random portfolios.
    Returns:
      - results: array of shape (3, n_portfolios) with [return, vol, sharpe]
      - weight_list: list of weight arrays
    """
    n_assets = len(mean_ret)
    results = np.zeros((3, n_portfolios))
    weight_list = []
    np.random.seed(42)

    for i in range(n_portfolios):
        w = np.random.random(n_assets)
        w /= w.sum()
        weight_list.append(w)

        port_ret = np.dot(w, mean_ret)
        port_vol = np.sqrt(w @ cov_mat @ w)
        port_sharpe = (port_ret - rf_rate) / port_vol

        results[:, i] = [port_ret, port_vol, port_sharpe]

    return results, weight_list

def find_optimal(results, weight_list):
    """
    Identify the max Sharpe and min volatility portfolios.
    Returns a dict with keys 'max_sharpe' and 'min_vol'.
    """
    idx_sharpe = np.argmax(results[2])
    idx_vol = np.argmin(results[1])
    return {
        'max_sharpe': {
            'idx': idx_sharpe,
            'return': results[0, idx_sharpe],
            'vol': results[1, idx_sharpe],
            'weights': weight_list[idx_sharpe]
        },
        'min_vol': {
            'idx': idx_vol,
            'return': results[0, idx_vol],
            'vol': results[1, idx_vol],
            'weights': weight_list[idx_vol]
        }
    }

def plot_efficient_frontier(results, optimal, asset_names, out_prefix):
    """
    Scatter-plot of all simulated portfolios (vol vs return, colored by Sharpe),
    with the optimal portfolios marked. Saves figure to file.
    """
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(
        results[1], results[0], c=results[2],
        cmap='viridis', alpha=0.7, edgecolor='k', linewidth=0.2
    )
    plt.colorbar(sc, label='Sharpe Ratio')
    plt.scatter(
        optimal['max_sharpe']['vol'], optimal['max_sharpe']['return'],
        color='red', marker='*', s=200, label='Max Sharpe'
    )
    plt.scatter(
        optimal['min_vol']['vol'], optimal['min_vol']['return'],
        color='blue', marker='*', s=200, label='Min Volatility'
    )
    plt.title('Efficient Frontier: Annualized Risk vs Return')
    plt.xlabel('Annualized Volatility')
    plt.ylabel('Annualized Return')
    plt.legend()
    plt.grid(True)
    fname = f"{out_prefix}_efficient_frontier.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved efficient frontier plot to {fname}")

def stress_test(returns, optimal, period, out_prefix):
    """
    For each optimal portfolio, plot cumulative returns over the given period.
    'period' should be a tuple (start_date, end_date) as strings YYYY-MM-DD.
    """
    start, end = period
    subset = returns.loc[start:end]
    for name, info in optimal.items():
        weights = info['weights']
        port_daily = subset.dot(weights)
        cum_ret = (1 + port_daily).cumprod() - 1

        plt.figure(figsize=(8, 5))
        cum_ret.plot()
        plt.title(f"Cumulative Return ({name.replace('_',' ').title()})\n{start} to {end}")
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.grid(True)
        fname = f"{out_prefix}_{name}_stress.png"
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"Saved stress-test plot for {name} to {fname}")

def display_weights(optimal, asset_names):
    """
    Print each portfolio's weights and key metrics in a readable format.
    """
    for label, info in optimal.items():
        title = label.replace('_', ' ').title()
        print(f"\n{title} Portfolio")
        for name, w in zip(asset_names, info['weights']):
            print(f"  {name}: {w:.2%}")
        print(f"  Expected Return: {info['return']:.2%}")
        print(f"  Volatility     : {info['vol']:.2%}")

# --------------------------- Main Routine ---------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="MPT Portfolio Optimization with Stress Testing"
    )
    parser.add_argument(
        '--start', default='2020-01-01',
        help='Data start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end', default='2024-12-31',
        help='Data end date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--risk_free', type=float, default=0.02,
        help='Annual risk-free rate (e.g., 0.02 for 2%%)'
    )
    parser.add_argument(
        '--stress', nargs=2, metavar=('START','END'),
        help='Stress-test period (two dates YYYY-MM-DD)'
    )
    parser.add_argument(
        '--profile',
        choices=['conservative', 'moderate', 'aggressive'],
        default='moderate',
        help='Select portfolio by risk profile'
    )
    parser.add_argument(
        '--out', default='output',
        help='Prefix for output filenames'
    )
    args = parser.parse_args()

    # 1) Download data
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
    prices = download_data(tickers, args.start, args.end)
    print(f"Downloaded price data for {len(prices.columns)} assets: {prices.columns.tolist()}")

    # 2) Compute returns & statistics
    returns, mean_annual, cov_annual = compute_statistics(prices)

    # 3) Simulate portfolios
    results, weight_list = simulate_portfolios(
        mean_ret=mean_annual.values,
        cov_mat=cov_annual.values,
        n_portfolios=10000,
        rf_rate=args.risk_free
    )

    # 4) Identify optimal portfolios
    optimal = find_optimal(results, weight_list)

    # 5) Display portfolio weights & metrics
    display_weights(optimal, prices.columns)

    # 6) Plot and save Efficient Frontier
    plot_efficient_frontier(results, optimal, prices.columns, args.out)

    # 7) Stress-test if requested
    if args.stress:
        stress_test(returns, optimal, tuple(args.stress), args.out)

    # 8) Select portfolio based on risk profile
    choice_map = {
        'conservative': 'min_vol',
        'moderate'    : 'max_sharpe',
        'aggressive'  : 'max_sharpe'
    }
    chosen = choice_map[args.profile]
    print(f"\nRisk Profile '{args.profile.title()}' selects the '{chosen.replace('_',' ').title()}' portfolio.")
