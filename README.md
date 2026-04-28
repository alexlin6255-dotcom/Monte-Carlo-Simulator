# Monte Carlo Portfolio Simulator

Simulates 100 possible futures for a portfolio of 6 ASX stocks using real market data.

## What it does
- Fetches real historical data via yfinance
- Models correlated stock movements using Cholesky decomposition
- Runs 100 Monte Carlo simulations over 100 days
- Visualises all possible portfolio outcomes

## Technologies
- Python, NumPy, yfinance, matplotlib

## How to run
pip install numpy yfinance matplotlib
python "monte carlo simulation.py"
