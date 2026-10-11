# Portfolio Optimization

This project calculates a long-only portfolio that equalizes each asset's
contribution to portfolio risk. It downloads adjusted daily prices from Yahoo
Finance, calculates returns and their covariance matrix, and minimizes the
squared differences between asset risk contributions.

## Setup

Use Python 3 and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the script

The default run uses AAPL, TSLA, AMZN, GOOGL, and META prices from 2020 onward:

```bash
python portfolio_optimization.py
```

Choose other tickers or a different start date with:

```bash
python portfolio_optimization.py --tickers AAPL MSFT NVDA --start-date 2022-01-01
```

The script requires an internet connection to retrieve market data from Yahoo
Finance. It prints the number of return observations and the optimized weights.

## Run the notebook

Open `Risk_Parity_Portfolio.ipynb` in VS Code or Jupyter, select the Python
kernel from `.venv`, and run the cells. The notebook uses the same download and
optimization functions as the script.

## Files

- `portfolio_optimization.py`: command-line application and reusable functions.
- `Risk_Parity_Portfolio.ipynb`: notebook walkthrough.
- `requirements.txt`: Python dependencies for both run options.
