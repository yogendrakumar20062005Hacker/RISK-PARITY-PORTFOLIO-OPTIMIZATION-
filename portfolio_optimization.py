"""Download market data and calculate long-only risk-parity portfolio weights."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import OptimizeResult, minimize
from tqdm import tqdm

DEFAULT_TICKERS = ("AAPL", "TSLA", "AMZN", "GOOGL", "META")
DEFAULT_START_DATE = "2020-01-01"


def download_returns(
    tickers: Sequence[str], start_date: str = DEFAULT_START_DATE
) -> pd.DataFrame:
    """Download adjusted closing prices and return a complete daily-return table."""
    if not tickers:
        raise ValueError("At least one ticker is required.")

    returns: dict[str, pd.Series] = {}
    for ticker in tqdm(tickers, desc="Downloading prices"):
        prices = yf.download(
            ticker,
            start=start_date,
            auto_adjust=True,
            progress=False,
        )
        if prices.empty or "Close" not in prices:
            raise ValueError(
                f"Yahoo Finance returned no closing prices for {ticker!r}."
            )

        close = prices["Close"]
        if isinstance(close, pd.DataFrame):
            close = close[ticker] if ticker in close.columns else close.iloc[:, 0]
        returns[ticker] = close.pct_change()

    return pd.DataFrame(returns).dropna()


def portfolio_volatility(weights: np.ndarray, covariance: np.ndarray) -> float:
    """Return the portfolio's daily volatility."""
    return float(np.sqrt(weights.T @ covariance @ weights))


def risk_contributions(weights: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Return each asset's contribution to portfolio volatility."""
    volatility = portfolio_volatility(weights, covariance)
    if volatility == 0:
        raise ValueError("Portfolio volatility is zero; risk contributions are undefined.")
    marginal_contributions = (covariance @ weights) / volatility
    return weights * marginal_contributions


def risk_parity_objective(weights: np.ndarray, covariance: np.ndarray) -> float:
    """Minimize squared pairwise differences in asset risk contributions."""
    contributions = risk_contributions(weights, covariance)
    contribution_shares = contributions / np.sum(contributions)
    differences = contribution_shares[:, None] - contribution_shares[None, :]
    return float(np.sum(differences**2))


def optimize_risk_parity(returns: pd.DataFrame) -> OptimizeResult:
    """Optimize long-only weights to equalize contributions to portfolio risk."""
    if returns.empty or returns.shape[1] == 0:
        raise ValueError("Returns must contain at least one asset and one observation.")
    if returns.shape[0] < 2:
        raise ValueError("At least two return observations are required.")

    covariance = returns.cov().to_numpy(dtype=float)
    if not np.isfinite(covariance).all():
        raise ValueError("The return covariance matrix contains non-finite values.")

    asset_count = returns.shape[1]
    initial_weights = np.full(asset_count, 1 / asset_count)
    result = minimize(
        risk_parity_objective,
        initial_weights,
        args=(covariance,),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * asset_count,
        constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1},
    )
    if not result.success:
        raise RuntimeError(f"Portfolio optimization failed: {result.message}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate long-only risk-parity portfolio weights."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="Yahoo Finance ticker symbols (default: %(default)s)",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help="First date to download in YYYY-MM-DD format (default: %(default)s)",
    )
    args = parser.parse_args()

    returns = download_returns(args.tickers, args.start_date)
    result = optimize_risk_parity(returns)
    weights = pd.Series(result.x, index=returns.columns, name="weight")

    print(f"Used {len(returns)} daily return observations.")
    print("Optimal portfolio weights:")
    print(weights.to_string(float_format=lambda weight: f"{weight:.4%}"))
    print(f"Objective function value: {result.fun:.8g}")


if __name__ == "__main__":
    main()
