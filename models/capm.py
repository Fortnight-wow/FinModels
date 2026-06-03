from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CAPMResult:
    beta: float
    alpha: float
    expected_return: float


def estimate_beta(asset_returns, market_returns) -> float:
    asset_returns = np.asarray(asset_returns, dtype=float)
    market_returns = np.asarray(market_returns, dtype=float)

    if asset_returns.shape != market_returns.shape:
        raise ValueError("asset_returns and market_returns must have the same shape.")
    if asset_returns.ndim != 1:
        raise ValueError("returns must be one-dimensional.")

    market_variance = np.var(market_returns, ddof=1)
    if market_variance == 0:
        raise ValueError("market_returns variance cannot be zero.")

    covariance = np.cov(asset_returns, market_returns, ddof=1)[0, 1]
    return float(covariance / market_variance)


def expected_return(risk_free_rate: float, beta: float, market_expected_return: float) -> float:
    return float(risk_free_rate + beta * (market_expected_return - risk_free_rate))


def analyze_capm(asset_returns, market_returns, risk_free_rate: float) -> CAPMResult:
    asset_returns = np.asarray(asset_returns, dtype=float)
    market_returns = np.asarray(market_returns, dtype=float)

    beta = estimate_beta(asset_returns, market_returns)
    market_mean = float(np.mean(market_returns))
    capm_expected = expected_return(risk_free_rate, beta, market_mean)
    alpha = float(np.mean(asset_returns) - capm_expected)

    return CAPMResult(
        beta=beta,
        alpha=alpha,
        expected_return=capm_expected,
    )
