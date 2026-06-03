import numpy as np
from scipy.stats import norm


def _returns_array(returns):
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1:
        raise ValueError("returns must be one-dimensional.")
    if len(values) == 0:
        raise ValueError("returns cannot be empty.")
    return values


def historical_var(returns, confidence_level: float = 0.95) -> float:
    """Historical Value at Risk as a positive loss number."""
    returns = _returns_array(returns)
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    percentile = 100 * (1 - confidence_level)
    return float(-np.percentile(returns, percentile))


def historical_cvar(returns, confidence_level: float = 0.95) -> float:
    """Conditional Value at Risk as the average loss beyond VaR."""
    returns = _returns_array(returns)
    var_threshold = -historical_var(returns, confidence_level)
    tail_returns = returns[returns <= var_threshold]

    if len(tail_returns) == 0:
        return 0.0

    return float(-tail_returns.mean())


def parametric_var(
    mean_return: float,
    volatility: float,
    confidence_level: float = 0.95,
) -> float:
    if volatility < 0:
        raise ValueError("volatility cannot be negative.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    quantile = norm.ppf(1 - confidence_level)
    return float(-(mean_return + volatility * quantile))


def monte_carlo_var(
    mean_return: float,
    volatility: float,
    confidence_level: float = 0.95,
    simulations: int = 10000,
    seed=None,
) -> float:
    if simulations <= 0:
        raise ValueError("simulations must be positive.")

    rng = np.random.default_rng(seed)
    simulated_returns = rng.normal(mean_return, volatility, simulations)
    return historical_var(simulated_returns, confidence_level)


def portfolio_returns(asset_returns, weights):
    asset_returns = np.asarray(asset_returns, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if asset_returns.ndim != 2:
        raise ValueError("asset_returns must be a two-dimensional matrix.")
    if asset_returns.shape[1] != len(weights):
        raise ValueError("weights length must match the number of assets.")

    return asset_returns @ weights


def stress_test_portfolio(base_value: float, weights, shock_returns) -> dict[str, float]:
    if base_value <= 0:
        raise ValueError("base_value must be positive.")

    weights = np.asarray(weights, dtype=float)
    shock_returns = np.asarray(shock_returns, dtype=float)

    if weights.shape != shock_returns.shape:
        raise ValueError("weights and shock_returns must have the same shape.")

    stressed_return = float(weights @ shock_returns)
    stressed_value = base_value * (1 + stressed_return)

    return {
        "stressed_return": stressed_return,
        "stressed_value": float(stressed_value),
        "loss": float(base_value - stressed_value),
    }
