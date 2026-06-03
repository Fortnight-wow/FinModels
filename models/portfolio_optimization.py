from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


def _as_array(values):
    return np.asarray(values, dtype=float)


@dataclass(frozen=True)
class PortfolioResult:
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float


class MarkowitzPortfolioOptimizer:
    """Mean-variance portfolio optimization using Markowitz ideas."""

    def __init__(self, expected_returns, covariance_matrix, risk_free_rate: float = 0.0):
        self.expected_returns = _as_array(expected_returns)
        self.covariance_matrix = _as_array(covariance_matrix)
        self.risk_free_rate = risk_free_rate

        if self.expected_returns.ndim != 1:
            raise ValueError("expected_returns must be one-dimensional.")
        if self.covariance_matrix.shape != (
            len(self.expected_returns),
            len(self.expected_returns),
        ):
            raise ValueError("covariance_matrix shape must match expected_returns.")

    @property
    def asset_count(self) -> int:
        return len(self.expected_returns)

    def portfolio_return(self, weights) -> float:
        weights = _as_array(weights)
        return float(weights @ self.expected_returns)

    def portfolio_volatility(self, weights) -> float:
        weights = _as_array(weights)
        variance = weights @ self.covariance_matrix @ weights
        return float(np.sqrt(max(variance, 0.0)))

    def sharpe_ratio(self, weights) -> float:
        volatility = self.portfolio_volatility(weights)
        if volatility == 0:
            return 0.0

        return (self.portfolio_return(weights) - self.risk_free_rate) / volatility

    def _bounds(self, allow_short: bool):
        if allow_short:
            return [(-1.0, 1.0)] * self.asset_count
        return [(0.0, 1.0)] * self.asset_count

    def _result(self, weights) -> PortfolioResult:
        return PortfolioResult(
            weights=np.asarray(weights, dtype=float),
            expected_return=self.portfolio_return(weights),
            volatility=self.portfolio_volatility(weights),
            sharpe_ratio=self.sharpe_ratio(weights),
        )

    def minimum_variance(self, allow_short: bool = False) -> PortfolioResult:
        initial_weights = np.full(self.asset_count, 1 / self.asset_count)
        constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}

        result = minimize(
            lambda weights: self.portfolio_volatility(weights),
            initial_weights,
            method="SLSQP",
            bounds=self._bounds(allow_short),
            constraints=constraints,
        )

        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")

        return self._result(result.x)

    def maximum_sharpe(self, allow_short: bool = False) -> PortfolioResult:
        initial_weights = np.full(self.asset_count, 1 / self.asset_count)
        constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}

        result = minimize(
            lambda weights: -self.sharpe_ratio(weights),
            initial_weights,
            method="SLSQP",
            bounds=self._bounds(allow_short),
            constraints=constraints,
        )

        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")

        return self._result(result.x)

    def efficient_frontier(self, points: int = 25, allow_short: bool = False) -> list[PortfolioResult]:
        if points <= 1:
            raise ValueError("points must be greater than 1.")

        target_returns = np.linspace(self.expected_returns.min(), self.expected_returns.max(), points)
        frontier = []
        initial_weights = np.full(self.asset_count, 1 / self.asset_count)

        for target_return in target_returns:
            constraints = (
                {"type": "eq", "fun": lambda weights: np.sum(weights) - 1},
                {
                    "type": "eq",
                    "fun": lambda weights, target=target_return: self.portfolio_return(weights) - target,
                },
            )
            result = minimize(
                lambda weights: self.portfolio_volatility(weights),
                initial_weights,
                method="SLSQP",
                bounds=self._bounds(allow_short),
                constraints=constraints,
            )

            if result.success:
                frontier.append(self._result(result.x))

        return frontier
