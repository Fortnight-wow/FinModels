from collections.abc import Callable

import numpy as np


class HullWhiteModel:
    """One-factor Hull-White short-rate model.

    The drift term can be a constant or a function of time. This keeps the
    implementation approachable while leaving room for curve calibration later.
    """

    def __init__(
        self,
        initial_rate: float,
        mean_reversion: float,
        volatility: float,
        theta: float | Callable[[float], float],
    ):
        if mean_reversion <= 0:
            raise ValueError("mean_reversion must be positive.")
        if volatility < 0:
            raise ValueError("volatility cannot be negative.")

        self.initial_rate = initial_rate
        self.mean_reversion = mean_reversion
        self.volatility = volatility
        self.theta = theta

    def _theta(self, time: float) -> float:
        if callable(self.theta):
            return float(self.theta(time))
        return float(self.theta)

    def simulate(self, maturity: float = 1.0, steps: int = 252, paths: int = 1000, seed=None):
        rng = np.random.default_rng(seed)
        dt = maturity / steps

        rates = np.empty((paths, steps + 1))
        rates[:, 0] = self.initial_rate

        for step in range(1, steps + 1):
            time = (step - 1) * dt
            previous = rates[:, step - 1]
            rates[:, step] = (
                previous
                + (self._theta(time) - self.mean_reversion * previous) * dt
                + self.volatility * np.sqrt(dt) * rng.normal(size=paths)
            )

        return rates

    @staticmethod
    def discount_factor_from_path(rate_path, maturity: float) -> float:
        rate_path = np.asarray(rate_path)
        if rate_path.ndim != 1:
            raise ValueError("rate_path must be one-dimensional.")

        dt = maturity / (len(rate_path) - 1)
        integral = np.trapezoid(rate_path, dx=dt)
        return float(np.exp(-integral))
