import numpy as np


class GeometricBrownianMotion:
    """Simulate stock-price paths using Geometric Brownian Motion."""

    def __init__(self, initial_price: float, drift: float, volatility: float):
        if initial_price <= 0:
            raise ValueError('Initial price must be positive.')

        self.initial_price = initial_price
        self.drift = drift
        self.volatility = volatility

    def simulate(self, maturity: float = 1.0, steps: int = 252, seed=None):
        rng = np.random.default_rng(seed)
        dt = maturity / steps

        prices = np.empty(steps + 1)
        prices[0] = self.initial_price

        for i in range(1, steps + 1):
            z = rng.normal()

            prices[i] = prices[i - 1] * np.exp(
                (self.drift - 0.5 * self.volatility**2) * dt
                + self.volatility * np.sqrt(dt) * z
            )

        return prices

    def simulate_paths(self, maturity: float = 1.0, steps: int = 252, paths: int = 1000, seed=None):
        """Return a matrix of simulated paths with shape (paths, steps + 1)."""
        if paths <= 0:
            raise ValueError('Number of paths must be positive.')

        rng = np.random.default_rng(seed)
        dt = maturity / steps

        shocks = rng.normal(size=(paths, steps))
        increments = (
            (self.drift - 0.5 * self.volatility**2) * dt
            + self.volatility * np.sqrt(dt) * shocks
        )

        log_paths = np.cumsum(increments, axis=1)
        log_paths = np.column_stack([np.zeros(paths), log_paths])

        return self.initial_price * np.exp(log_paths)
