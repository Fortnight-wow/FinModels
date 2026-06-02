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
        if seed is not None:
            np.random.seed(seed)

        dt = maturity / steps

        prices = np.empty(steps + 1)
        prices[0] = self.initial_price

        for i in range(1, steps + 1):
            z = np.random.normal()

            prices[i] = prices[i - 1] * np.exp(
                (self.drift - 0.5 * self.volatility**2) * dt
                + self.volatility * np.sqrt(dt) * z
            )

        return prices
