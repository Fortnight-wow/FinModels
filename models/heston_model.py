import numpy as np


class HestonModel:
    """Heston stochastic volatility model using full-truncation Euler steps."""

    def __init__(
        self,
        initial_price: float,
        initial_variance: float,
        risk_free_rate: float,
        kappa: float,
        theta: float,
        volatility_of_variance: float,
        rho: float,
    ):
        if initial_price <= 0:
            raise ValueError("Initial price must be positive.")
        if initial_variance < 0:
            raise ValueError("Initial variance cannot be negative.")
        if kappa <= 0:
            raise ValueError("kappa must be positive.")
        if theta <= 0:
            raise ValueError("theta must be positive.")
        if volatility_of_variance < 0:
            raise ValueError("volatility_of_variance cannot be negative.")
        if not -1 < rho < 1:
            raise ValueError("rho must be between -1 and 1.")

        self.initial_price = initial_price
        self.initial_variance = initial_variance
        self.risk_free_rate = risk_free_rate
        self.kappa = kappa
        self.theta = theta
        self.volatility_of_variance = volatility_of_variance
        self.rho = rho

    def simulate(self, maturity: float = 1.0, steps: int = 252, paths: int = 1000, seed=None):
        rng = np.random.default_rng(seed)
        dt = maturity / steps

        prices = np.empty((paths, steps + 1))
        variances = np.empty((paths, steps + 1))
        prices[:, 0] = self.initial_price
        variances[:, 0] = self.initial_variance

        for step in range(1, steps + 1):
            z_price = rng.normal(size=paths)
            z_independent = rng.normal(size=paths)
            z_variance = self.rho * z_price + np.sqrt(1 - self.rho**2) * z_independent

            variance_previous = np.maximum(variances[:, step - 1], 0.0)
            variance_sqrt = np.sqrt(variance_previous)

            variances[:, step] = np.maximum(
                variance_previous
                + self.kappa * (self.theta - variance_previous) * dt
                + self.volatility_of_variance * variance_sqrt * np.sqrt(dt) * z_variance,
                0.0,
            )

            prices[:, step] = prices[:, step - 1] * np.exp(
                (self.risk_free_rate - 0.5 * variance_previous) * dt
                + variance_sqrt * np.sqrt(dt) * z_price
            )

        return prices, variances

    def monte_carlo_call_price(
        self,
        strike: float,
        maturity: float = 1.0,
        steps: int = 252,
        paths: int = 10000,
        seed=None,
    ) -> float:
        prices, _ = self.simulate(maturity=maturity, steps=steps, paths=paths, seed=seed)
        payoff = np.maximum(prices[:, -1] - strike, 0.0)
        return float(np.exp(-self.risk_free_rate * maturity) * payoff.mean())
