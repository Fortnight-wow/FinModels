import numpy as np


class VasicekModel:
    """Mean-reverting short-rate model with normally distributed rates."""

    def __init__(self, initial_rate: float, mean_reversion: float, long_term_mean: float, volatility: float):
        if mean_reversion <= 0:
            raise ValueError("mean_reversion must be positive.")
        if volatility < 0:
            raise ValueError("volatility cannot be negative.")

        self.initial_rate = initial_rate
        self.mean_reversion = mean_reversion
        self.long_term_mean = long_term_mean
        self.volatility = volatility

    def simulate(self, maturity: float = 1.0, steps: int = 252, paths: int = 1000, seed=None):
        rng = np.random.default_rng(seed)
        dt = maturity / steps

        rates = np.empty((paths, steps + 1))
        rates[:, 0] = self.initial_rate

        for step in range(1, steps + 1):
            previous = rates[:, step - 1]
            rates[:, step] = (
                previous
                + self.mean_reversion * (self.long_term_mean - previous) * dt
                + self.volatility * np.sqrt(dt) * rng.normal(size=paths)
            )

        return rates

    def zero_coupon_bond_price(self, maturity: float, current_rate: float | None = None) -> float:
        if maturity < 0:
            raise ValueError("maturity cannot be negative.")

        rate = self.initial_rate if current_rate is None else current_rate
        a = self.mean_reversion
        b = self.long_term_mean
        sigma = self.volatility

        if maturity == 0:
            return 1.0

        big_b = (1 - np.exp(-a * maturity)) / a
        big_a = np.exp(
            (b - sigma**2 / (2 * a**2)) * (big_b - maturity)
            - sigma**2 * big_b**2 / (4 * a)
        )

        return float(big_a * np.exp(-big_b * rate))
