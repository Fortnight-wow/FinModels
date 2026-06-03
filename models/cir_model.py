import numpy as np


class CIRModel:
    """Cox-Ingersoll-Ross model for positive mean-reverting short rates."""

    def __init__(self, initial_rate: float, mean_reversion: float, long_term_mean: float, volatility: float):
        if initial_rate < 0:
            raise ValueError("initial_rate cannot be negative.")
        if mean_reversion <= 0:
            raise ValueError("mean_reversion must be positive.")
        if long_term_mean <= 0:
            raise ValueError("long_term_mean must be positive.")
        if volatility <= 0:
            raise ValueError("volatility must be positive.")

        self.initial_rate = initial_rate
        self.mean_reversion = mean_reversion
        self.long_term_mean = long_term_mean
        self.volatility = volatility

    @property
    def feller_ratio(self) -> float:
        return 2 * self.mean_reversion * self.long_term_mean / self.volatility**2

    def simulate(self, maturity: float = 1.0, steps: int = 252, paths: int = 1000, seed=None):
        rng = np.random.default_rng(seed)
        dt = maturity / steps

        rates = np.empty((paths, steps + 1))
        rates[:, 0] = self.initial_rate

        for step in range(1, steps + 1):
            previous = np.maximum(rates[:, step - 1], 0.0)
            rates[:, step] = np.maximum(
                previous
                + self.mean_reversion * (self.long_term_mean - previous) * dt
                + self.volatility * np.sqrt(previous) * np.sqrt(dt) * rng.normal(size=paths),
                0.0,
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

        gamma = np.sqrt(a**2 + 2 * sigma**2)
        exp_gamma_t = np.exp(gamma * maturity)
        denominator = (gamma + a) * (exp_gamma_t - 1) + 2 * gamma

        big_b = 2 * (exp_gamma_t - 1) / denominator
        big_a = (
            2 * gamma * np.exp((a + gamma) * maturity / 2) / denominator
        ) ** (2 * a * b / sigma**2)

        return float(big_a * np.exp(-big_b * rate))
