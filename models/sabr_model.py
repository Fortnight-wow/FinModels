from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SABRModel:
    """SABR volatility model with Hagan's implied-volatility approximation."""

    forward: float
    alpha: float
    beta: float
    rho: float
    nu: float

    def __post_init__(self):
        if self.forward <= 0:
            raise ValueError("forward must be positive.")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive.")
        if not 0 <= self.beta <= 1:
            raise ValueError("beta must be between 0 and 1.")
        if not -1 < self.rho < 1:
            raise ValueError("rho must be between -1 and 1.")
        if self.nu <= 0:
            raise ValueError("nu must be positive.")

    def implied_volatility(self, strike: float, maturity: float) -> float:
        if strike <= 0:
            raise ValueError("strike must be positive.")
        if maturity <= 0:
            raise ValueError("maturity must be positive.")

        f = self.forward
        k = strike
        alpha = self.alpha
        beta = self.beta
        rho = self.rho
        nu = self.nu
        one_minus_beta = 1 - beta

        if abs(f - k) < 1e-12:
            f_factor = f ** one_minus_beta
            correction = (
                (one_minus_beta**2 / 24) * (alpha**2 / f_factor**2)
                + (rho * beta * nu * alpha) / (4 * f_factor)
                + ((2 - 3 * rho**2) * nu**2 / 24)
            ) * maturity
            return float((alpha / f_factor) * (1 + correction))

        log_fk = np.log(f / k)
        fk_beta = (f * k) ** (one_minus_beta / 2)
        denominator_correction = (
            1
            + (one_minus_beta**2 / 24) * log_fk**2
            + (one_minus_beta**4 / 1920) * log_fk**4
        )

        z = (nu / alpha) * fk_beta * log_fk
        x_z = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))

        if abs(x_z) < 1e-12:
            z_over_x_z = 1.0
        else:
            z_over_x_z = z / x_z

        time_correction = (
            (one_minus_beta**2 / 24) * (alpha**2 / fk_beta**2)
            + (rho * beta * nu * alpha) / (4 * fk_beta)
            + ((2 - 3 * rho**2) * nu**2 / 24)
        ) * maturity

        volatility = (
            alpha
            / (fk_beta * denominator_correction)
            * z_over_x_z
            * (1 + time_correction)
        )
        return float(volatility)

    def simulate(self, maturity: float = 1.0, steps: int = 252, paths: int = 1000, seed=None):
        rng = np.random.default_rng(seed)
        dt = maturity / steps

        forwards = np.empty((paths, steps + 1))
        volatilities = np.empty((paths, steps + 1))
        forwards[:, 0] = self.forward
        volatilities[:, 0] = self.alpha

        for step in range(1, steps + 1):
            z_forward = rng.normal(size=paths)
            z_independent = rng.normal(size=paths)
            z_volatility = self.rho * z_forward + np.sqrt(1 - self.rho**2) * z_independent

            previous_forward = np.maximum(forwards[:, step - 1], 1e-12)
            previous_volatility = np.maximum(volatilities[:, step - 1], 1e-12)

            forwards[:, step] = np.maximum(
                previous_forward
                + previous_volatility * previous_forward**self.beta * np.sqrt(dt) * z_forward,
                1e-12,
            )
            volatilities[:, step] = np.maximum(
                previous_volatility
                + self.nu * previous_volatility * np.sqrt(dt) * z_volatility,
                1e-12,
            )

        return forwards, volatilities
