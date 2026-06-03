from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


@dataclass(frozen=True)
class BlackScholesOption:
    """European option pricing under the Black-Scholes model."""

    spot: float
    strike: float
    maturity: float
    risk_free_rate: float
    volatility: float
    option_type: str = "call"
    dividend_yield: float = 0.0

    def __post_init__(self):
        if self.spot <= 0:
            raise ValueError("Spot price must be positive.")
        if self.strike <= 0:
            raise ValueError("Strike price must be positive.")
        if self.maturity <= 0:
            raise ValueError("Maturity must be positive.")
        if self.volatility <= 0:
            raise ValueError("Volatility must be positive.")
        if self.option_type not in {"call", "put"}:
            raise ValueError("option_type must be either 'call' or 'put'.")

    @property
    def d1(self) -> float:
        numerator = np.log(self.spot / self.strike)
        numerator += (
            self.risk_free_rate
            - self.dividend_yield
            + 0.5 * self.volatility**2
        ) * self.maturity

        return numerator / (self.volatility * np.sqrt(self.maturity))

    @property
    def d2(self) -> float:
        return self.d1 - self.volatility * np.sqrt(self.maturity)

    def price(self) -> float:
        discount_rate = np.exp(-self.risk_free_rate * self.maturity)
        discount_dividend = np.exp(-self.dividend_yield * self.maturity)

        if self.option_type == "call":
            return (
                self.spot * discount_dividend * norm.cdf(self.d1)
                - self.strike * discount_rate * norm.cdf(self.d2)
            )

        return (
            self.strike * discount_rate * norm.cdf(-self.d2)
            - self.spot * discount_dividend * norm.cdf(-self.d1)
        )

    def delta(self) -> float:
        discount_dividend = np.exp(-self.dividend_yield * self.maturity)

        if self.option_type == "call":
            return discount_dividend * norm.cdf(self.d1)

        return discount_dividend * (norm.cdf(self.d1) - 1)

    def gamma(self) -> float:
        discount_dividend = np.exp(-self.dividend_yield * self.maturity)
        denominator = self.spot * self.volatility * np.sqrt(self.maturity)
        return discount_dividend * norm.pdf(self.d1) / denominator

    def vega(self) -> float:
        discount_dividend = np.exp(-self.dividend_yield * self.maturity)
        return self.spot * discount_dividend * norm.pdf(self.d1) * np.sqrt(self.maturity)

    def theta(self) -> float:
        discount_rate = np.exp(-self.risk_free_rate * self.maturity)
        discount_dividend = np.exp(-self.dividend_yield * self.maturity)

        first_term = (
            -self.spot
            * discount_dividend
            * norm.pdf(self.d1)
            * self.volatility
            / (2 * np.sqrt(self.maturity))
        )

        if self.option_type == "call":
            return (
                first_term
                - self.risk_free_rate * self.strike * discount_rate * norm.cdf(self.d2)
                + self.dividend_yield * self.spot * discount_dividend * norm.cdf(self.d1)
            )

        return (
            first_term
            + self.risk_free_rate * self.strike * discount_rate * norm.cdf(-self.d2)
            - self.dividend_yield * self.spot * discount_dividend * norm.cdf(-self.d1)
        )

    def rho(self) -> float:
        discount_rate = np.exp(-self.risk_free_rate * self.maturity)

        if self.option_type == "call":
            return self.strike * self.maturity * discount_rate * norm.cdf(self.d2)

        return -self.strike * self.maturity * discount_rate * norm.cdf(-self.d2)

    def greeks(self) -> dict[str, float]:
        return {
            "delta": self.delta(),
            "gamma": self.gamma(),
            "vega": self.vega(),
            "theta": self.theta(),
            "rho": self.rho(),
        }


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    maturity: float,
    risk_free_rate: float,
    option_type: str = "call",
    dividend_yield: float = 0.0,
    lower_bound: float = 1e-6,
    upper_bound: float = 5.0,
) -> float:
    """Solve for the volatility that matches an observed option price."""
    if market_price <= 0:
        raise ValueError("Market price must be positive.")

    def objective(volatility):
        option = BlackScholesOption(
            spot=spot,
            strike=strike,
            maturity=maturity,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            option_type=option_type,
            dividend_yield=dividend_yield,
        )
        return option.price() - market_price

    try:
        return brentq(objective, lower_bound, upper_bound)
    except ValueError as exc:
        raise ValueError("Market price is outside the solvable volatility range.") from exc
