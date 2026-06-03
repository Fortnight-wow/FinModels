from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinomialTreeOption:
    """Cox-Ross-Rubinstein binomial tree for option pricing."""

    spot: float
    strike: float
    maturity: float
    risk_free_rate: float
    volatility: float
    steps: int = 100
    option_type: str = "call"
    american: bool = False
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
        if self.steps <= 0:
            raise ValueError("steps must be positive.")
        if self.option_type not in {"call", "put"}:
            raise ValueError("option_type must be either 'call' or 'put'.")

    def _payoff(self, prices):
        if self.option_type == "call":
            return np.maximum(prices - self.strike, 0.0)
        return np.maximum(self.strike - prices, 0.0)

    def price(self) -> float:
        dt = self.maturity / self.steps
        up = np.exp(self.volatility * np.sqrt(dt))
        down = 1 / up
        discount = np.exp(-self.risk_free_rate * dt)
        growth = np.exp((self.risk_free_rate - self.dividend_yield) * dt)
        probability = (growth - down) / (up - down)

        if probability < 0 or probability > 1:
            raise ValueError("Invalid risk-neutral probability. Increase steps or check inputs.")

        node_ids = np.arange(self.steps + 1)
        terminal_prices = self.spot * up ** (self.steps - node_ids) * down**node_ids
        option_values = self._payoff(terminal_prices)

        for step in range(self.steps - 1, -1, -1):
            option_values = discount * (
                probability * option_values[:-1]
                + (1 - probability) * option_values[1:]
            )

            if self.american:
                node_ids = np.arange(step + 1)
                prices = self.spot * up ** (step - node_ids) * down**node_ids
                option_values = np.maximum(option_values, self._payoff(prices))

        return float(option_values[0])
