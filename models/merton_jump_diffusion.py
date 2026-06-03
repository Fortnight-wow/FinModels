import numpy as np


class MertonJumpDiffusion:
    """Simulate price paths with continuous diffusion plus Poisson jumps."""

    def __init__(
        self,
        initial_price: float,
        drift: float,
        volatility: float,
        jump_intensity: float,
        jump_mean: float,
        jump_std: float,
    ):
        if initial_price <= 0:
            raise ValueError("Initial price must be positive.")
        if volatility < 0:
            raise ValueError("Volatility cannot be negative.")
        if jump_intensity < 0:
            raise ValueError("Jump intensity cannot be negative.")
        if jump_std < 0:
            raise ValueError("Jump standard deviation cannot be negative.")

        self.initial_price = initial_price
        self.drift = drift
        self.volatility = volatility
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std

    @property
    def jump_compensator(self) -> float:
        expected_jump_size = np.exp(self.jump_mean + 0.5 * self.jump_std**2) - 1
        return self.jump_intensity * expected_jump_size

    def simulate(self, maturity: float = 1.0, steps: int = 252, paths: int = 1000, seed=None):
        rng = np.random.default_rng(seed)
        dt = maturity / steps

        prices = np.empty((paths, steps + 1))
        prices[:, 0] = self.initial_price

        drift_adjustment = self.drift - self.jump_compensator

        for step in range(1, steps + 1):
            normal_shock = rng.normal(size=paths)
            jump_count = rng.poisson(self.jump_intensity * dt, size=paths)
            jump_shock = rng.normal(
                loc=jump_count * self.jump_mean,
                scale=np.sqrt(jump_count) * self.jump_std,
            )

            prices[:, step] = prices[:, step - 1] * np.exp(
                (drift_adjustment - 0.5 * self.volatility**2) * dt
                + self.volatility * np.sqrt(dt) * normal_shock
                + jump_shock
            )

        return prices

    def monte_carlo_call_price(
        self,
        strike: float,
        risk_free_rate: float,
        maturity: float = 1.0,
        steps: int = 252,
        paths: int = 10000,
        seed=None,
    ) -> float:
        simulated = self.simulate(maturity=maturity, steps=steps, paths=paths, seed=seed)
        payoff = np.maximum(simulated[:, -1] - strike, 0.0)
        return float(np.exp(-risk_free_rate * maturity) * payoff.mean())
