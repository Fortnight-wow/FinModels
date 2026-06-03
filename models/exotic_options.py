import numpy as np


def simulate_gbm_paths(
    spot: float,
    risk_free_rate: float,
    volatility: float,
    maturity: float = 1.0,
    steps: int = 252,
    paths: int = 10000,
    dividend_yield: float = 0.0,
    seed=None,
):
    if spot <= 0:
        raise ValueError("spot must be positive.")
    if volatility < 0:
        raise ValueError("volatility cannot be negative.")
    if maturity <= 0:
        raise ValueError("maturity must be positive.")
    if steps <= 0 or paths <= 0:
        raise ValueError("steps and paths must be positive.")

    rng = np.random.default_rng(seed)
    dt = maturity / steps
    shocks = rng.normal(size=(paths, steps))
    drift = (risk_free_rate - dividend_yield - 0.5 * volatility**2) * dt
    diffusion = volatility * np.sqrt(dt) * shocks

    log_paths = np.cumsum(drift + diffusion, axis=1)
    log_paths = np.column_stack([np.zeros(paths), log_paths])
    return spot * np.exp(log_paths)


def price_arithmetic_asian_call(
    spot: float,
    strike: float,
    risk_free_rate: float,
    volatility: float,
    maturity: float = 1.0,
    steps: int = 252,
    paths: int = 10000,
    seed=None,
) -> tuple[float, float]:
    simulated = simulate_gbm_paths(
        spot=spot,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        maturity=maturity,
        steps=steps,
        paths=paths,
        seed=seed,
    )
    average_prices = simulated[:, 1:].mean(axis=1)
    payoffs = np.maximum(average_prices - strike, 0.0)
    discount = np.exp(-risk_free_rate * maturity)
    discounted = discount * payoffs
    standard_error = discounted.std(ddof=1) / np.sqrt(paths)
    return float(discounted.mean()), float(standard_error)


def price_up_and_out_call(
    spot: float,
    strike: float,
    barrier: float,
    risk_free_rate: float,
    volatility: float,
    maturity: float = 1.0,
    steps: int = 252,
    paths: int = 10000,
    rebate: float = 0.0,
    seed=None,
) -> tuple[float, float]:
    if barrier <= spot:
        raise ValueError("For an up-and-out call, barrier should be above spot.")

    simulated = simulate_gbm_paths(
        spot=spot,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        maturity=maturity,
        steps=steps,
        paths=paths,
        seed=seed,
    )
    knocked_out = (simulated >= barrier).any(axis=1)
    vanilla_payoff = np.maximum(simulated[:, -1] - strike, 0.0)
    payoffs = np.where(knocked_out, rebate, vanilla_payoff)
    discount = np.exp(-risk_free_rate * maturity)
    discounted = discount * payoffs
    standard_error = discounted.std(ddof=1) / np.sqrt(paths)
    return float(discounted.mean()), float(standard_error)


def price_fixed_strike_lookback_call(
    spot: float,
    strike: float,
    risk_free_rate: float,
    volatility: float,
    maturity: float = 1.0,
    steps: int = 252,
    paths: int = 10000,
    seed=None,
) -> tuple[float, float]:
    simulated = simulate_gbm_paths(
        spot=spot,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        maturity=maturity,
        steps=steps,
        paths=paths,
        seed=seed,
    )
    maximum_prices = simulated.max(axis=1)
    payoffs = np.maximum(maximum_prices - strike, 0.0)
    discount = np.exp(-risk_free_rate * maturity)
    discounted = discount * payoffs
    standard_error = discounted.std(ddof=1) / np.sqrt(paths)
    return float(discounted.mean()), float(standard_error)
