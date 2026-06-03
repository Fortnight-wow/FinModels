# FinModels

FinModels is a collection of financial engineering models, simulations, and visualizations built for learning quantitative finance by coding it.

The idea is simple: a model becomes easier to understand when you can change its inputs, simulate its paths, plot its behavior, and compare the result with the theory. This repository is meant to grow into a practical playground for pricing, stochastic processes, interest rates, volatility, risk, and portfolio ideas.

## What Is Inside

- Stochastic process simulations
- Option pricing models
- Stochastic volatility models
- Jump diffusion models
- Interest rate models
- Exotic option Monte Carlo tools
- Implementation notes for extending each model
- Small tests that check core numerical behavior

## Current Model Map

| Area | Model | Status | Main use |
| --- | --- | --- | --- |
| Stochastic processes | Geometric Brownian Motion | Implemented | Simulate stock price paths |
| Derivatives | Black-Scholes | Implemented | European option pricing and Greeks |
| Derivatives | Binomial Tree | Implemented | European and American option pricing |
| Jump processes | Merton Jump Diffusion | Implemented | Model sudden market jumps |
| Volatility | Heston Model | Implemented | Stochastic volatility simulation and pricing |
| Rates | Vasicek | Implemented | Mean-reverting short rates and bond pricing |
| Rates | CIR | Implemented | Positive short-rate simulation and bond pricing |
| Rates | Hull-White | Implemented | One-factor short-rate simulation |
| Volatility smile | SABR | Implemented | Implied volatility smile approximation |
| Exotics | Asian, barrier, lookback options | Implemented | Monte Carlo pricing examples |
| Portfolio and risk | Markowitz, VaR, CVaR | Planned | Portfolio construction and risk reporting |

## Quick Start

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run a GBM visualization:

```bash
python visualizations/gbm_visualization.py
```

Price a European call option:

```python
from models.black_scholes import BlackScholesOption

option = BlackScholesOption(
    spot=100,
    strike=105,
    maturity=1.0,
    risk_free_rate=0.05,
    volatility=0.20,
    option_type="call",
)

print(option.price())
print(option.greeks())
```

Simulate Heston paths:

```python
from models.heston_model import HestonModel

model = HestonModel(
    initial_price=100,
    initial_variance=0.04,
    risk_free_rate=0.03,
    kappa=2.0,
    theta=0.04,
    volatility_of_variance=0.35,
    rho=-0.6,
)

prices, variances = model.simulate(paths=5, seed=7)
```

## Implementation Guide

The detailed plan for adding and extending models lives in:

[docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md)

That guide explains how each model should be implemented, tested, visualized, and improved over time.

## Repository Structure

```text
FinModels/
  docs/
    IMPLEMENTATION_GUIDE.md
  models/
    black_scholes.py
    binomial_tree.py
    cir_model.py
    exotic_options.py
    geometric_brownian_motion.py
    heston_model.py
    hull_white_model.py
    merton_jump_diffusion.py
    sabr_model.py
    vasicek_model.py
  tests/
    test_models.py
  visualizations/
    black_scholes_visualization.py
    gbm_visualization.py
    heston_visualization.py
    sabr_smile_visualization.py
```

## Project Philosophy

This is not trying to be a black-box finance library. The goal is to keep the code readable enough that a student can open a file, understand the equation behind it, run the model, and then improve it.

Each model should eventually include:

1. A short theory note
2. Mathematical formulation
3. Clean implementation
4. Example usage
5. Visualization
6. Tests or sanity checks
7. References for deeper study

## Next Steps

- Add visualizations for interest-rate paths and jump diffusion behavior
- Add portfolio optimization models
- Add VaR and CVaR risk tools
- Add calibration examples using market-like data
- Add notebooks for guided explanations
- Add generated plot screenshots to the README

## License

Copyright (c) 2026 Krishna Kumar Jangid (Fortnight-wow).

This repository is available for educational viewing and study. Redistribution or commercial use requires explicit permission from the owner.
