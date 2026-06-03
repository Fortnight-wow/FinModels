# FinModels Implementation Guide

This guide explains how to build the project in the same spirit as the original idea: clean models, visible intuition, and enough math to show that the implementation is not random code.

The best way to grow this repository is to add one model at a time with:

1. Theory
2. Inputs and assumptions
3. Implementation
4. Example usage
5. Visualization
6. Tests or numerical sanity checks
7. Possible extensions

## Suggested Build Order

Start with the models that explain the most ideas with the least machinery.

| Order | Project | Why it matters |
| --- | --- | --- |
| 1 | Geometric Brownian Motion | Foundation for equity price simulation |
| 2 | Black-Scholes | Baseline for option pricing and Greeks |
| 3 | Binomial Tree | Shows backward induction and American exercise |
| 4 | Jump Diffusion | Adds sudden shocks and non-normal returns |
| 5 | Heston Model | Adds stochastic volatility and correlated risk |
| 6 | Vasicek and CIR | Introduces short-rate modeling |
| 7 | Hull-White | Shows curve-aware interest-rate modeling |
| 8 | SABR | Explains volatility smiles and skew |
| 9 | Exotic Options | Uses Monte Carlo for path-dependent payoffs |
| 10 | Portfolio and Risk | Connects models to allocation and risk control |

## 1. Geometric Brownian Motion

### Goal

Simulate stock prices where returns follow a drift plus random volatility term.

### Model

```text
dS = mu S dt + sigma S dW
```

### Implementation Steps

1. Accept initial price, drift, volatility, maturity, time steps, and number of paths.
2. Generate standard normal shocks.
3. Convert shocks into log-return increments.
4. Cumulatively sum the increments.
5. Exponentiate to get price paths.

### Visualizations

- Simulated stock paths
- Distribution of final prices
- Effect of changing volatility

### Extensions

- Add confidence bands
- Compare simulated mean with theoretical expectation
- Add risk-neutral drift for option-pricing use

## 2. Black-Scholes

### Goal

Price European call and put options using a closed-form formula.

### Model Inputs

- Spot price
- Strike price
- Time to maturity
- Risk-free rate
- Volatility
- Option type
- Optional dividend yield

### Implementation Steps

1. Validate all inputs.
2. Compute `d1` and `d2`.
3. Use the normal CDF for call and put prices.
4. Add Greeks: delta, gamma, vega, theta, rho.
5. Add implied volatility by solving for the volatility that matches a market price.

### Visualizations

- Option price vs spot price
- Delta and gamma curves
- Vega across strikes
- Implied volatility smile from sample market prices

### Extensions

- Add dividend-adjusted pricing examples
- Compare Black-Scholes prices with binomial tree prices
- Add calibration from option chains

## 3. Binomial Tree

### Goal

Build option prices by stepping backward through a recombining price tree.

### Implementation Steps

1. Split maturity into small time intervals.
2. Compute up and down price moves.
3. Compute the risk-neutral probability.
4. Build terminal option payoffs.
5. Discount backward one layer at a time.
6. For American options, compare continuation value with immediate exercise value.

### Visualizations

- Small tree diagram for 3 to 5 steps
- Convergence to Black-Scholes as steps increase
- American vs European put prices

### Extensions

- Add trinomial tree
- Add discrete dividends
- Add early-exercise boundary plots

## 4. Merton Jump Diffusion

### Goal

Improve GBM by allowing sudden price jumps.

### Model

```text
dS / S = (mu - lambda k) dt + sigma dW + dJ
```

Where:

- `lambda` is jump intensity
- `k` is expected jump size adjustment
- `J` is a compound Poisson jump process

### Implementation Steps

1. Simulate regular diffusion shocks.
2. Simulate jump counts using a Poisson distribution.
3. Simulate jump sizes using a normal distribution.
4. Add the jump compensator so the drift is adjusted correctly.
5. Return full simulated paths.
6. Price calls by discounting Monte Carlo payoffs.

### Visualizations

- GBM paths vs jump diffusion paths
- Histogram of terminal returns
- Effect of jump intensity on option prices

### Extensions

- Add put pricing
- Add calibration to implied volatility smiles
- Compare with Black-Scholes under zero jump intensity

## 5. Heston Stochastic Volatility

### Goal

Model volatility as its own random process instead of keeping it constant.

### Model

```text
dS = r S dt + sqrt(v) S dW1
dv = kappa(theta - v)dt + xi sqrt(v) dW2
corr(dW1, dW2) = rho
```

### Implementation Steps

1. Store price and variance paths.
2. Generate two correlated normal shocks.
3. Use full-truncation Euler so variance does not become negative.
4. Update variance first.
5. Update the asset price using the previous variance.
6. Discount Monte Carlo payoffs for option pricing.

### Visualizations

- Asset paths and volatility paths
- Effect of negative correlation on downside volatility
- Monte Carlo price convergence

### Extensions

- Add semi-closed-form Heston pricing
- Add calibration to volatility surfaces
- Add variance process diagnostics

## 6. Vasicek Interest Rate Model

### Goal

Simulate mean-reverting short rates.

### Model

```text
dr = a(b - r)dt + sigma dW
```

### Implementation Steps

1. Accept initial rate, mean reversion speed, long-term mean, and volatility.
2. Simulate rates using Euler steps.
3. Add the closed-form zero-coupon bond price.
4. Use the model to create simple yield curves.

### Visualizations

- Short-rate paths
- Yield curve for different starting rates
- Effect of mean reversion

### Extensions

- Add exact simulation
- Add bond option pricing
- Compare against CIR

## 7. CIR Interest Rate Model

### Goal

Model positive mean-reverting rates with volatility proportional to `sqrt(r)`.

### Model

```text
dr = a(b - r)dt + sigma sqrt(r) dW
```

### Implementation Steps

1. Validate that inputs are compatible with non-negative rates.
2. Simulate using full truncation.
3. Track the Feller ratio.
4. Add the closed-form zero-coupon bond price.

### Visualizations

- CIR rate paths
- Vasicek vs CIR comparison
- Bond prices across maturities

### Extensions

- Add exact non-central chi-square simulation
- Add caplet pricing
- Add calibration to a yield curve

## 8. Hull-White

### Goal

Build a one-factor short-rate model that can eventually be calibrated to a term structure.

### Model

```text
dr = (theta(t) - a r)dt + sigma dW
```

### Implementation Steps

1. Allow `theta` to be either a constant or a function of time.
2. Simulate short-rate paths.
3. Estimate pathwise discount factors.
4. Keep the implementation flexible for curve calibration later.

### Visualizations

- Short-rate paths under different theta functions
- Discount factor distribution
- Comparison against Vasicek when theta is constant

### Extensions

- Add term-structure calibration
- Add swaption pricing
- Add Jamshidian decomposition

## 9. SABR

### Goal

Model volatility smiles for rates, forwards, or option markets.

### Model

```text
dF = alpha F^beta dW1
dalpha = nu alpha dW2
corr(dW1, dW2) = rho
```

### Implementation Steps

1. Validate forward, alpha, beta, rho, and nu.
2. Implement Hagan's implied volatility approximation.
3. Handle the at-the-money case separately.
4. Add path simulation for intuition.
5. Plot implied volatility across strikes.

### Visualizations

- SABR smile
- Effect of beta on curvature
- Effect of rho on skew
- Effect of nu on smile width

### Extensions

- Calibrate SABR parameters to option quotes
- Add normal SABR for negative-rate environments
- Compare SABR smile with Heston-generated smile

## 10. Exotic Options

### Goal

Price path-dependent options using Monte Carlo.

### Options To Start With

- Arithmetic Asian call
- Up-and-out barrier call
- Fixed-strike lookback call

### Implementation Steps

1. Simulate GBM paths under risk-neutral drift.
2. Define payoff logic separately for each option.
3. Discount average payoff.
4. Return both price and standard error.
5. Use a fixed seed in examples for reproducibility.

### Visualizations

- Sample paths that knock out vs survive
- Monte Carlo convergence chart
- Payoff comparison across strikes

### Extensions

- Add antithetic variates
- Add control variates
- Add down-and-out, up-and-in, and digital options

## 11. Portfolio And Risk Projects

These are planned next because they make the repository useful beyond pricing.

### Markowitz Portfolio Optimization

Implementation plan:

1. Load returns data.
2. Compute expected returns and covariance matrix.
3. Optimize portfolio weights for maximum Sharpe ratio.
4. Add constraints such as long-only weights.
5. Plot the efficient frontier.

### CAPM

Implementation plan:

1. Estimate beta using regression against market returns.
2. Compute expected return using the CAPM formula.
3. Compare actual returns with CAPM-implied returns.
4. Visualize security market line.

### Value at Risk

Implementation plan:

1. Accept portfolio returns.
2. Add historical VaR.
3. Add parametric Gaussian VaR.
4. Add Monte Carlo VaR.
5. Report results at 95 percent and 99 percent confidence.

### Conditional Value at Risk

Implementation plan:

1. Use the same return samples as VaR.
2. Identify losses beyond the VaR threshold.
3. Average those tail losses.
4. Compare VaR and CVaR under stressed scenarios.


