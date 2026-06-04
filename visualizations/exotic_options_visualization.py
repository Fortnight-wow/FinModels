import matplotlib.pyplot as plt
import numpy as np

from models.exotic_options import ExoticOptions
from models.geometric_brownian_motion import GeometricBrownianMotion


def main():
    spot = 100.0
    strike = 100.0
    barrier = 120.0
    maturity = 1.0
    steps = 252
    paths = 2000
    seed = 42

    gbm = GeometricBrownianMotion(
        initial_price=spot,
        drift=0.08,
        volatility=0.20,
    )

    simulated_paths = gbm.simulate_paths(
        maturity=maturity,
        steps=steps,
        paths=paths,
        seed=seed,
    ).T

    discount_factor = np.exp(-0.05 * maturity)

    asian_price, asian_se = ExoticOptions.asian_option(
        simulated_paths,
        strike=strike,
        option_type="call",
        averaging="arithmetic",
        discount_factor=discount_factor,
    )

    barrier_price, barrier_se = ExoticOptions.barrier_option(
        simulated_paths,
        strike=strike,
        barrier=barrier,
        option_type="call",
        barrier_type="knock_out",
        barrier_side="up",
        discount_factor=discount_factor,
    )

    lookback_price, lookback_se = ExoticOptions.lookback_option(
        simulated_paths,
        strike=strike,
        option_type="call",
        lookback_type="fixed",
        discount_factor=discount_factor,
    )

    time = np.linspace(0, maturity, simulated_paths.shape[0])
    sample_indices = np.linspace(0, simulated_paths.shape[1] - 1, 12, dtype=int)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Path intuition
    for idx in sample_indices:
        axes[0, 0].plot(time, simulated_paths[:, idx], linewidth=1)
    axes[0, 0].axhline(strike, linestyle="--", linewidth=1.5, label="Strike")
    axes[0, 0].axhline(barrier, linestyle=":", linewidth=1.5, label="Barrier")
    axes[0, 0].set_title("Sample Price Paths")
    axes[0, 0].set_xlabel("Time (years)")
    axes[0, 0].set_ylabel("Asset price")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Asian option payoff intuition
    running_average = np.mean(simulated_paths, axis=0)
    terminal_prices = simulated_paths[-1, :]
    axes[0, 1].scatter(running_average, terminal_prices, s=8, alpha=0.3)
    axes[0, 1].axvline(strike, linestyle="--", linewidth=1.5, label="Strike")
    axes[0, 1].set_title("Asian Option: Average Price vs Final Price")
    axes[0, 1].set_xlabel("Average path price")
    axes[0, 1].set_ylabel("Final price")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Barrier logic
    max_prices = np.max(simulated_paths, axis=0)
    hit_barrier = max_prices >= barrier
    axes[1, 0].scatter(max_prices[~hit_barrier], terminal_prices[~hit_barrier], s=8, alpha=0.3, label="Barrier not hit")
    axes[1, 0].scatter(max_prices[hit_barrier], terminal_prices[hit_barrier], s=8, alpha=0.3, label="Barrier hit")
    axes[1, 0].axvline(barrier, linestyle="--", linewidth=1.5, label="Barrier level")
    axes[1, 0].set_title("Barrier Option: Maximum Price Reached")
    axes[1, 0].set_xlabel("Maximum path price")
    axes[1, 0].set_ylabel("Final price")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Lookback intuition
    min_prices = np.min(simulated_paths, axis=0)
    axes[1, 1].scatter(min_prices, terminal_prices, s=8, alpha=0.3)
    axes[1, 1].axvline(strike, linestyle="--", linewidth=1.5, label="Strike")
    axes[1, 1].set_title("Lookback Option: Minimum Price Reached")
    axes[1, 1].set_xlabel("Minimum path price")
    axes[1, 1].set_ylabel("Final price")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    fig.suptitle(
        "Exotic Options: Why the Whole Path Matters",
        fontsize=14,
    )

    summary = (
        f"Asian call ≈ {asian_price:.2f} ± {asian_se:.2f}   |   "
        f"Barrier call ≈ {barrier_price:.2f} ± {barrier_se:.2f}   |   "
        f"Lookback call ≈ {lookback_price:.2f} ± {lookback_se:.2f}"
    )
    fig.text(0.5, 0.02, summary, ha="center")

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()
