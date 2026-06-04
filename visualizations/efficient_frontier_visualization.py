import numpy as np
import matplotlib.pyplot as plt

from models.portfolio_optimization import Markowitz


def main():
    # Example with four assets. The numbers are chosen so the chart is readable
    # and the efficient frontier has a visible curve.
    expected_returns = np.array([0.08, 0.12, 0.10, 0.07])

    covariance_matrix = np.array(
        [
            [0.040, 0.006, 0.008, 0.004],
            [0.006, 0.090, 0.010, 0.005],
            [0.008, 0.010, 0.062, 0.006],
            [0.004, 0.005, 0.006, 0.030],
        ]
    )

    risk_free_rate = 0.03
    model = Markowitz(expected_returns, covariance_matrix, risk_free_rate)

    target_returns = np.linspace(0.07, 0.12, 35)
    frontier = model.efficient_frontier(target_returns)

    min_var_weights, min_var_return, min_var_var = model.min_variance_portfolio()
    min_var_std = np.sqrt(min_var_var)

    max_sharpe_weights, max_sharpe_return, max_sharpe_std, max_sharpe_ratio = model.max_sharpe_portfolio()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Efficient frontier
    ax.plot(
        frontier["stds"],
        frontier["returns"],
        linewidth=2.5,
        label="Efficient Frontier",
    )

    # Individual assets
    asset_stds = np.sqrt(np.diag(covariance_matrix))
    ax.scatter(asset_stds, expected_returns, marker="o", s=60, label="Individual Assets")

    for idx, (x, y) in enumerate(zip(asset_stds, expected_returns), start=1):
        ax.annotate(f"Asset {idx}", (x, y), textcoords="offset points", xytext=(6, 6))

    # Minimum variance portfolio
    ax.scatter(
        min_var_std,
        min_var_return,
        marker="*",
        s=180,
        label="Minimum Variance Portfolio",
    )

    # Maximum Sharpe portfolio
    ax.scatter(
        max_sharpe_std,
        max_sharpe_return,
        marker="D",
        s=90,
        label=f"Maximum Sharpe Portfolio (S={max_sharpe_ratio:.2f})",
    )

    ax.axvline(0, linewidth=1)
    ax.axhline(risk_free_rate, linestyle="--", linewidth=1.5, label="Risk-free rate")
    ax.set_title("Markowitz Efficient Frontier")
    ax.set_xlabel("Portfolio Volatility (Std Dev)")
    ax.set_ylabel("Expected Return")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
