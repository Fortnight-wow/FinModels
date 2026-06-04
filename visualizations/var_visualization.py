import numpy as np
import matplotlib.pyplot as plt

from models.risk_metrics import RiskMetrics


def main():
    # A simple return distribution with a mild negative skew helps students
    # see why VaR cuts into the left tail.
    rng = np.random.RandomState(42)
    returns = rng.normal(loc=0.001, scale=0.02, size=5000)
    returns[:250] -= rng.uniform(0.03, 0.10, size=250)

    confidence = 0.95
    var_value = RiskMetrics.value_at_risk(returns, confidence=confidence)
    cvar_value = RiskMetrics.conditional_value_at_risk(returns, confidence=confidence)

    fig, ax = plt.subplots(figsize=(10, 6))

    counts, bins, patches = ax.hist(returns, bins=50)

    # Highlight the tail region beyond VaR.
    for left, right, patch in zip(bins[:-1], bins[1:], patches):
        if right <= var_value:
            patch.set_alpha(0.6)

    ax.axvline(
        var_value,
        linestyle="--",
        linewidth=2.5,
        label=f"VaR (95%) = {var_value:.2%}",
    )
    ax.axvline(
        cvar_value,
        linestyle=":",
        linewidth=2.5,
        label=f"CVaR (95%) = {cvar_value:.2%}",
    )
    ax.axvline(0.0, linewidth=1.2, label="Zero return")

    ax.set_title("Value at Risk (VaR) Illustration")
    ax.set_xlabel("Portfolio return")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
