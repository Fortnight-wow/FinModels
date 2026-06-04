import numpy as np
import matplotlib.pyplot as plt

from models.risk_metrics import RiskMetrics


def main():
    # Same style of return sample as the VaR plot, but here the tail itself is
    # highlighted so students can see that CVaR is the average of the worst losses.
    rng = np.random.RandomState(42)
    returns = rng.normal(loc=0.001, scale=0.02, size=5000)
    returns[:250] -= rng.uniform(0.03, 0.10, size=250)

    confidence = 0.95
    var_value = RiskMetrics.value_at_risk(returns, confidence=confidence)
    cvar_value = RiskMetrics.conditional_value_at_risk(returns, confidence=confidence)

    tail_losses = returns[returns <= var_value]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(returns, bins=50)

    # Emphasize the full tail region.
    tail_mask = returns <= var_value
    ax.hist(returns[tail_mask], bins=25)

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

    ax.fill_betweenx(
        [0, ax.get_ylim()[1]],
        -1,
        var_value,
        alpha=0.12,
        label="Tail region",
    )

    ax.set_title("Conditional Value at Risk (CVaR) Illustration")
    ax.set_xlabel("Portfolio return")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
