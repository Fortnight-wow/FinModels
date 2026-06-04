import matplotlib.pyplot as plt
import numpy as np

from models.risk_metrics import RiskMetrics


def main():
    base_prices = np.array([100.0, 80.0, 120.0, 60.0])

    shock_scenarios = {
        "Mild correction": -0.05,
        "Market selloff": -0.12,
        "Crash": -0.25,
        "Recovery": 0.08,
    }

    stressed = RiskMetrics.stress_test(base_prices, shock_scenarios)

    scenario_names = list(stressed.keys())
    x = np.arange(len(base_prices))
    width = 0.2

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(x - 1.5 * width, base_prices, width, label="Base")

    for idx, scenario in enumerate(scenario_names):
        ax.bar(x - 0.5 * width + idx * width, stressed[scenario], width, label=scenario)

    ax.set_title("Stress Testing Under Market Shocks")
    ax.set_xlabel("Asset")
    ax.set_ylabel("Price")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Asset {i}" for i in range(1, len(base_prices) + 1)])
    ax.legend()
    ax.grid(True, axis="y")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
