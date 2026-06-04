import numpy as np
import matplotlib.pyplot as plt

from models.vasicek_model import VasicekModel


def main():
    model = VasicekModel(
        initial_rate=0.03,
        mean_reversion=1.4,
        long_term_mean=0.05,
        volatility=0.02,
        seed=42,
    )

    rates, times = model.simulate_paths_and_times(T=5, dt=1 / 252, paths=20)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for path_idx in range(rates.shape[1]):
        axes[0].plot(times, rates[:, path_idx], linewidth=1)

    axes[0].axhline(
        model.b,
        linestyle="--",
        linewidth=2,
        label=f"Long-run mean: {model.b:.2%}",
    )
    axes[0].set_title("Vasicek Interest Rate Paths")
    axes[0].set_xlabel("Time (years)")
    axes[0].set_ylabel("Short rate")
    axes[0].legend()
    axes[0].grid(True)

    terminal_rates = rates[-1]
    axes[1].hist(terminal_rates)
    axes[1].axvline(
        np.mean(terminal_rates),
        linestyle="--",
        linewidth=2,
        label=f"Average terminal rate: {np.mean(terminal_rates):.2%}",
    )
    axes[1].axvline(
        model.b,
        linestyle=":",
        linewidth=2,
        label="Long-run mean",
    )
    axes[1].set_title("Distribution of Terminal Rates")
    axes[1].set_xlabel("Rate")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
