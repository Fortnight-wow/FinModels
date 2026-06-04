import numpy as np
import matplotlib.pyplot as plt

from models.hull_white_model import HullWhiteModel


def main():
    # A gently time-varying theta curve makes the idea of term-structure fitting visible.
    def theta_curve(t):
        return 0.04 + 0.01 * np.sin(np.pi * t / 2)

    model = HullWhiteModel(
        initial_rate=0.03,
        mean_reversion=1.2,
        volatility=0.02,
        theta_curve=theta_curve,
        seed=42,
    )

    rates, times = model.simulate_paths_and_times(T=5, dt=1 / 252, paths=20)
    theta_values = np.array([theta_curve(t) for t in times])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for path_idx in range(rates.shape[1]):
        axes[0].plot(times, rates[:, path_idx], linewidth=1)

    axes[0].plot(
        times,
        theta_values,
        linestyle="--",
        linewidth=2.5,
        label=r"$\theta(t)$",
    )
    axes[0].set_title("Hull-White Short Rate Paths")
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
        label=f"Mean terminal rate: {np.mean(terminal_rates):.2%}",
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
