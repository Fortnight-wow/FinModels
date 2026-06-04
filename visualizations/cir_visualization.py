import matplotlib.pyplot as plt
import numpy as np

from models.cir_model import CIRModel
from models.vasicek_model import VasicekModel


def main():
    vasicek = VasicekModel(
        initial_rate=0.03,
        mean_reversion=1.4,
        long_term_mean=0.05,
        volatility=0.08,
        seed=42,
    )

    cir = CIRModel(
        initial_rate=0.03,
        mean_reversion=1.4,
        long_term_mean=0.05,
        volatility=0.08,
        seed=42,
    )

    vasicek_rates, times = vasicek.simulate_paths_and_times(T=5, dt=1 / 252, paths=20)
    cir_rates, _ = cir.simulate_paths_and_times(T=5, dt=1 / 252, paths=20)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for path_idx in range(vasicek_rates.shape[1]):
        axes[0].plot(times, vasicek_rates[:, path_idx], linewidth=1)

    axes[0].axhline(0.0, linestyle=":", linewidth=1.5)
    axes[0].set_title("Vasicek Rate Paths")
    axes[0].set_xlabel("Time (years)")
    axes[0].set_ylabel("Short rate")
    axes[0].grid(True)

    for path_idx in range(cir_rates.shape[1]):
        axes[1].plot(times, cir_rates[:, path_idx], linewidth=1)

    axes[1].axhline(0.0, linestyle=":", linewidth=1.5)
    axes[1].set_title("CIR Rate Paths")
    axes[1].set_xlabel("Time (years)")
    axes[1].set_ylabel("Short rate")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
