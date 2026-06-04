import matplotlib.pyplot as plt
import numpy as np

from models.geometric_brownian_motion import GeometricBrownianMotion
from models.merton_jump_diffusion import MertonJumpDiffusion


def main():
    start_price = 100
    maturity = 1.0
    steps = 252
    seed = 42

    jump_model = MertonJumpDiffusion(
        initial_price=start_price,
        drift=0.08,
        volatility=0.20,
        jump_intensity=0.75,
        jump_mean=-0.10,
        jump_std=0.20,
    )

    gbm_model = GeometricBrownianMotion(
        initial_price=start_price,
        drift=0.08,
        volatility=0.20,
    )

    jump_paths = jump_model.simulate(maturity=maturity, steps=steps, paths=8, seed=seed)
    gbm_path = gbm_model.generate_path(maturity=maturity, num_steps=steps, random_seed=seed)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for path in jump_paths:
        axes[0].plot(path, linewidth=1)

    axes[0].plot(gbm_path, linestyle='--', linewidth=2, label='GBM reference')
    axes[0].set_title('Jump Diffusion vs GBM Paths')
    axes[0].set_xlabel('Trading Days')
    axes[0].set_ylabel('Asset Price')
    axes[0].legend()
    axes[0].grid(True)

    terminal_prices = jump_model.simulate(maturity=maturity, steps=steps, paths=5000, seed=seed)[:, -1]
    axes[1].hist(terminal_prices, bins=40)
    axes[1].axvline(np.mean(terminal_prices), linestyle='--', label='Mean terminal price')
    axes[1].set_title('Distribution of Terminal Prices')
    axes[1].set_xlabel('Terminal Price')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
