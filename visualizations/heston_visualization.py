import matplotlib.pyplot as plt

from models.heston_model import HestonModel


model = HestonModel(
    initial_price=100,
    initial_variance=0.04,
    risk_free_rate=0.03,
    kappa=2.0,
    theta=0.04,
    volatility_of_variance=0.35,
    rho=-0.6,
)

prices, variances = model.simulate(paths=10, steps=252, seed=21)

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

for path in prices:
    axes[0].plot(path)

for path in variances:
    axes[1].plot(path)

axes[0].set_title("Heston Price Paths")
axes[0].set_ylabel("Asset Price")
axes[0].grid(True)

axes[1].set_title("Heston Variance Paths")
axes[1].set_xlabel("Time Step")
axes[1].set_ylabel("Variance")
axes[1].grid(True)

plt.tight_layout()
plt.show()
