import matplotlib.pyplot as plt
import numpy as np

from models.black_scholes import BlackScholesOption


spots = np.linspace(50, 150, 200)
prices = []
deltas = []
gammas = []

for spot in spots:
    option = BlackScholesOption(
        spot=spot,
        strike=100,
        maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.20,
        option_type="call",
    )
    prices.append(option.price())
    deltas.append(option.delta())
    gammas.append(option.gamma())

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

axes[0].plot(spots, prices)
axes[0].set_ylabel("Call Price")
axes[0].grid(True)

axes[1].plot(spots, deltas)
axes[1].set_ylabel("Delta")
axes[1].grid(True)

axes[2].plot(spots, gammas)
axes[2].set_ylabel("Gamma")
axes[2].set_xlabel("Spot Price")
axes[2].grid(True)

fig.suptitle("Black-Scholes Price and Greeks")
plt.tight_layout()
plt.show()
