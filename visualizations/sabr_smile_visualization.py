import matplotlib.pyplot as plt
import numpy as np

from models.sabr_model import SABRModel


model = SABRModel(forward=100, alpha=0.25, beta=0.7, rho=-0.35, nu=0.6)

strikes = np.linspace(60, 140, 120)
volatilities = [model.implied_volatility(strike=strike, maturity=1.0) for strike in strikes]

plt.figure(figsize=(10, 6))
plt.plot(strikes, volatilities)
plt.title("SABR Implied Volatility Smile")
plt.xlabel("Strike")
plt.ylabel("Implied Volatility")
plt.grid(True)
plt.show()
