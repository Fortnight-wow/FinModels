import matplotlib.pyplot as plt

from models.geometric_brownian_motion import GeometricBrownianMotion


model = GeometricBrownianMotion(
    initial_price=100,
    drift=0.08,
    volatility=0.20,
)

plt.figure(figsize=(10, 6))

for _ in range(15):
    path = model.simulate()
    plt.plot(path)

plt.title('Geometric Brownian Motion Simulation')
plt.xlabel('Trading Days')
plt.ylabel('Asset Price')
plt.grid(True)
plt.show()
