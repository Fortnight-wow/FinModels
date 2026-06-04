import matplotlib.pyplot as plt
import numpy as np

from models.binomial_tree import BinomialTreeOption


def build_tree(option: BinomialTreeOption):
    dt = option.maturity / option.steps
    up = np.exp(option.volatility * np.sqrt(dt))
    down = 1 / up
    discount = np.exp(-option.risk_free_rate * dt)
    growth = np.exp((option.risk_free_rate - option.dividend_yield) * dt)
    probability = (growth - down) / (up - down)

    stock_tree = []
    for step in range(option.steps + 1):
        node_ids = np.arange(step + 1)
        prices = option.spot * up ** (step - node_ids) * down**node_ids
        stock_tree.append(prices)

    option_tree = [None] * (option.steps + 1)
    option_tree[-1] = option._payoff(stock_tree[-1])

    for step in range(option.steps - 1, -1, -1):
        continuation = discount * (
            probability * option_tree[step + 1][:-1]
            + (1 - probability) * option_tree[step + 1][1:]
        )

        if option.american:
            exercise = option._payoff(stock_tree[step])
            option_tree[step] = np.maximum(continuation, exercise)
        else:
            option_tree[step] = continuation

    return stock_tree, option_tree, up, down, probability


def plot_tree(ax, tree, title, y_label):
    for step, values in enumerate(tree):
        x = np.full(len(values), step)
        ax.scatter(x, values, s=35)

        if step > 0:
            prev = tree[step - 1]
            for i, value in enumerate(values):
                ax.plot([step - 1, step], [prev[max(i - 1, 0)], value], linewidth=0.8)
                if i < len(prev):
                    ax.plot([step - 1, step], [prev[min(i, len(prev) - 1)], value], linewidth=0.8)

    ax.set_title(title)
    ax.set_xlabel('Step')
    ax.set_ylabel(y_label)
    ax.grid(True)


if __name__ == '__main__':
    option = BinomialTreeOption(
        spot=100,
        strike=100,
        maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.2,
        steps=5,
        option_type='call',
        american=False,
    )

    stock_tree, option_tree, _, _, probability = build_tree(option)
    price = option.price()

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    plot_tree(axes[0], stock_tree, 'Binomial Stock Price Tree', 'Stock Price')
    plot_tree(axes[1], option_tree, f'Option Value Tree (p = {probability:.3f}, price = {price:.2f})', 'Option Value')

    plt.tight_layout()
    plt.show()
