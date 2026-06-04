import numpy as np
import matplotlib.pyplot as plt


def main():
    risk_free_rate = 0.03
    market_return = 0.11

    betas = np.linspace(0, 2, 100)
    expected_returns = risk_free_rate + betas * (market_return - risk_free_rate)

    sample_betas = np.array([0.4, 0.8, 1.0, 1.3, 1.7])
    sample_returns = risk_free_rate + sample_betas * (market_return - risk_free_rate)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        betas,
        expected_returns,
        linewidth=2.5,
        label='Security Market Line'
    )

    ax.scatter(
        sample_betas,
        sample_returns,
        s=80,
        label='Example Assets'
    )

    for idx, (beta, ret) in enumerate(zip(sample_betas, sample_returns), start=1):
        ax.annotate(
            f'Asset {idx}',
            (beta, ret),
            textcoords='offset points',
            xytext=(5, 5)
        )

    ax.axvline(1.0, linestyle='--', linewidth=1.5, label='Market Beta = 1')

    ax.set_title('CAPM Security Market Line')
    ax.set_xlabel('Beta (Systematic Risk)')
    ax.set_ylabel('Expected Return')
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
