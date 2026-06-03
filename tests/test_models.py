import numpy as np
import pytest

from models.binomial_tree import BinomialTreeOption
from models.black_scholes import BlackScholesOption, implied_volatility
from models.cir_model import CIRModel
from models.exotic_options import price_arithmetic_asian_call
from models.geometric_brownian_motion import GeometricBrownianMotion
from models.heston_model import HestonModel
from models.sabr_model import SABRModel
from models.vasicek_model import VasicekModel


def test_black_scholes_put_call_parity():
    call = BlackScholesOption(100, 100, 1.0, 0.05, 0.2, "call")
    put = BlackScholesOption(100, 100, 1.0, 0.05, 0.2, "put")

    parity_left = call.price() - put.price()
    parity_right = 100 - 100 * np.exp(-0.05)

    assert parity_left == pytest.approx(parity_right, rel=1e-10)


def test_implied_volatility_recovers_input_volatility():
    option = BlackScholesOption(100, 105, 1.0, 0.03, 0.25, "call")

    solved = implied_volatility(
        market_price=option.price(),
        spot=100,
        strike=105,
        maturity=1.0,
        risk_free_rate=0.03,
        option_type="call",
    )

    assert solved == pytest.approx(0.25, rel=1e-6)


def test_binomial_tree_converges_near_black_scholes():
    bs_price = BlackScholesOption(100, 100, 1.0, 0.05, 0.2, "call").price()
    tree_price = BinomialTreeOption(100, 100, 1.0, 0.05, 0.2, steps=500).price()

    assert tree_price == pytest.approx(bs_price, rel=2e-3)


def test_gbm_paths_have_expected_shape_and_positive_prices():
    model = GeometricBrownianMotion(initial_price=100, drift=0.05, volatility=0.2)
    paths = model.simulate_paths(paths=8, steps=12, seed=42)

    assert paths.shape == (8, 13)
    assert np.all(paths > 0)


def test_heston_paths_keep_variance_non_negative():
    model = HestonModel(
        initial_price=100,
        initial_variance=0.04,
        risk_free_rate=0.03,
        kappa=2.0,
        theta=0.04,
        volatility_of_variance=0.35,
        rho=-0.5,
    )
    prices, variances = model.simulate(paths=20, steps=30, seed=3)

    assert prices.shape == (20, 31)
    assert variances.shape == (20, 31)
    assert np.all(variances >= 0)


def test_interest_rate_bond_prices_are_reasonable():
    vasicek = VasicekModel(0.03, mean_reversion=1.2, long_term_mean=0.04, volatility=0.01)
    cir = CIRModel(0.03, mean_reversion=1.2, long_term_mean=0.04, volatility=0.08)

    assert 0 < vasicek.zero_coupon_bond_price(5.0) < 1.2
    assert 0 < cir.zero_coupon_bond_price(5.0) < 1.2


def test_sabr_implied_volatility_is_positive():
    model = SABRModel(forward=100, alpha=0.25, beta=0.7, rho=-0.3, nu=0.5)

    assert model.implied_volatility(strike=95, maturity=1.0) > 0
    assert model.implied_volatility(strike=100, maturity=1.0) > 0


def test_asian_option_returns_price_and_standard_error():
    price, error = price_arithmetic_asian_call(
        spot=100,
        strike=100,
        risk_free_rate=0.03,
        volatility=0.2,
        paths=2000,
        steps=50,
        seed=11,
    )

    assert price > 0
    assert error > 0
