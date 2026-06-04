import numpy as np
from typing import Tuple


class HullWhiteModel:
    """
    Hull-White (one-factor) model for interest rate simulation.
    
    dr_t = (θ(t) - a*r_t)*dt + σ*dW_t
    
    where:
    - r_t: short rate at time t
    - θ(t): time-dependent drift (fitted to initial yield curve)
    - a: mean reversion speed
    - σ: volatility
    
    Key features:
    - Allows negative rates (more flexible than CIR)
    - Can be calibrated to initial term structure
    - Has closed-form solutions for bond prices and options
    - θ(t) can be fitted to match market prices exactly
    """
    
    def __init__(
        self,
        initial_rate: float,
        mean_reversion: float,
        volatility: float,
        theta_curve: callable = None,
        seed: int = None
    ):
        """
        Initialize Hull-White model parameters.
        
        Args:
            initial_rate: r_0, initial short rate
            mean_reversion: a, speed of mean reversion
            volatility: σ, volatility of the short rate
            theta_curve: callable function θ(t) or None for constant theta
            seed: random seed for reproducibility
        """
        self.r0 = initial_rate
        self.a = mean_reversion
        self.sigma = volatility
        self.theta_curve = theta_curve if theta_curve is not None else lambda t: initial_rate
        self.rng = np.random.RandomState(seed)
    
    def set_constant_theta(self, theta: float):
        """
        Set θ(t) to a constant value.
        
        Args:
            theta: constant value for θ(t)
        """
        self.theta_curve = lambda t: theta
    
    def simulate(
        self,
        T: float,
        dt: float = 1/252,
        paths: int = 1
    ) -> np.ndarray:
        """
        Simulate Hull-White short rate paths using Euler discretization.
        
        Args:
            T: total time horizon (in years)
            dt: time step size
            paths: number of simulation paths
            
        Returns:
            Array of shape (steps, paths) containing simulated short rates
        """
        steps = int(T / dt)
        r = np.zeros((steps + 1, paths))
        r[0] = self.r0
        
        sqrt_dt = np.sqrt(dt)
        times = np.linspace(0, T, steps + 1)
        
        for i in range(steps):
            t = times[i]
            theta_t = self.theta_curve(t)
            dW = self.rng.normal(0, sqrt_dt, paths)
            
            r[i + 1] = r[i] + (theta_t - self.a * r[i]) * dt + self.sigma * dW
        
        return r
    
    def simulate_paths_and_times(
        self,
        T: float,
        dt: float = 1/252,
        paths: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Hull-White paths and return rates and time grid.
        
        Args:
            T: total time horizon (in years)
            dt: time step size
            paths: number of paths
            
        Returns:
            Tuple of (rates array, time grid)
        """
        rates = self.simulate(T, dt, paths)
        times = np.linspace(0, T, rates.shape[0])
        return rates, times
    
    def bond_price(
        self,
        current_time: float,
        maturity: float,
        current_rate: float,
        market_price_func: callable = None
    ) -> float:
        """
        Calculate bond price using Hull-White formula.
        
        For calibration to initial term structure, market_price_func should
        be provided to get market prices of zero-coupon bonds.
        
        Args:
            current_time: t, current time
            maturity: T, bond maturity time
            current_rate: r_t, current short rate
            market_price_func: callable(T) returning market price P(0, T)
            
        Returns:
            Bond price at current time
        """
        tau = maturity - current_time
        
        if tau <= 0:
            return 1.0
        
        # Discount factor based on exponential term
        B = (1 - np.exp(-self.a * tau)) / self.a
        
        # For simple model without market calibration
        # Use constant theta (long-term mean)
        if market_price_func is None:
            theta = self.theta_curve(0)
            exponent = (
                (theta - self.sigma**2 / (2 * self.a**2)) * (B - tau)
                - self.sigma**2 * B**2 / (4 * self.a)
            )
            A = np.exp(exponent)
        else:
            # For calibrated model, use market prices
            P_0_T = market_price_func(maturity)
            P_0_t = market_price_func(current_time)
            
            # Adjust using current rate
            A = (P_0_T / P_0_t) * np.exp(-B * (self.r0 - current_rate))
        
        price = A * np.exp(-B * current_rate)
        
        return price
    
    def zero_coupon_yield(
        self,
        current_time: float,
        maturity: float,
        current_rate: float,
        market_price_func: callable = None
    ) -> float:
        """
        Calculate zero-coupon bond yield (spot rate).
        
        Args:
            current_time: t, current time
            maturity: T, bond maturity time
            current_rate: r_t, current short rate
            market_price_func: callable for market price calibration
            
        Returns:
            Spot yield for the given maturity
        """
        tau = maturity - current_time
        
        if tau <= 0:
            return current_rate
        
        bond_price = self.bond_price(current_time, maturity, current_rate, market_price_func)
        
        yield_value = -np.log(bond_price) / tau
        
        return yield_value
