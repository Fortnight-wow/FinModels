import numpy as np
from typing import Tuple


class VasicekModel:
    """
    Vasicek model for interest rate simulation.
    
    dr_t = a*(b - r_t)*dt + σ*dW_t
    
    where:
    - r_t: short rate at time t
    - a: mean reversion speed
    - b: long-term mean level
    - σ: volatility
    - dW_t: Brownian motion increment
    
    The Vasicek model allows for negative rates and is analytically tractable.
    Bond prices and options on bonds have closed-form solutions.
    """
    
    def __init__(
        self,
        initial_rate: float,
        mean_reversion: float,
        long_term_mean: float,
        volatility: float,
        seed: int = None
    ):
        """
        Initialize Vasicek model parameters.
        
        Args:
            initial_rate: r_0, initial short rate
            mean_reversion: a, speed of mean reversion
            long_term_mean: b, long-term mean level
            volatility: σ, volatility of the short rate
            seed: random seed for reproducibility
        """
        self.r0 = initial_rate
        self.a = mean_reversion
        self.b = long_term_mean
        self.sigma = volatility
        self.rng = np.random.RandomState(seed)
    
    def simulate(
        self,
        T: float,
        dt: float = 1/252,
        paths: int = 1
    ) -> np.ndarray:
        """
        Simulate Vasicek short rate paths using Euler discretization.
        
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
        
        for i in range(steps):
            dW = self.rng.normal(0, sqrt_dt, paths)
            r[i + 1] = r[i] + self.a * (self.b - r[i]) * dt + self.sigma * dW
        
        return r
    
    def simulate_paths_and_times(
        self,
        T: float,
        dt: float = 1/252,
        paths: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Vasicek paths and return rates and time grid.
        
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
        current_rate: float
    ) -> float:
        """
        Calculate bond price using Vasicek closed-form formula.
        
        Bond price: P(t, T) = A(t, T) * exp(-B(t, T) * r_t)
        
        where:
        B(t, T) = (1 - exp(-a*(T-t))) / a
        A(t, T) = exp((b - σ²/(2*a²)) * (B(t,T) - (T-t)) - σ² * B(t,T)² / (4*a))
        
        Args:
            current_time: t, current time
            maturity: T, bond maturity time
            current_rate: r_t, current short rate
            
        Returns:
            Bond price at current time
        """
        tau = maturity - current_time
        
        if tau <= 0:
            return 1.0
        
        B = (1 - np.exp(-self.a * tau)) / self.a
        
        exponent = (
            (self.b - self.sigma**2 / (2 * self.a**2)) * (B - tau)
            - self.sigma**2 * B**2 / (4 * self.a)
        )
        
        A = np.exp(exponent)
        
        price = A * np.exp(-B * current_rate)
        
        return price
    
    def zero_coupon_yield(
        self,
        current_time: float,
        maturity: float,
        current_rate: float
    ) -> float:
        """
        Calculate zero-coupon bond yield (spot rate).
        
        Args:
            current_time: t, current time
            maturity: T, bond maturity time
            current_rate: r_t, current short rate
            
        Returns:
            Spot yield for the given maturity
        """
        tau = maturity - current_time
        
        if tau <= 0:
            return current_rate
        
        bond_price = self.bond_price(current_time, maturity, current_rate)
        
        yield_value = -np.log(bond_price) / tau
        
        return yield_value
