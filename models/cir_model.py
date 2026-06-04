import numpy as np
from typing import Tuple
from scipy.special import iv, jn  # Bessel functions


class CIRModel:
    """
    Cox-Ingersoll-Ross (CIR) model for interest rate simulation.
    
    dr_t = a*(b - r_t)*dt + σ*√(r_t)*dW_t
    
    where:
    - r_t: short rate at time t
    - a: mean reversion speed
    - b: long-term mean level
    - σ: volatility
    
    Key features:
    - Mean-reverting square-root process
    - Ensures non-negative rates (Feller condition: 2ab ≥ σ²)
    - More realistic than Vasicek for interest rate behavior
    - Closed-form solutions available for bond prices
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
        Initialize CIR model parameters.
        
        Args:
            initial_rate: r_0, initial short rate (must be positive)
            mean_reversion: a, speed of mean reversion
            long_term_mean: b, long-term mean level
            volatility: σ, volatility of the short rate
            seed: random seed for reproducibility
            
        Note:
            For non-negative rates, Feller condition requires: 2*a*b >= σ²
        """
        if initial_rate < 0:
            raise ValueError("Initial rate must be non-negative")
        
        self.r0 = initial_rate
        self.a = mean_reversion
        self.b = long_term_mean
        self.sigma = volatility
        self.rng = np.random.RandomState(seed)
        
        # Check Feller condition
        feller_condition = 2 * self.a * self.b
        if feller_condition < self.sigma**2:
            print(f"Warning: Feller condition not satisfied. 2ab={feller_condition:.4f} < σ²={self.sigma**2:.4f}")
            print("Rates may become negative in simulations.")
    
    def simulate(
        self,
        T: float,
        dt: float = 1/252,
        paths: int = 1
    ) -> np.ndarray:
        """
        Simulate CIR short rate paths using Milstein discretization.
        Milstein scheme reduces bias compared to Euler for square-root process.
        
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
            sqrt_r = np.sqrt(np.maximum(r[i], 0))  # Ensure non-negative
            
            # Milstein scheme
            r[i + 1] = (
                r[i]
                + self.a * (self.b - r[i]) * dt
                + self.sigma * sqrt_r * dW
                + 0.25 * self.sigma**2 * (dW**2 - dt)
            )
            
            # Enforce non-negative constraint
            r[i + 1] = np.maximum(r[i + 1], 0)
        
        return r
    
    def simulate_paths_and_times(
        self,
        T: float,
        dt: float = 1/252,
        paths: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate CIR paths and return rates and time grid.
        
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
        Calculate bond price using CIR closed-form formula.
        
        Bond price: P(t, T) = A(t, T) * exp(-B(t, T) * r_t)
        
        where B and A have specific forms involving square-root processes.
        
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
        
        # CIR parameters
        h = np.sqrt(self.a**2 + 2 * self.sigma**2)
        
        # B(t, T)
        B = 2 * (np.exp(h * tau) - 1) / (2 * h + (self.a + h) * (np.exp(h * tau) - 1))
        
        # A(t, T)
        numerator = 2 * h * np.exp((self.a + h) * tau / 2)
        denominator = 2 * h + (self.a + h) * (np.exp(h * tau) - 1)
        A = (numerator / denominator) ** (2 * self.a * self.b / self.sigma**2)
        
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
