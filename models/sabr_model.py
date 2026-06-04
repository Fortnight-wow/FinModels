import numpy as np
from scipy.special import gamma, beta as beta_func
from scipy.integrate import quad


class SABRModel:
    """
    SABR (Stochastic Alpha Beta Rho) model for modeling implied volatility smile.
    
    The SABR model is defined as:
    dF_t = α * β * F_t^(β-1) * dW^F_t
    dα_t = ν * α_t * dW^α_t
    
    with correlation ρ between the two Brownian motions.
    
    It provides a closed-form approximation for implied volatility:
    σ_impl(F, K) ≈ closed-form function of F, K, and SABR parameters
    
    Parameters:
    - F: forward rate or price
    - α: stochastic volatility (ATM volatility scale)
    - β: elasticity parameter (0 ≤ β ≤ 1)
      β = 0: normal model
      β = 1: lognormal model
    - ν: volatility of volatility
    - ρ: correlation between price and volatility
    """
    
    def __init__(
        self,
        forward: float,
        alpha: float,
        beta: float,
        nu: float,
        rho: float
    ):
        """
        Initialize SABR model parameters.
        
        Args:
            forward: F, forward rate/price
            alpha: α, initial volatility level
            beta: β, elasticity (typically 0.5 or 1.0)
            nu: ν, volatility of volatility
            rho: ρ, correlation between processes (-1 to 1)
        """
        if not (0 <= beta <= 1):
            raise ValueError("beta must be between 0 and 1")
        if not (-1 <= rho <= 1):
            raise ValueError("rho must be between -1 and 1")
        
        self.F = forward
        self.alpha = alpha
        self.beta = beta
        self.nu = nu
        self.rho = rho
    
    def implied_volatility(
        self,
        strike: float,
        maturity: float,
        method: str = "hagan"
    ) -> float:
        """
        Calculate implied volatility smile using SABR approximation.
        
        Args:
            strike: K, option strike
            maturity: T, time to maturity (in years)
            method: "hagan" (primary) or "simple" (zeroth order)
            
        Returns:
            Implied volatility at the given strike
        """
        if strike <= 0 or self.F <= 0:
            raise ValueError("Strike and forward must be positive")
        
        if method == "hagan":
            return self._hagan_approximation(strike, maturity)
        elif method == "simple":
            return self._simple_approximation(strike, maturity)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _hagan_approximation(self, strike: float, maturity: float) -> float:
        """
        Hagan et al. (2002) approximation for implied volatility.
        This is the most widely used SABR formula.
        """
        F = self.F
        K = strike
        T = maturity
        alpha = self.alpha
        beta = self.beta
        nu = self.nu
        rho = self.rho
        
        # Avoid division by zero
        if abs(F - K) < 1e-10:
            # ATM case
            numerator = alpha * (1 + (
                (1 - beta)**2 / 24 * (alpha / F)**(2 - beta) * T
                + rho * beta * nu * T / 4
                + ((2 - 3 * rho**2) / 24) * nu**2 * T
            ))
            return numerator / F**(1 - beta)
        
        # OTM case
        # Compute FK term
        FK = F * K
        FK_sqrt = np.sqrt(FK)
        
        # z and x parameters
        z = nu / alpha * FK_sqrt**(1 - beta) * np.log(F / K)
        x = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
        
        # Numerator
        numerator = alpha * z / x
        
        # Denominator correction factor
        denominator_correction = FK_sqrt**(1 - beta) * (
            1 + ((1 - beta)**2 / 24) * (np.log(F / K))**2
            + ((1 - beta)**4 / 1920) * (np.log(F / K))**4
        )
        
        # First correction factor
        first_correction = 1 + (
            (1 - beta)**2 / 24 * (alpha / FK_sqrt)**(2 - beta)
            + rho * beta * nu * alpha / (4 * FK_sqrt**(1 - beta))
            + ((2 - 3 * rho**2) / 24) * nu**2
        ) * T
        
        sigma_impl = numerator / denominator_correction * first_correction
        
        return sigma_impl
    
    def _simple_approximation(self, strike: float, maturity: float) -> float:
        """
        Simple (zeroth order) approximation - ATM volatility.
        """
        return self.alpha / (self.F**(1 - self.beta))
    
    def volatility_smile(self, strikes: np.ndarray, maturity: float) -> np.ndarray:
        """
        Compute the entire volatility smile for a given maturity.
        
        Args:
            strikes: Array of strikes
            maturity: Time to maturity (in years)
            
        Returns:
            Array of implied volatilities
        """
        return np.array([self.implied_volatility(K, maturity) for K in strikes])
    
    def atm_volatility(self, maturity: float) -> float:
        """
        At-the-money volatility.
        
        Args:
            maturity: Time to maturity (in years)
            
        Returns:
            ATM implied volatility
        """
        return self.implied_volatility(self.F, maturity)
