import numpy as np
from typing import Dict, Tuple
from scipy.stats import linregress


class CAPM:
    """
    Capital Asset Pricing Model (CAPM).
    
    The CAPM relates expected return of an asset to its systematic risk:
    E[R_i] = R_f + β_i * (E[R_m] - R_f)
    
    where:
    - R_i: return of asset i
    - R_f: risk-free rate
    - β_i: beta (systematic risk relative to market)
    - R_m: market return
    - (E[R_m] - R_f): market risk premium
    
    Beta is calculated as:
    β_i = Cov(R_i, R_m) / Var(R_m)
    """
    
    def __init__(
        self,
        asset_returns: np.ndarray,
        market_returns: np.ndarray,
        risk_free_rate: float
    ):
        """
        Initialize CAPM with return series.
        
        Args:
            asset_returns: Array of asset returns
            market_returns: Array of market returns (same length as asset_returns)
            risk_free_rate: Risk-free rate (annualized)
        """
        if len(asset_returns) != len(market_returns):
            raise ValueError("Asset and market returns must have same length")
        
        self.asset_returns = np.asarray(asset_returns)
        self.market_returns = np.asarray(market_returns)
        self.rf = risk_free_rate
    
    def calculate_beta(self) -> float:
        """
        Calculate asset beta (systematic risk).
        
        Beta = Cov(Asset, Market) / Var(Market)
        
        Returns:
            Beta coefficient
        """
        covariance = np.cov(self.asset_returns, self.market_returns)[0, 1]
        market_variance = np.var(self.market_returns)
        
        if market_variance == 0:
            return 0
        
        beta = covariance / market_variance
        return beta
    
    def calculate_alpha(self) -> float:
        """
        Calculate Jensen's alpha (excess return not explained by beta).
        
        Alpha = E[R_i] - (R_f + β_i * (E[R_m] - R_f))
        
        Returns:
            Alpha (annualized if inputs are annualized)
        """
        expected_asset_return = np.mean(self.asset_returns)
        expected_market_return = np.mean(self.market_returns)
        beta = self.calculate_beta()
        
        capm_return = self.rf + beta * (expected_market_return - self.rf)
        alpha = expected_asset_return - capm_return
        
        return alpha
    
    def capm_return(self, beta: float = None) -> float:
        """
        Calculate expected return according to CAPM.
        
        Args:
            beta: Beta coefficient (calculated if None)
            
        Returns:
            Expected return according to CAPM
        """
        if beta is None:
            beta = self.calculate_beta()
        
        market_risk_premium = np.mean(self.market_returns) - self.rf
        expected_return = self.rf + beta * market_risk_premium
        
        return expected_return
    
    def regression_stats(self) -> Dict[str, float]:
        """
        Perform linear regression of asset returns on market returns.
        Returns both alpha and beta with statistical information.
        
        Returns:
            Dictionary with slope (beta), intercept (alpha), R-squared, p-value
        """
        slope, intercept, r_value, p_value, std_err = linregress(
            self.market_returns,
            self.asset_returns
        )
        
        return {
            'beta': slope,
            'alpha': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_error': std_err
        }
    
    def tracking_error(self, portfolio_returns: np.ndarray) -> float:
        """
        Calculate tracking error (volatility of excess returns).
        
        Args:
            portfolio_returns: Array of portfolio returns
            
        Returns:
            Tracking error (standard deviation of excess returns)
        """
        excess_returns = portfolio_returns - self.market_returns
        return np.std(excess_returns)
    
    def information_ratio(self, portfolio_returns: np.ndarray) -> float:
        """
        Calculate information ratio (excess return per unit of tracking error).
        
        Args:
            portfolio_returns: Array of portfolio returns
            
        Returns:
            Information ratio
        """
        excess_return = np.mean(portfolio_returns) - np.mean(self.market_returns)
        tracking_error = self.tracking_error(portfolio_returns)
        
        if tracking_error == 0:
            return 0
        
        return excess_return / tracking_error
