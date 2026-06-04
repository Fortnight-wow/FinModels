import numpy as np
from typing import Dict, Tuple
from scipy.stats import norm


class RiskMetrics:
    """
    Portfolio risk metrics: VaR, CVaR (Expected Shortfall), and stress testing.
    
    VaR: Value at Risk - maximum loss at a given confidence level
    CVaR: Conditional Value at Risk (Expected Shortfall) - average loss beyond VaR
    """
    
    @staticmethod
    def value_at_risk(
        returns: np.ndarray,
        confidence: float = 0.95,
        method: str = "historical"
    ) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Array of historical or simulated returns
            confidence: Confidence level (e.g., 0.95 for 95%)
            method: "historical", "parametric", or "cornish_fisher"
            
        Returns:
            VaR as a negative number (loss magnitude)
        """
        if method == "historical":
            return np.percentile(returns, (1 - confidence) * 100)
        
        elif method == "parametric":
            # Assumes normal distribution
            mu = np.mean(returns)
            sigma = np.std(returns)
            return mu + sigma * norm.ppf(1 - confidence)
        
        elif method == "cornish_fisher":
            # Adjusted for skewness and kurtosis
            mu = np.mean(returns)
            sigma = np.std(returns)
            skew = (np.mean((returns - mu)**3)) / sigma**3
            kurt = (np.mean((returns - mu)**4)) / sigma**4
            
            z = norm.ppf(1 - confidence)
            z_cf = z + (z**2 - 1) * skew / 6 + (z**3 - 3*z) * (kurt - 3) / 24
            
            return mu + sigma * z_cf
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def conditional_value_at_risk(
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR / Expected Shortfall).
        Average loss beyond the VaR threshold.
        
        Args:
            returns: Array of historical or simulated returns
            confidence: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            CVaR as a negative number (average loss magnitude)
        """
        var = RiskMetrics.value_at_risk(returns, confidence, method="historical")
        cvar = np.mean(returns[returns <= var])
        return cvar
    
    @staticmethod
    def portfolio_risk_metrics(
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics for a portfolio.
        
        Args:
            returns: Array of portfolio returns
            confidence: Confidence level
            
        Returns:
            Dictionary with VaR, CVaR, volatility, Sharpe ratio, etc.
        """
        var = RiskMetrics.value_at_risk(returns, confidence)
        cvar = RiskMetrics.conditional_value_at_risk(returns, confidence)
        
        return {
            'mean': np.mean(returns),
            'std': np.std(returns),
            'min': np.min(returns),
            'max': np.max(returns),
            'skewness': (np.mean((returns - np.mean(returns))**3)) / np.std(returns)**3,
            'kurtosis': (np.mean((returns - np.mean(returns))**4)) / np.std(returns)**4,
            'var': var,
            'cvar': cvar,
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        }
    
    @staticmethod
    def stress_test(
        base_prices: np.ndarray,
        shock_scenarios: Dict[str, float],
        correlations: np.ndarray = None
    ) -> Dict[str, np.ndarray]:
        """
        Perform stress testing by applying shocks to asset prices.
        
        Args:
            base_prices: Base prices of assets
            shock_scenarios: Dict mapping scenario name to shock magnitude
            correlations: Correlation matrix for correlated shocks
            
        Returns:
            Dictionary mapping scenario names to shocked prices
        """
        stressed_prices = {}
        
        for scenario_name, shock in shock_scenarios.items():
            if correlations is not None:
                # Apply correlated shocks
                num_assets = len(base_prices)
                independent_shocks = np.random.normal(shock, abs(shock) * 0.1, num_assets)
                correlated_shocks = correlations @ independent_shocks
                stressed_prices[scenario_name] = base_prices * (1 + correlated_shocks)
            else:
                # Independent shocks
                stressed_prices[scenario_name] = base_prices * (1 + shock)
        
        return stressed_prices
    
    @staticmethod
    def drawdown_analysis(prices: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        Calculate maximum drawdown and drawdown series.
        
        Args:
            prices: Array of prices over time
            
        Returns:
            Tuple of (max_drawdown, duration, drawdown_series)
        """
        cumulative_max = np.maximum.accumulate(prices)
        drawdown = (prices - cumulative_max) / cumulative_max
        max_drawdown = np.min(drawdown)
        
        # Calculate duration of maximum drawdown
        max_dd_idx = np.argmin(drawdown)
        peak_idx = np.argmax(prices[:max_dd_idx])
        duration = max_dd_idx - peak_idx
        
        return max_drawdown, duration, drawdown
    
    @staticmethod
    def rolling_volatility(
        returns: np.ndarray,
        window: int = 20
    ) -> np.ndarray:
        """
        Calculate rolling volatility (moving standard deviation).
        
        Args:
            returns: Array of returns
            window: Rolling window size
            
        Returns:
            Array of rolling volatilities
        """
        rolling_vols = np.zeros(len(returns) - window + 1)
        
        for i in range(len(rolling_vols)):
            rolling_vols[i] = np.std(returns[i:i+window])
        
        return rolling_vols
