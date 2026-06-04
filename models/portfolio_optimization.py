import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from typing import Tuple, Dict


class Markowitz:
    """
    Markowitz Mean-Variance Portfolio Optimization.
    
    Finds optimal portfolio weights that minimize variance for a target return,
    or maximize Sharpe ratio (return per unit of risk).
    
    Mathematical formulation:
    min: w'Σw
    s.t.: w'μ = target_return
          Σw_i = 1
          w_i ≥ 0 (or allow short selling)
    """
    
    def __init__(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_free_rate: float = 0.0
    ):
        """
        Initialize portfolio optimizer.
        
        Args:
            expected_returns: Array of expected returns for each asset
            covariance_matrix: Covariance matrix of asset returns
            risk_free_rate: Risk-free rate for Sharpe ratio calculations
        """
        self.mu = expected_returns
        self.sigma = covariance_matrix
        self.rf = risk_free_rate
        self.n_assets = len(expected_returns)
        
        if covariance_matrix.shape != (self.n_assets, self.n_assets):
            raise ValueError("Covariance matrix shape doesn't match returns")
    
    def min_variance_portfolio(self) -> Tuple[np.ndarray, float, float]:
        """
        Find the Global Minimum Variance Portfolio (MVP).
        
        Returns:
            Tuple of (optimal weights, expected return, variance)
        """
        def portfolio_variance(w):
            return w @ self.sigma @ w
        
        constraints = {
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1  # Weights sum to 1
        }
        
        bounds = Bounds(0, 1)  # No short selling
        initial_guess = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(
            portfolio_variance,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        w_opt = result.x
        p_return = w_opt @ self.mu
        p_variance = w_opt @ self.sigma @ w_opt
        
        return w_opt, p_return, p_variance
    
    def max_sharpe_portfolio(
        self,
        allow_short_selling: bool = False
    ) -> Tuple[np.ndarray, float, float, float]:
        """
        Find the Maximum Sharpe Ratio Portfolio.
        
        Args:
            allow_short_selling: If True, allow negative weights (short selling)
            
        Returns:
            Tuple of (optimal weights, expected return, std deviation, Sharpe ratio)
        """
        def negative_sharpe_ratio(w):
            p_return = w @ self.mu
            p_std = np.sqrt(w @ self.sigma @ w)
            if p_std == 0:
                return 0
            return -(p_return - self.rf) / p_std
        
        constraints = {
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1  # Weights sum to 1
        }
        
        if allow_short_selling:
            bounds = None  # Allow any weights
        else:
            bounds = Bounds(0, 1)  # No short selling
        
        initial_guess = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(
            negative_sharpe_ratio,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        w_opt = result.x
        p_return = w_opt @ self.mu
        p_std = np.sqrt(w_opt @ self.sigma @ w_opt)
        sharpe = (p_return - self.rf) / p_std if p_std > 0 else 0
        
        return w_opt, p_return, p_std, sharpe
    
    def efficient_frontier(
        self,
        target_returns: np.ndarray,
        allow_short_selling: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Compute the efficient frontier by finding minimum variance portfolios
        for each target return.
        
        Args:
            target_returns: Array of target returns
            allow_short_selling: If True, allow short selling
            
        Returns:
            Dictionary with 'returns', 'stds', and 'weights' arrays
        """
        frontier_returns = []
        frontier_stds = []
        frontier_weights = []
        
        for target_ret in target_returns:
            def portfolio_variance(w):
                return w @ self.sigma @ w
            
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: w @ self.mu - target_ret}
            ]
            
            if allow_short_selling:
                bounds = None
            else:
                bounds = Bounds(0, 1)
            
            initial_guess = np.ones(self.n_assets) / self.n_assets
            
            result = minimize(
                portfolio_variance,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                w = result.x
                frontier_weights.append(w)
                frontier_returns.append(target_ret)
                frontier_stds.append(np.sqrt(w @ self.sigma @ w))
        
        return {
            'returns': np.array(frontier_returns),
            'stds': np.array(frontier_stds),
            'weights': np.array(frontier_weights)
        }
    
    def portfolio_stats(self, weights: np.ndarray) -> Dict[str, float]:
        """
        Calculate statistics for a given portfolio.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Dictionary with return, std, variance, and Sharpe ratio
        """
        p_return = weights @ self.mu
        p_variance = weights @ self.sigma @ weights
        p_std = np.sqrt(p_variance)
        sharpe = (p_return - self.rf) / p_std if p_std > 0 else 0
        
        return {
            'return': p_return,
            'std': p_std,
            'variance': p_variance,
            'sharpe': sharpe
        }
