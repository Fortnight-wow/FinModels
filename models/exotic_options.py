import numpy as np
from typing import Callable, Tuple


class ExoticOptions:
    """
    Monte Carlo pricing for exotic options.
    
    Provides pricing methods for path-dependent options that don't have
    closed-form solutions: Asian, Barrier, and Lookback options.
    """
    
    @staticmethod
    def asian_option(
        paths: np.ndarray,
        strike: float,
        option_type: str = "call",
        averaging: str = "arithmetic",
        discount_factor: float = 1.0
    ) -> Tuple[float, float]:
        """
        Price Asian option (average price option).
        
        Payoff (Call): max(A - K, 0) where A is average price
        Payoff (Put): max(K - A, 0)
        
        Args:
            paths: Array of shape (steps, paths) with simulated prices
            strike: K, strike price
            option_type: "call" or "put"
            averaging: "arithmetic" or "geometric"
            discount_factor: Discount factor (e.g., exp(-r*T))
            
        Returns:
            Tuple of (option price, standard error)
        """
        if averaging == "arithmetic":
            averages = np.mean(paths, axis=0)
        elif averaging == "geometric":
            averages = np.exp(np.mean(np.log(paths), axis=0))
        else:
            raise ValueError("averaging must be 'arithmetic' or 'geometric'")
        
        if option_type == "call":
            payoffs = np.maximum(averages - strike, 0)
        elif option_type == "put":
            payoffs = np.maximum(strike - averages, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        price = discount_factor * np.mean(payoffs)
        std_error = discount_factor * np.std(payoffs) / np.sqrt(len(payoffs))
        
        return price, std_error
    
    @staticmethod
    def barrier_option(
        paths: np.ndarray,
        strike: float,
        barrier: float,
        option_type: str = "call",
        barrier_type: str = "knock_out",
        barrier_side: str = "up",
        discount_factor: float = 1.0,
        rebate: float = 0.0
    ) -> Tuple[float, float]:
        """
        Price Barrier option (knock-in or knock-out).
        
        Knock-out: Option becomes void if barrier is hit
        Knock-in: Option becomes active only if barrier is hit
        
        Args:
            paths: Array of shape (steps, paths) with simulated prices
            strike: K, strike price
            barrier: H, barrier level
            option_type: "call" or "put"
            barrier_type: "knock_out" or "knock_in"
            barrier_side: "up" (up-and-out/in) or "down" (down-and-out/in)
            discount_factor: Discount factor
            rebate: Rebate paid if option knocked out
            
        Returns:
            Tuple of (option price, standard error)
        """
        num_paths = paths.shape[1]
        final_prices = paths[-1, :]
        
        if option_type == "call":
            payoffs = np.maximum(final_prices - strike, 0)
        elif option_type == "put":
            payoffs = np.maximum(strike - final_prices, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        # Check barrier conditions
        if barrier_side == "up":
            barrier_hit = np.max(paths, axis=0) >= barrier
        elif barrier_side == "down":
            barrier_hit = np.min(paths, axis=0) <= barrier
        else:
            raise ValueError("barrier_side must be 'up' or 'down'")
        
        # Apply barrier logic
        if barrier_type == "knock_out":
            payoffs[barrier_hit] = rebate
        elif barrier_type == "knock_in":
            payoffs[~barrier_hit] = 0
        else:
            raise ValueError("barrier_type must be 'knock_out' or 'knock_in'")
        
        price = discount_factor * np.mean(payoffs)
        std_error = discount_factor * np.std(payoffs) / np.sqrt(num_paths)
        
        return price, std_error
    
    @staticmethod
    def lookback_option(
        paths: np.ndarray,
        strike: float = None,
        option_type: str = "call",
        lookback_type: str = "fixed",
        discount_factor: float = 1.0
    ) -> Tuple[float, float]:
        """
        Price Lookback option (depends on minimum/maximum price during path).
        
        Fixed strike: Payoff depends on running max/min
        Floating strike: Strike is running min/max
        
        Args:
            paths: Array of shape (steps, paths) with simulated prices
            strike: K, strike price (required for fixed strike)
            option_type: "call" or "put"
            lookback_type: "fixed" (fixed strike) or "floating" (floating strike)
            discount_factor: Discount factor
            
        Returns:
            Tuple of (option price, standard error)
        """
        num_paths = paths.shape[1]
        final_prices = paths[-1, :]
        
        max_prices = np.max(paths, axis=0)
        min_prices = np.min(paths, axis=0)
        
        if lookback_type == "fixed":
            if strike is None:
                raise ValueError("strike required for fixed strike lookback")
            
            if option_type == "call":
                payoffs = np.maximum(max_prices - strike, 0)
            elif option_type == "put":
                payoffs = np.maximum(strike - min_prices, 0)
            else:
                raise ValueError("option_type must be 'call' or 'put'")
        
        elif lookback_type == "floating":
            if option_type == "call":
                payoffs = final_prices - min_prices
            elif option_type == "put":
                payoffs = max_prices - final_prices
            else:
                raise ValueError("option_type must be 'call' or 'put'")
        
        else:
            raise ValueError("lookback_type must be 'fixed' or 'floating'")
        
        price = discount_factor * np.mean(payoffs)
        std_error = discount_factor * np.std(payoffs) / np.sqrt(num_paths)
        
        return price, std_error
