"""
Optimization functions for LLM ODE evolution.

This module contains loss function creation and parameter optimization logic.
"""

from typing import Callable, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

from config import (
    PENALTY_VALUE,
    MAX_OPTIMIZATION_ITERATIONS,
    BFGS_TOLERANCE,
    DE_TOLERANCE,
    DIFFERENTIAL_EVOLUTION_CONFIG,
    DE_BOUNDS,
    MAX_PARAMS,
)
from utils import get_x_arrays_for_function, get_function_args








def create_loss_function(
    func: Callable, 
    df: pd.DataFrame, 
    y_true: np.ndarray,
) -> Callable[[np.ndarray], float]:
    """Create MSE loss function for a given function (NumPy version).
    
    Args:
        func: The function to evaluate
        df: DataFrame containing x variable data
        y_true: Ground truth array
        
    Returns:
        Loss function that takes params array and returns MSE loss
    """
    def loss_function(params: np.ndarray) -> float:
        try:
            input_args = get_x_arrays_for_function(func, df, params)
            y_pred = func(*input_args)
            # NaN/Inf check
            if np.isnan(y_pred).any() or np.isinf(y_pred).any():
                return PENALTY_VALUE
            loss = np.mean((y_pred - y_true) ** 2)
            return float(loss)
        except:
            return PENALTY_VALUE
    
    return loss_function


def run_bfgs_optimization(
    func: Callable,
    df: pd.DataFrame,
    y_true: np.ndarray,
    initial_params: np.ndarray,
    max_iterations: int = MAX_OPTIMIZATION_ITERATIONS,
    tol: float = None,
) -> np.ndarray:
    """Run Scipy BFGS optimization for a single function.
    
    Args:
        func: Function to optimize parameters for
        df: DataFrame containing x variable data  
        y_true: Ground truth array
        initial_params: Initial parameter array
        max_iterations: Maximum number of optimization iterations
        tol: Convergence tolerance (if None, uses config.BFGS_TOLERANCE)
        
    Returns:
        Optimized parameter array
    """
    if tol is None:
        tol = BFGS_TOLERANCE

    def objective(params):
        loss = create_loss_function(func, df, y_true)(params)
        return loss

    try:
        # Use Scipy Minimize with BFGS
        result = minimize(
            objective, 
            initial_params, 
            method='BFGS',
            options={'maxiter': max_iterations, 'gtol': tol}
        )
        optimized_params = result.x
    except Exception as e:
        print(f"  Optimization error: {e}")
        optimized_params = initial_params

    # Return optimized parameters directly (no rounding)
    return optimized_params



def run_differential_evolution(
    func: Callable,
    df: pd.DataFrame,
    y_true: np.ndarray,
    max_params: int = None,
    tol: float = None,
) -> tuple[np.ndarray, float]:
    """Run Differential Evolution to find good initial parameters.
    
    Args:
        func: Function to optimize
        df: DataFrame containing x variable data
        y_true: Ground truth array
        max_params: Number of parameters to optimize (defaults to config.MAX_PARAMS)
        tol: Tolerance (defaults to config.DE_TOLERANCE)
        
    Returns:
        Tuple of (optimized parameter array, score)
    """
    n_params = max_params if max_params is not None else MAX_PARAMS
    
    current_de_config = DIFFERENTIAL_EVOLUTION_CONFIG.copy()
    if tol is not None:
        current_de_config['tol'] = tol
        current_de_config['atol'] = tol
    
    # Must use workers=1 when called from multiprocessing.Pool (daemon limitation)
    current_de_config['workers'] = 1
    
    bounds = [DE_BOUNDS] * n_params
    
    def objective(params):
        return create_loss_function(func, df, y_true)(params)

    try:
        result = differential_evolution(objective, bounds, **current_de_config)
        return result.x, result.fun
    except Exception as e:
        print(f"  DE error: {e}")
        return np.ones(n_params), float('inf')


def calculate_scores(
    func_list: List[Callable],
    params_list: List[np.ndarray],
    df_dict: dict,
    dt_cols: List[str]
) -> dict:
    """Calculate MSE scores for all functions across all datasets.
    
    Args:
        func_list: List of callable functions
        params_list: List of parameter arrays for each function
        df_dict: Dictionary of DataFrames ('train', 'test_id', 'test_ood')
        dt_cols: List of dt column names
        
    Returns:
        Dictionary with keys 'train', 'test_id', 'test_ood' and score lists as values
    """
    score_dict = {}
    
    for key, df in df_dict.items():
        score_list = []
        for idx, func in enumerate(func_list):
            try:
                # Handle dt_cols index fallback
                dt_col_idx = idx if idx < len(dt_cols) else 0
                y_true = df[dt_cols[dt_col_idx]].values.astype(np.float32)
                
                # Handle params_list index fallback
                params_idx = idx if idx < len(params_list) else 0
                params = params_list[params_idx]
                
                # Get input arguments and compute prediction
                input_args = get_x_arrays_for_function(func, df, params)
                y_pred = func(*input_args)
                
                # Check for NaN/Inf
                if np.isnan(y_pred).any() or np.isinf(y_pred).any():
                    score = PENALTY_VALUE
                else:
                    # Calculate MSE
                    mse = np.mean((y_pred - y_true) ** 2)
                    score = float(mse)
                
                score_list.append(score)
            except (IndexError, RuntimeError, ValueError, TypeError) as e:
                print(f"Error occurred in calculate_scores for function {idx}: {e}")
                score_list.append(1e10)
        
        score_dict[key] = score_list
    
    return score_dict



