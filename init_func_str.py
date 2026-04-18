"""
Initial function strings for LLM ODE evolution.

This module provides initial function string templates for different dimensions.
It generates **Fully Coupled Linear Systems** as initial guesses, ensuring that
the starting point is already treated as a system pair (or tuple).
Uses terms_to_function_code from utils.py for consistency with term-based architecture.
"""

from typing import List

from utils import terms_to_function_code


def generate_init_func_str(dim: int, max_params: int = 8) -> List[str]:
    """Generate initial function strings for given dimension.
    
    Uses terms_to_function_code to generate code consistent with term-based architecture.
    Initial terms use params-included format: params[0]*x0, params[1]*x1, ...
    
    Args:
        dim: Dimension (1, 2, 3, or 4)
        max_params: Maximum number of parameters
        
    Returns:
        List of function strings, one per x{i}_t function
        
    Example:
        For dim=2, max_params=4:
        [
            "def x0_t(x0, x1, params):\\n    import numpy as np\\n    return (params[0]*x0) + (params[1]*x1) + params[2] * 1",
            "def x1_t(x0, x1, params):\\n    import numpy as np\\n    return (params[0]*x0) + (params[1]*x1) + params[2] * 1"
        ]
    """
    # Generate params-included linear terms: params[0]*x0, params[1]*x1, ...
    terms = [f"params[{i}]*x{i}" for i in range(dim)]
    
    func_list = []
    for i in range(dim):
        func_name = f"x{i}_t"
        code, _ = terms_to_function_code(terms, func_name, dim, max_params)
        func_list.append(code)
    
    return func_list


# Backward compatibility: pre-generate for common dimensions
init_func_str_1D = generate_init_func_str(1, max_params=8)
init_func_str_2D = generate_init_func_str(2, max_params=8)
init_func_str_3D = generate_init_func_str(3, max_params=8)
init_func_str_4D = generate_init_func_str(4, max_params=8)

