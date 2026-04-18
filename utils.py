"""
Utility functions for LLM ODE evolution.

This module contains common utility functions used across the codebase.
"""

import re
import inspect
from typing import List, Callable, Optional, Dict

import numpy as np
import pandas as pd

from config import PARAMS_PRECISION, PARAMS_ZERO_THRESHOLD


_PARAM_INDEX_PATTERN = re.compile(r'params\[(\d+)\]')


def _param_indices(text: str) -> List[int]:
    """Return params[N] indices found in a term-like string."""
    return [int(match) for match in _PARAM_INDEX_PATTERN.findall(text)]


def _max_param_index(terms: List[str]) -> int:
    """Return the highest params[N] index in terms, or -1 when absent."""
    max_param_idx = -1
    for term in terms:
        term_max = max(_param_indices(term), default=-1)
        max_param_idx = max(max_param_idx, term_max)
    return max_param_idx



def clean_code_string(code: str) -> str:
    """Remove outer wrapping (code fences/quotes etc.) from LLM-returned code.
    
    Args:
        code: Raw code string potentially with markdown code fences or quotes
        
    Returns:
        Cleaned code string
    """
    if code is None:
        return ""
    s = str(code).strip()
    # Remove ```python ... ``` or ``` ... ```
    s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    # Remove outer triple quotes
    s = re.sub(r"^\s*(?:'''|\"\"\")", "", s)
    s = re.sub(r"(?:'''|\"\"\")\\s*$", "", s)
    # Remove outer single/double quotes (one layer)
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        s = s[1:-1]
    return s.strip()


def make_function_from_code(parsed_code: str, function_name: str) -> Callable:
    """Convert Python code string to actual function.
    
    Args:
        parsed_code: Python code string containing the function definition
        function_name: Name of the function to extract (e.g., 'dt_x0')
        
    Returns:
        The extracted callable function
    """
    code = clean_code_string(parsed_code)
    local_vars = {}
    exec(code, globals(), local_vars)
    func = local_vars.get(function_name)
    if func is None:
        raise ValueError(f"Could not find '{function_name}' in code: {parsed_code}")
    return func



def get_x_cols(df: pd.DataFrame) -> List[str]:
    """Find columns matching x{n} pattern.
    
    Args:
        df: DataFrame to search for x columns
        
    Returns:
        List of column names matching x{n} pattern (e.g., ['x0', 'x1', 'x2'])
    """
    return [col for col in df.columns if re.fullmatch(r'x\d+', col)]


def get_func_names(df: pd.DataFrame) -> List[str]:
    """Get function names (always x{n}_t pattern, regardless of use_gt).
    
    Args:
        df: DataFrame to search for x columns
        
    Returns:
        List of function names (e.g., ['x0_t', 'x1_t'])
    """
    x_cols = get_x_cols(df)
    return sorted([f"{col}_t" for col in x_cols])


def get_target_cols(df: pd.DataFrame, use_gt: bool = False) -> List[str]:
    """Get target column names for fitting (x{n}_t or x{n}_t_gt).
    
    Args:
        df: DataFrame to search for target columns
        use_gt: If True, return x{n}_t_gt columns (ground truth).
                If False, return x{n}_t columns (gradient-based).
        
    Returns:
        List of target column names
    """
    if use_gt:
        return sorted([col for col in df.columns if re.fullmatch(r'x\d+_t_gt', col)])
    else:
        return sorted([col for col in df.columns if re.fullmatch(r'x\d+_t', col)])


def get_function_args(func: Callable) -> List[str]:
    """Analyze function signature and return list of required x variable names.
    
    Args:
        func: Function to analyze
        
    Returns:
        List of parameter names from the function signature
    """
    sig = inspect.signature(func)
    return list(sig.parameters.keys())


def get_x_arrays_for_function(func: Callable, df: pd.DataFrame, params: np.ndarray) -> List[np.ndarray]:
    """Return list of arrays corresponding to x variables required by function.
    
    Args:
        func: Function whose signature determines which x variables are needed
        df: DataFrame containing the x variable data
        params: Parameters (numpy array)
        
    Returns:
        List of arrays for each required x variable (including params)
    """
    args = get_function_args(func)
    args_arrays = []
    for arg in args:
        if arg != 'params':
            args_arrays.append(df[arg].values)
        else:
            args_arrays.append(params)
    return args_arrays


def create_functions_from_codes(
    code_str_list: List[str], 
    func_names: List[str]
) -> tuple[List[Optional[Callable]], bool]:
    """Create function list from code strings and return syntax error status.
    
    Args:
        code_str_list: List of Python code strings
        func_names: List of function names (e.g., ['x0_t', 'x1_t', ...])
    
    Returns:
        Tuple of (function_list, syntax_error_occurred)
    """
    func_list = []
    for idx, code_str in enumerate(code_str_list):
        func = make_function_from_code(code_str, func_names[idx])
        func_list.append(func)
    return func_list

def round_params(params: np.ndarray, precision: int = PARAMS_PRECISION, zero_threshold: float = PARAMS_ZERO_THRESHOLD) -> np.ndarray:
    """Round parameters and set small values to zero.
    
    Args:
        params: Parameter array to process
        precision: Number of decimal places for rounding
        zero_threshold: Values with absolute value below this are set to 0
        
    Returns:
        Processed parameter array
    """
    params_array = params.copy()
    params_array[np.abs(params_array) < zero_threshold] = 0
    return np.round(params_array, precision)


def remap_param_indices(terms: List[str]) -> tuple[List[str], dict]:
    """
    Remap params indices to start from 0 consecutively.
    
    Args:
        terms: List of term strings with arbitrary params indices
        
    Returns:
        (remapped_terms, index_mapping) tuple
        - remapped_terms: Terms with params[0], params[1], ...
        - index_mapping: Dict mapping old_idx -> new_idx
    """
    # 1. Collect all params indices used
    all_indices = {idx for term in terms for idx in _param_indices(term)}
    
    # If no indices found or already starts from 0, no remapping needed
    if not all_indices:
        return terms, {}
    
    # 2. Create mapping from sorted indices to consecutive 0-based indices
    sorted_indices = sorted(all_indices)
    index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}
    
    # If already 0-based consecutive, no remapping needed
    if sorted_indices == list(range(len(sorted_indices))):
        return terms, {}
    
    # 3. Replace all params indices in terms
    remapped_terms = []
    for term in terms:
        new_term = term
        # Replace from highest to lowest index to avoid conflicts
        for old_idx in sorted(index_mapping.keys(), reverse=True):
            new_idx = index_mapping[old_idx]
            new_term = re.sub(
                rf'params\[{old_idx}\]',
                f'params[{new_idx}]',
                new_term
            )
        remapped_terms.append(new_term)
    
    return remapped_terms, index_mapping


def convert_C_to_params(term: str, start_idx: int) -> tuple[str, int]:
    """Convert 'C' placeholders to params[N] in term.
    
    Converts 'C' placeholders in skeleton-format terms generated by LLM to params[N].
    Each 'C' is mapped to a unique params index.
    
    Args:
        term: Term string potentially containing 'C' (e.g., "C*x0", "C*np.sin(C*x0)")
        start_idx: Starting index for params (after existing params indices)
    
    Returns:
        (converted_term, next_available_idx): Converted term and next available index
        
    Examples:
        convert_C_to_params("C*x0", 0) -> ("params[0]*x0", 1)
        convert_C_to_params("C*np.sin(C*x0)", 0) -> ("params[0]*np.sin(params[1]*x0)", 2)
        convert_C_to_params("params[0]*x0 + C*x1", 1) -> ("params[0]*x0 + params[1]*x1", 2)
    """
    next_idx = start_idx
    result = term
    
    # Replace each 'C' with params[N], ensuring word boundaries
    # Pattern: 'C' that is not part of a larger identifier (e.g., not "cos", "C1", etc.)
    pattern = r'\bC\b'
    
    def replace_C(match):
        nonlocal next_idx
        idx = next_idx
        next_idx += 1
        return f'params[{idx}]'
    
    result = re.sub(pattern, replace_C, result)
    return result, next_idx


def terms_to_function_code(
    terms: List[str],
    func_name: str,
    dim: int,
    max_params: int
) -> tuple[str, int]:
    """Convert term list to Python function code.
    
    Terms already contain params[i], so we just combine them.
    A constant term (last used param index + 1) is automatically added at the end.
    Auto-remapping ensures consecutive indices; returns the actual number of params required
    so that the caller can allocate an array of that size (exceeding MAX_PARAMS is allowed).
    
    Args:
        terms: List of term strings with params (e.g., ["params[0]*sin(params[1]*x0)", "params[2]*(1+x1)"])
        func_name: Function name (e.g., "x0_t")
        dim: Dimension for function signature (determines x0, x1, ...)
        max_params: For validation only; returned n_params_required can exceed this.
        
    Returns:
        (code, n_params_required): Python function code and the params array length needed.
        
    Example:
        Input: terms=["params[0]*sin(params[1]*x0)", "params[2]*(1+x1)"], func_name="x0_t", dim=2
        Output:
            def x0_t(x0, x1, params):
                import numpy as np
                return (params[0]*sin(params[1]*x0)) + (params[2]*(1+x1)) + params[3] * 1
    """
    # Auto-remap params indices to start from 0 consecutively
    terms, index_mapping = remap_param_indices(terms)
    
    # Log remapping if it occurred
    if index_mapping and (not index_mapping or min(index_mapping.keys()) != 0):
        print(f"[Auto-remap] {func_name}: {index_mapping}")
    
    # Find the highest params index used in terms (before C conversion)
    max_param_idx = _max_param_index(terms)
    
    # Convert 'C' placeholders to params[N] (after remapping, before numpy prefix fix)
    # Start from next available index after existing params
    converted_terms = []
    next_C_idx = max_param_idx + 1
    
    for term in terms:
        # Check if term contains 'C'
        if re.search(r'\bC\b', term):
            converted_term, next_C_idx = convert_C_to_params(term, next_C_idx)
            converted_terms.append(converted_term)
        else:
            converted_terms.append(term)
    
    terms = converted_terms
    
    # Recalculate max_param_idx after C conversion to ensure accuracy
    max_param_idx = _max_param_index(terms)
    
    # Auto-fix: Add np. prefix to common numpy functions if missing
    numpy_funcs = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'tanh', 'sinh', 'cosh', 'arcsin', 'arccos', 'arctan']
    
    def fix_numpy_prefix(term: str) -> str:
        """Add np. prefix to numpy functions if missing."""
        result = term
        for func in numpy_funcs:
            # Match bare function name (not already prefixed with np.)
            # Use word boundary to avoid matching partial names like "cosine"
            pattern = rf'(?<!np\.)(?<![a-zA-Z_]){func}\s*\('
            replacement = f'np.{func}('
            result = re.sub(pattern, replacement, result)
        return result
    
    # Generate x arguments string
    x_args = ", ".join([f"x{i}" for i in range(dim)])
    
    # Build return expression
    expr_parts = []
    
    # Track the next available param index for terms without params
    next_wrap_idx = max_param_idx + 1
    
    for term in terms:
        term = term.strip()
        if term:
            # Fix numpy prefix if needed
            term = fix_numpy_prefix(term)
            
            # Auto-wrap: If term has no params[], wrap it with params[N]*
            if not re.search(r'params\[\d+\]', term):
                term = f"params[{next_wrap_idx}]*({term})"
                next_wrap_idx += 1
                # Update max_param_idx for constant term
                max_param_idx = next_wrap_idx - 1
            
            expr_parts.append(f"({term})")
    
    # Add constant term (next param after highest used)
    constant_idx = max_param_idx + 1
    expr_parts.append(f"params[{constant_idx}] * 1")
    
    return_expr = " + ".join(expr_parts) if expr_parts else "0"
    
    # Build function code
    code = f"""def {func_name}({x_args}, params):
    import numpy as np
    return {return_expr}"""
    
    # Required params: indices 0..constant_idx inclusive -> length = constant_idx + 1
    # (In actual logic, the generated code expands as needed regardless of MAX_PARAMS)
    n_params_required = constant_idx + 1
    return code, n_params_required



def validate_terms(
    terms: List[str],
    func_name: str,
    dim: int,
    max_params: int
) -> tuple[bool, str]:
    """Validate term list for correctness.
    
    Args:
        terms: List of term strings (with params)
        func_name: Function name for error messages
        dim: Dimension (valid variables are x0 to x{dim-1})
        max_params: Maximum params (for reference)
        
    Returns:
        (is_valid, error_message) tuple
    """
    # Check term count (leave room for constant)
    if len(terms) > 20: # arbitrary large limit, real limit is usually by token window or user instruction
         return False, f"{func_name}: too many terms ({len(terms)} > 20)"
    
    # Check for invalid variable references
    for term in terms:
        # Check for x{dim} and higher (invalid)
        for i in range(dim, 10):
            if re.search(rf'\bx{i}\b', term):
                return False, f"{func_name}: invalid variable x{i} in term '{term}'"
    
    # Validate params usage: must contain params
    all_param_indices = []
    
    # Check params per term (if params are used)
    for term in terms:
        term_indices = _param_indices(term)
        # Allow terms without params (direct constants like "2*x0" or "x0/(x0+1)")
        if not term_indices:
            continue  # Skip validation for terms without params
        
        unique_term_indices = len(set(term_indices))
        if unique_term_indices > max_params:
             return False, f"{func_name}: term '{term}' uses too many coefficients ({unique_term_indices} > {max_params})"
        
        all_param_indices.extend(term_indices)
            
    # Note: Global total param count is NOT checked against max_params anymore.
    # We rely on auto-remapping to handle arbitrary total params.
    
    return True, ""



def code_to_equation(
    code: str,
    func_name: str,
    params: List[float]
) -> str:
    """Convert Python function code with params to readable math equation.
    
    Args:
        code: Python function code string
        func_name: Function name (e.g., "x0_t")
        params: List of parameter values
        
    Returns:
        Readable equation string like "x0_t = 1.23 * x0 + 0.45 * x1**2 + 0.01"
        
    Example:
        Input: code="def x0_t(x0, x1, params): return params[0] * (x0) + params[1] * (x1**2) + params[2] * 1"
               params=[1.23, 0.45, 0.01]
        Output: "x0_t = 1.2300 * (x0) + 0.4500 * (x1**2) + 0.0100"
    """
    import re
    
    # Extract return expression from code
    match = re.search(r'return\s+(.+)$', code, re.MULTILINE)
    if not match:
        return f"{func_name} = (parsing error)"
    
    return_expr = match.group(1).strip()
    
    # Replace params[i] with actual values
    def replace_param(m):
        idx = int(m.group(1))
        if 0 <= idx < len(params):
            val = params[idx]
            # Format with sign for cleaner output
            return f"{val:.4f}"
        return m.group(0)
    
    equation = re.sub(r'params\[(\d+)\]', replace_param, return_expr)
    
    # Clean up formatting
    equation = equation.replace("+ -", "- ")
    equation = equation.replace("- -", "+ ")
    equation = equation.replace("* 1", "")  # Remove constant multiplier
    equation = equation.strip()
    
    return f"{func_name} = {equation}"


def extract_term_skeleton(term: str) -> str:
    """Extract skeleton structure from a term by replacing all coefficients with 'C'.
    
    This allows comparing terms by structure rather than specific coefficient values.
    
    Args:
        term: Term string with coefficients (e.g., "params[0]*x0", "0.8172*x1", "3.2349*x3")
        
    Returns:
        Skeleton string (e.g., "C*x0", "C*x1", "C*x3")
        
    Examples:
        "params[0]*x0" -> "C*x0"
        "0.8172*x1" -> "C*x1"
        "params[1]*np.sin(params[2]*x0)" -> "C*np.sin(C*x0)"
        "-0.5183*x2" -> "C*x2"
        "3.2349*x3" -> "C*x3"
        "0.9999*x0*np.cos(x1)" -> "C*x0*np.cos(x1)"
        "C * x0" -> "C*x0" (whitespace removed)
    """
    # Step 1: Replace all params[N] patterns with 'C'
    skeleton = re.sub(r'params\[\d+\]', 'C', term)
    
    # Step 2: Replace numeric coefficients that are standalone (not part of variable names)
    # Strategy: Match numbers that are:
    # - At the start of string, or
    # - After whitespace/operators (+, -, *, /, (, )
    # - And followed by *
    
    # Match pattern: (start or operator) + optional-minus + digits + optional-decimal + digits + *
    # Replace the number part with 'C'
    # Fixed to handle '0.0000 * 1' where 1 is not [a-zA-Z]
    skeleton = re.sub(r'(^|[\s\+\-\*/\(])-?\d+\.?\d*(?=\s*\*)', r'\1C', skeleton)
    
    # Step 3: Remove ALL whitespace for consistent comparison
    skeleton = re.sub(r'\s+', '', skeleton)
    
    return skeleton


def merge_remove_list(
    existing_list: List[str],
    new_terms: List[str]
) -> List[str]:
    """Merge new removed terms into existing list, avoiding skeleton duplicates.
    
    Args:
        existing_list: Current list of removed term skeletons
        new_terms: New terms to add (will be converted to skeletons)
        
    Returns:
        Updated list with new unique skeletons added
        
    Example:
        existing: ["C*x0", "C*x1**2"]
        new_terms: ["params[0]*x0", "params[1]*np.sin(x0)"]
        result: ["C*x0", "C*x1**2", "C*np.sin(x0)"]
        # Note: "params[0]*x0" -> "C*x0" is a duplicate, so not added
    """
    result = list(existing_list) if existing_list else []
    existing_skeletons = set(result)
    
    for term in new_terms:
        skeleton = extract_term_skeleton(term)
        if skeleton not in existing_skeletons:
            result.append(skeleton)
            existing_skeletons.add(skeleton)
    
    return result


# ===== Ablation Study Functions =====

def ablation_test_single_term(
    code_str: str,
    func_name: str,
    term_idx: int,
    params: np.ndarray,
    df: pd.DataFrame,
    target_col: str
) -> float:
    """Test removing a single term by setting its parameter to zero.
    
    Args:
        code_str: Python function code string
        func_name: Function name (e.g., 'x0_t')
        term_idx: Index of term/parameter to remove (set to 0)
        params: Parameter array
        df: DataFrame with input data
        target_col: Target column name
        
    Returns:
        MSE when the specified term is removed
    """
    # Create ablated params (copy and set one param to 0)
    params_ablated = params.copy()
    params_ablated[term_idx] = 0.0
    
    # Execute function with ablated params
    func = make_function_from_code(code_str, func_name)
    x_arrays = get_x_arrays_for_function(func, df, params_ablated)
    y_pred = func(*x_arrays)  # params already included in x_arrays
    y_true = df[target_col].values
    
    # Calculate MSE
    mse = np.mean((y_pred - y_true) ** 2)
    return mse


def calculate_term_performance_impacts(
    current_codes: Dict[str, str],
    current_params: Dict[str, np.ndarray],
    df_train: pd.DataFrame,
    target_cols: List[str],
    threshold: float = 0.05
) -> Dict[str, List[Dict]]:
    """Calculate performance impact of each term using ablation study.
    
    Args:
        current_codes: Dict of function code strings {func_name: code}
        current_params: Dict of parameter arrays {func_name: params}
        df_train: Training DataFrame
        target_cols: List of target column names
        threshold: Threshold for classification (default 5%)
        
    Returns:
        Dict of term impacts: {
            'x0_t': [
                {'term_idx': 0, 'impact': 'positive', 'change_rate': -0.12, ...},
                ...
            ],
            ...
        }
    """
    results = {}
    func_names = list(current_codes.keys())
    
    for i, func_name in enumerate(func_names):
        code = current_codes.get(func_name, '')
        params = current_params.get(func_name, np.array([]))
        
        if not code or len(params) == 0:
            results[func_name] = []
            continue
            
        target_col = target_cols[i]
        
        # Calculate baseline MSE (with all terms)
        try:
            func = make_function_from_code(code, func_name)
            x_arrays = get_x_arrays_for_function(func, df_train, params)
            y_pred_full = func(*x_arrays)  # params already included in x_arrays
            y_true = df_train[target_col].values
            mse_baseline = np.mean((y_pred_full - y_true) ** 2)
        except Exception as e:
            print(f"Error calculating baseline MSE for {func_name}: {e}")
            results[func_name] = []
            continue
        
        
        # Extract readable term strings for logging
        try:
            term_strings = extract_terms_from_code_with_params(code, params)
        except Exception as e:
            print(f"Error extracting term strings for {func_name}: {e}")
            term_strings = []
            
        # Test each parameter/term
        term_impacts = []
        for term_idx in range(len(params)):
            # Skip if parameter is already 0 (term not used)
            if abs(params[term_idx]) < 1e-10:
                continue
            
            try:
                # Calculate MSE without this term
                mse_without = ablation_test_single_term(
                    code, func_name, term_idx, params, df_train, target_col
                )
                
                # Calculate change rate
                # Positive change_rate: MSE increases when term removed -> term is helpful
                # Negative change_rate: MSE decreases when term removed -> term is harmful
                change_rate = (mse_without - mse_baseline) / (mse_baseline + 1e-10)
                
                # Classify impact
                if change_rate > threshold:
                    impact = "positive"  # Term removal makes MSE worse -> term is good
                elif change_rate < -threshold:
                    impact = "negative"  # Term removal makes MSE better -> term is bad
                else:
                    impact = "neutral"
                
                # Get term string if available
                term_str = term_strings[term_idx] if term_idx < len(term_strings) else f"params[{term_idx}]"
                
                term_impacts.append({
                    'term_idx': term_idx,
                    'term_str': term_str,  # Added term string for logging
                    'impact': impact,
                    'change_rate': change_rate,
                    'mse_baseline': mse_baseline,
                    'mse_without': mse_without
                })
            except Exception as e:
                print(f"Error in ablation test for {func_name} term {term_idx}: {e}")
                continue
        
        results[func_name] = term_impacts
    
    return results

def determine_action(
    semantic_quality: str, 
    performance_impact: str,
    prev_action: Optional[str] = None
) -> str:
    """Determine action for a term based on semantic quality, impact, and previous action.
    
    Args:
        semantic_quality: LLM's semantic quality ('good', 'neutral', 'bad')
        performance_impact: Calculated performance impact ('positive', 'neutral', 'negative')
        prev_action: Previous iteration's action for this term (e.g., 'hold', 'hold1', 'hold2')
        
    Returns:
        Action: 'keep', 'hold', 'hold1', 'hold2', or 'remove'
    """
    # Normalize inputs
    sem = semantic_quality.lower()
    imp = performance_impact.lower()
    
    # Pre-determined removal
    if sem == 'bad':
        return 'remove'
    
    # Decision logic
    base_action = 'hold'
    if sem == 'good':
        if imp == 'positive':
            return 'keep'
        else:
            base_action = 'hold'
    elif sem == 'neutral':
        base_action = 'hold'
    
    # State machine for consecutive holds
    if base_action == 'hold':
        if prev_action == 'hold1':
            return 'hold2'
        if prev_action == 'hold2':
            return 'remove'
        return 'hold1'
            
    # Fallback
    return base_action


def extract_terms_from_code_with_params(
    code: str, 
    params: List[float]
) -> List[str]:
    """Extract readable term strings from code using params.
    
    Parses 'return (params[0]*x0) + ...' structure and substitutes params.
    Returns list of term strings used in the equation.
    
    Args:
        code: Function code string
        params: List of parameter values
        
    Returns:
        List of formatted term strings (e.g. ['0.5*x0', '0.1*x1**2'])
    """
    if not code:
        return []
        
    # Extract return expression
    match = re.search(r'return\s+(.+)$', code, re.MULTILINE)
    if not match:
        return []
    
    return_expr = match.group(1).strip()
    
    # Split by " + "
    # terms_to_function_code constructs: (term1) + (term2) + ... + params[N]*1
    term_chunks = return_expr.split(' + ')
    
    def replace_param_in_str(s, p_vals):
         def _repl(m):
            idx = int(m.group(1))
            if 0 <= idx < len(p_vals):
                val = p_vals[idx]
                return f"{val:.4f}"
            return m.group(0)
         return re.sub(r'params\[(\d+)\]', _repl, s)
    
    readable_terms = []
    for chunk in term_chunks:
        chunk = chunk.strip()
        # Remove outer parens: (term) -> term
        if chunk.startswith('(') and chunk.endswith(')'):
            chunk = chunk[1:-1]
        
        # Replace params
        readable_term = replace_param_in_str(chunk, params)
        readable_terms.append(readable_term)
        
    # Note: The last term is usually the auto-added constant (params[N] * 1)
    # The caller can decide whether to include it or not based on context.
    # We return all terms found.
    
    return readable_terms
