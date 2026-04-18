"""
Data loading functions for LLM ODE evolution.

This module contains data loading and variable description generation functions.
"""

import json
from typing import Tuple, Dict

import pandas as pd


def load_dataframes(
    problem_name: str,
    dim: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, test_id, and test_ood datasets for a problem.
    
    Args:
        problem_name: Problem name (e.g., "ID_02")
        dim: Dimension (2 or 4 in the bundled benchmark set)
        
    Returns:
        Tuple of (df_train, df_test_id, df_test_ood)
    """
    base_path = f"./data/{dim}D/{problem_name}"
    df_train = pd.read_csv(f"{base_path}/{problem_name}_train.csv")
    df_test_id = pd.read_csv(f"{base_path}/{problem_name}_test_id.csv")
    df_test_ood = pd.read_csv(f"{base_path}/{problem_name}_test_ood.csv")
    return df_train, df_test_id, df_test_ood


def create_describe(problem_name: str) -> str:
    """Generate variable description from JSON file.
    
    Args:
        problem_name: Problem name (e.g., "ID_02")
        
    Returns:
        Formatted variable description string
        
    Raises:
        ValueError: If the description file is not found
    """
    var_desc_path = f"./data/json/{problem_name}.json"
    
    try:
        with open(var_desc_path, 'r', encoding='utf-8') as f:
            var_desc = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Could not find {var_desc_path}. Returning empty describe.")
        raise ValueError(f"Could not find {var_desc_path}")
    
    describe = "\n"
    describe += var_desc.get("description", "")

    
    return describe.strip()


def create_df_dict(
    df_train: pd.DataFrame, 
    df_test_id: pd.DataFrame, 
    df_test_ood: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """Create a dictionary of dataframes for scoring.
    
    Args:
        df_train: Training dataframe
        df_test_id: In-distribution test dataframe
        df_test_ood: Out-of-distribution test dataframe
        
    Returns:
        Dictionary with 'train', 'test_id', 'test_ood' keys
    """
    return {
        'train': df_train,
        'test_id': df_test_id,
        'test_ood': df_test_ood
    }
