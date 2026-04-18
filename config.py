"""
Configuration constants for LLM ODE evolution.

This module centralizes all configuration values and constants
used throughout the codebase.
"""



# ==============================================================================
# Optimization Constants
# ==============================================================================
PENALTY_VALUE = 1e10  # Large penalty for invalid/failed evaluations

MAX_OPTIMIZATION_ITERATIONS = 400
DE_TOLERANCE = 1e-5
BFGS_TOLERANCE = 1e-9

DE_MAXITER = 1000
DE_POPSIZE = 20

DIFFERENTIAL_EVOLUTION_CONFIG = {
    'strategy': 'best1bin',
    'maxiter': DE_MAXITER,
    'popsize': DE_POPSIZE,
    'tol': DE_TOLERANCE,
    'atol': DE_TOLERANCE,
    'mutation': (0.5, 1),
    'recombination': 0.7,
    'disp': False,
    'polish': False,
    'seed': 42,
}

# Parallel Processing Configuration
# PARALLEL_N_JOBS: Workers for candidate evaluation loop (-1 = all cores, 1 = sequential)
PARALLEL_N_JOBS = -1

DE_BOUNDS = (-100.0, 100.0)
MAX_PARAMS = 10

# ==============================================================================
# Parameter Processing
# ==============================================================================
PARAMS_PRECISION = 5  # Decimal places for rounding parameters
PARAMS_ZERO_THRESHOLD = 1e-5  # Values below this are set to 0



# ==============================================================================
# LLM Configuration
# ==============================================================================
DEFAULT_MODEL_NAME = "google/gemini-2.5-flash-lite"

# Sampler Model Config
SAMPLER_MAX_TOKENS = 3000
SAMPLER_TEMPERATURE = 0.9

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Scientist (Reasoning) Model
# Using a stronger model for reasoning is recommended
SCIENTIST_MODEL_NAME = "google/gemini-2.5-flash-lite"
SCIENTIST_MAX_TOKENS = 10000
SCIENTIST_TEMPERATURE = 0.6

# Notebook History Limits (Prompts)
MAX_LEARNINGS_HISTORY = 50   # Limit for Structural Learnings (combined confirmed + refuted)


# Evolution Configuration
# ==============================================================================
DEFAULT_NUM_EQUATIONS = 3
REMOVED_TERMS_FORGET_PROBABILITY = 0.01  # Probability to "forget" a removed term for exploration





