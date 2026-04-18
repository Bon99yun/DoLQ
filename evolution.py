"""
Evolution chain for LLM-guided ODE function discovery.

This module implements the main evolution workflow using LangGraph.
"""

import os
import time
import warnings
from datetime import datetime, timezone
from typing import List, Optional, Callable, Dict, Any
from abc import ABC, abstractmethod
from functools import wraps

import numpy as np
import pandas as pd
import random

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph, START
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict, Annotated

# Local imports
from config import (
    OPENROUTER_BASE_URL,
    SCIENTIST_MODEL_NAME,
    SAMPLER_MAX_TOKENS,
    SAMPLER_TEMPERATURE,
    SCIENTIST_MAX_TOKENS,
    SCIENTIST_TEMPERATURE,
    DEFAULT_NUM_EQUATIONS,
    MAX_LEARNINGS_HISTORY,
    PARALLEL_N_JOBS,
    REMOVED_TERMS_FORGET_PROBABILITY,
)
from utils import (
    get_x_cols,
    get_func_names,
    get_target_cols,
    get_function_args,
    get_x_arrays_for_function,
    make_function_from_code,
    create_functions_from_codes,
)
from optimization import (
    calculate_scores,
    run_differential_evolution,
    run_bfgs_optimization,
)
from compare import is_better_than
from io_utils import get_experiment_logger


LOGGER = get_experiment_logger(__name__)


def should_log_score_improvement(previous_score: float, current_score: float) -> bool:
    """Return True for improvements large enough to be useful in human logs."""
    if not (np.isfinite(previous_score) and np.isfinite(current_score)):
        return True
    return not np.isclose(previous_score, current_score, rtol=1e-12, atol=1e-15)


# ==============================================================================
# Module-level helper function for parallel candidate evaluation (pickle-safe)
# ==============================================================================
def _evaluate_single_candidate(args):
    """Evaluate a single candidate function with DE and BFGS optimization.

    Defined at module level for pickle compatibility with multiprocessing.Pool.

    Args:
        args: Tuple of (code_str, func_name, df_train_dict, y_true, n_params,
              use_de, de_tol, bfgs_tol, cand_original)
              func_name: Function name (e.g., 'x0_t').
                         Note: Also used as DataFrame column name for target data access.
              n_params: Number of params the code actually uses (from terms_to_function_code return value, can exceed MAX_PARAMS)

    Returns:
        Dict with candidate info, score, optimized_params, and opt_details
    """
    from utils import make_function_from_code

    (
        code_str,
        func_name,
        df_train_dict,
        y_true,
        n_params,
        use_de,
        de_tol,
        bfgs_tol,
        cand_original,
    ) = args
    df_train = pd.DataFrame(df_train_dict)

    opt_details = {}  # Store loss and params for each method

    try:
        func = make_function_from_code(code_str, func_name)
        candidates = []

        # 1. Pure BFGS (Always check)
        init_params_ones = np.ones(n_params)
        bfgs_params = run_bfgs_optimization(
            func, df_train, y_true, init_params_ones, tol=bfgs_tol
        )
        bfgs_score_res = calculate_scores(
            func_list=[func],
            params_list=[bfgs_params],
            df_dict={"train": df_train},
            dt_cols=[func_name],
        )
        bfgs_score = bfgs_score_res["train"][0]
        candidates.append(
            {"score": bfgs_score, "params": bfgs_params, "method": "BFGS"}
        )

        opt_details["BFGS"] = {
            "loss": bfgs_score,
            "params": bfgs_params.tolist()
            if isinstance(bfgs_params, np.ndarray)
            else bfgs_params,
        }

        if use_de:
            # 2. Differential Evolution
            de_params, de_score = run_differential_evolution(
                func, df_train, y_true, max_params=n_params, tol=de_tol
            )
            candidates.append({"score": de_score, "params": de_params, "method": "DE"})

            opt_details["DE"] = {
                "loss": de_score,
                "params": de_params.tolist()
                if isinstance(de_params, np.ndarray)
                else de_params,
            }

            # 3. DE + BFGS (Hybrid)
            hybrid_params = run_bfgs_optimization(
                func, df_train, y_true, de_params, tol=bfgs_tol
            )
            hybrid_score_res = calculate_scores(
                func_list=[func],
                params_list=[hybrid_params],
                df_dict={"train": df_train},
                dt_cols=[func_name],
            )
            hybrid_score = hybrid_score_res["train"][0]
            candidates.append(
                {"score": hybrid_score, "params": hybrid_params, "method": "DE+BFGS"}
            )

            opt_details["DE+BFGS"] = {
                "loss": hybrid_score,
                "params": hybrid_params.tolist()
                if isinstance(hybrid_params, np.ndarray)
                else hybrid_params,
            }

        # Select best based on score (minimized)
        best_candidate = min(candidates, key=lambda x: x["score"])

        best_score = best_candidate["score"]
        best_params = best_candidate["params"]
        best_method = best_candidate["method"]

        return {
            **cand_original,
            "score": best_score,
            "optimized_params": best_params,
            "optimization_method": best_method,
            "opt_details": opt_details,
        }

    except Exception as e:
        LOGGER.exception("Error evaluating candidate for %s: %s", func_name, e)
        return {
            **cand_original,
            "score": 1e10,
            "optimized_params": np.ones(n_params),
            "optimization_method": "error",
            "opt_details": {},
        }


from prompt import make_sampler_ODE_prompt
from with_structured_output import create_function_output_class

warnings.filterwarnings("ignore")
load_dotenv(".env")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")


def init_score(
    evo, init_func_str_list: List[str], df_dict: dict, params_list: List
) -> dict:
    """Calculate initial scores for given function strings.

    Args:
        evo: Evolution chain instance
        init_func_str_list: List of initial function code strings
        df_dict: Dictionary of DataFrames
        params_list: List of parameter arrays

    Returns:
        Dictionary with scores for each dataset
    """
    func_names = evo.get_func_names()
    target_cols = evo.get_target_cols()
    func_list = [
        make_function_from_code(code_str, func_names[idx])
        for idx, code_str in enumerate(init_func_str_list)
    ]

    # Ensure params are numpy arrays
    params_numpy = [
        np.array(p) if not isinstance(p, np.ndarray) else p for p in params_list
    ]

    score_dict = calculate_scores(func_list, params_numpy, df_dict, target_cols)
    return score_dict


class RuntimeMetrics(TypedDict, total=False):
    started_at: Optional[str]
    ended_at: Optional[str]
    elapsed_seconds: Optional[float]
    started_at_monotonic: Optional[float]
    ended_at_monotonic: Optional[float]
    process_cpu_seconds_start: Optional[float]
    process_cpu_seconds_end: Optional[float]
    process_cpu_seconds_delta: Optional[float]


def _process_cpu_seconds() -> float:
    return time.process_time()


def start_runtime_metrics() -> RuntimeMetrics:
    started_at = datetime.now(timezone.utc).isoformat()
    started_at_monotonic = time.perf_counter()
    process_cpu_seconds_start = _process_cpu_seconds()
    return {
        "started_at": started_at,
        "ended_at": None,
        "elapsed_seconds": None,
        "started_at_monotonic": started_at_monotonic,
        "ended_at_monotonic": None,
        "process_cpu_seconds_start": process_cpu_seconds_start,
        "process_cpu_seconds_end": None,
        "process_cpu_seconds_delta": None,
    }


def finish_runtime_metrics(runtime_metrics: Dict[str, Any]) -> RuntimeMetrics:
    ended_at = datetime.now(timezone.utc).isoformat()
    ended_at_monotonic = time.perf_counter()
    process_cpu_seconds_end = _process_cpu_seconds()
    started_at_monotonic = runtime_metrics.get("started_at_monotonic")
    process_cpu_seconds_start = runtime_metrics.get("process_cpu_seconds_start")
    return {
        **runtime_metrics,
        "ended_at": ended_at,
        "elapsed_seconds": ended_at_monotonic - started_at_monotonic
        if started_at_monotonic is not None
        else None,
        "ended_at_monotonic": ended_at_monotonic,
        "process_cpu_seconds_end": process_cpu_seconds_end,
        "process_cpu_seconds_delta": process_cpu_seconds_end - process_cpu_seconds_start
        if process_cpu_seconds_start is not None
        else None,
    }


class GraphState(TypedDict, total=False):
    """State definition for the evolution graph."""

    ### Initial state
    initial_state: Annotated[Optional[dict], "Initial state"] = None

    ### Boolean flags (served from main.py)
    use_var_desc: Annotated[bool, "Whether to use variable description"]
    use_differential_evolution: Annotated[
        bool, "Whether to use Differential Evolution"
    ] = False
    use_scientist: Annotated[bool, "Whether to run scientist agent"] = False

    ### Iteration tracking (served from main.py)
    current_iteration: Annotated[
        Optional[int], "Current iteration number (1-based)"
    ] = None
    total_iterations: Annotated[Optional[int], "Total number of iterations"] = None

    ### Generated by LLM (Scientist)
    research_notebook: Annotated[
        Optional[Dict[str, Any]], "Accumulated Research Notebook (insight, suggestion)"
    ] = None

    ### Scientist Prompts (for Logging)
    scientist_analysis_prompt: Annotated[
        Optional[str], "Prompt used for analyze_and_record_observation"
    ] = None

    ### Scientist Metadata (Raw JSON Outputs)
    scientist_analysis_metadata: Annotated[
        Optional[Dict[str, Any]], "Raw structured output from ExperimentAnalysis"
    ] = None

    ### served from main.py
    generated_code: Annotated[
        Optional[List[str]], "Generated Python function strings"
    ] = None
    describe: Annotated[Optional[str], "Variable/System description string"] = None
    score: Annotated[
        Optional[Dict[str, List[float]]], "MSE scores (train/test_id/test_ood)"
    ] = None

    ### made by evolution_chain
    sampler_prompt: Annotated[Optional[str], "Full prompt string sent to LLM"] = None
    func_list: Annotated[
        Optional[List[Callable]], "Executable function objects compiled from code"
    ] = None
    params_list: Annotated[
        Optional[List[np.ndarray]], "Optimized parameters (numpy arrays)"
    ] = None
    error_occurred: Annotated[
        bool, "Flag indicating if an error occurred in the chain"
    ] = False
    total_error_count: Annotated[int, "Total accumulated errors during evolution"] = 0
    sampler_output: Annotated[
        Optional[Dict[str, Any]], "Raw structured output from functional generation"
    ] = None
    raw_candidates: Annotated[
        Optional[List[Dict[str, Any]]],
        "List of ODE system pairs (ODE system-based architecture)",
    ] = None

    ### ODE system-based fields
    evaluated_candidates: Annotated[
        Optional[List[Dict[str, Any]]], "List of evaluated ODE system pairs"
    ] = None
    best_ode_system: Annotated[
        Optional[Dict[str, Any]], "Best ODE system (codes, dim_scores, combined_mse)"
    ] = None
    current_ode_system: Annotated[
        Optional[Dict[str, Any]], "Current iteration's best ODE system for observation"
    ] = None
    prev_best_ode_system: Annotated[
        Optional[Dict[str, Any]], "Previous iteration's best ODE system for comparison"
    ] = None
    previous_generation_ode_system: Annotated[
        Optional[Dict[str, Any]], "ODE system from immediately preceding generation"
    ] = None
    global_improvement: Annotated[
        bool, "Whether this iteration improved over previous best"
    ] = False
    local_improvement: Annotated[
        bool, "Whether this iteration improved over previous generation"
    ] = False

    ### Remove List (accumulated across iterations)
    removed_terms_per_dim: Annotated[
        Optional[Dict[str, List[str]]],
        "Per-dimension list of removed term skeletons (e.g., {'x0_t': ['C*x0', 'C*np.sin(x0)']})",
    ] = None
    resurrection_counts: Annotated[
        Optional[Dict[str, Dict[str, int]]],
        "Count of how many times each term has been resurrected (pardoned)",
    ] = None
    best_iteration_per_dim: Annotated[
        Optional[Dict[str, int]],
        "Iteration number where the best equation for each dimension was found",
    ] = None

    ### Logging
    token_usage: Annotated[
        Optional[Dict[str, Dict[str, int]]],
        "Token usage stats for this iteration (sampler, scientist)",
    ] = None
    dim_term_details: Annotated[
        Optional[Dict[str, List[Dict[str, Any]]]], "Detailed term ablation results"
    ] = None
    iteration_runtime_metrics: Annotated[
        Optional[RuntimeMetrics], "Per-iteration runtime metrics"
    ] = None
    stage_runtime_metrics: Annotated[
        Optional[Dict[str, RuntimeMetrics]],
        "Per-stage runtime metrics keyed by node name",
    ] = None
    runtime_metrics_history: Annotated[
        Optional[List[RuntimeMetrics]], "Optional runtime metric snapshots/history"
    ] = None


class EvolutionBase(ABC):
    """Abstract base class for evolution chain with error handling."""

    @abstractmethod
    def handle_error(
        self, error: Exception, state: GraphState, func_name: str = ""
    ) -> GraphState:
        """Handle errors during node execution."""
        pass


def safe_node(func: Callable) -> Callable:
    """Decorator to wrap node functions with error handling."""

    @wraps(func)
    def wrapper(self, state: GraphState, *args, **kwargs) -> GraphState:
        stage_name = func.__name__
        stage_metrics = start_runtime_metrics()
        try:
            result = func(self, state, *args, **kwargs)
            finished_stage_metrics = finish_runtime_metrics(stage_metrics)
            stage_runtime_metrics = dict(
                result.get("stage_runtime_metrics")
                or state.get("stage_runtime_metrics")
                or {}
            )
            stage_runtime_metrics[stage_name] = finished_stage_metrics
            return {**result, "stage_runtime_metrics": stage_runtime_metrics}
        except Exception as e:
            finished_stage_metrics = finish_runtime_metrics(stage_metrics)
            stage_runtime_metrics = dict(state.get("stage_runtime_metrics") or {})
            stage_runtime_metrics[stage_name] = finished_stage_metrics
            error_state = {**state, "stage_runtime_metrics": stage_runtime_metrics}
            return self.handle_error(e, error_state, func_name=stage_name)

    return wrapper


class evolution_chain(EvolutionBase):
    """Main evolution chain class for LLM-guided ODE discovery."""

    def __init__(
        self,
        df_train: pd.DataFrame,
        df_test_id: pd.DataFrame,
        df_test_ood: pd.DataFrame,
        max_params: int,
        dim: int,
        num_equations: int = DEFAULT_NUM_EQUATIONS,
        de_tolerance: float = None,
        bfgs_tolerance: float = None,
        use_gt: bool = False,
        sampler_model_name: str = "google/gemini-2.5-flash-lite",
        scientist_model_name: str = "google/gemini-2.5-flash-lite",
        forget_probability: float = REMOVED_TERMS_FORGET_PROBABILITY,
    ):
        """Initialize evolution chain.

        Args:
            df_train: Training dataframe
            df_test_id: In-distribution test dataframe
            df_test_ood: Out-of-distribution test dataframe
            max_params: Maximum number of parameters allowed
            dim: Dimension of the ODE system
            num_equations: Number of equations in the system
            de_tolerance: Differential Evolution tolerance
            bfgs_tolerance: BFGS Optimization tolerance
            use_gt: Whether to use ground truth targets
            sampler_model_name: Model name for Sampler LLM
            scientist_model_name: Model name for Scientist LLM
        """
        self.df_train = df_train
        self.df_test_id = df_test_id
        self.df_test_ood = df_test_ood
        self.max_params = max_params
        self.sampler_model_name = sampler_model_name
        self.scientist_model_name = scientist_model_name
        self.dim = dim
        self.num_equations = num_equations
        self.de_tolerance = de_tolerance
        self.bfgs_tolerance = bfgs_tolerance
        self.use_gt = use_gt
        self.forget_probability = forget_probability

        # Sampler LLM (Code Generation)
        self.llm = ChatOpenAI(
            model=sampler_model_name,
            api_key=openrouter_api_key,
            base_url=OPENROUTER_BASE_URL,
            max_tokens=SAMPLER_MAX_TOKENS,
            verbose=True,
            temperature=SAMPLER_TEMPERATURE,
        )

        # Scientist (Reasoning) LLM
        self.scientist_llm = ChatOpenAI(
            model=scientist_model_name,
            api_key=openrouter_api_key,
            base_url=OPENROUTER_BASE_URL,
            temperature=SCIENTIST_TEMPERATURE,
            max_tokens=SCIENTIST_MAX_TOKENS,
        )

        self.graph = StateGraph(GraphState)

    def get_x_cols(self) -> List[str]:
        """Get x column names from training data."""
        return get_x_cols(self.df_train)

    def get_func_names(self) -> List[str]:
        """Get function names (always x*_t, regardless of use_gt)."""
        return get_func_names(self.df_train)

    def get_target_cols(self) -> List[str]:
        """Get target column names from training data (x*_t or x*_t_gt based on use_gt)."""
        return get_target_cols(self.df_train, use_gt=self.use_gt)

    def get_function_args(self, func: Callable) -> List[str]:
        """Get function x arguments."""
        return get_function_args(func)

    def get_x_arrays_for_function(
        self, func: Callable, df: pd.DataFrame, params: List
    ) -> List[np.ndarray]:
        """Get x arrays for function."""
        # Ensure params is numpy array
        if not isinstance(params, np.ndarray):
            params = np.array(params)
        return get_x_arrays_for_function(func, df, params)

    def make_function(self, parsed_code: str, dt_col: str) -> Callable:
        """Convert Python code string to actual function."""
        return make_function_from_code(parsed_code, dt_col)

    def handle_error(
        self, error: Exception, state: GraphState, func_name: str = ""
    ) -> GraphState:
        """Handle errors by logging and reverting to initial state."""
        LOGGER.exception("Error occurred in %s: %s", func_name, error)
        initial_state = state.get("initial_state", state)
        current_errors = state.get("total_error_count", 0)
        stage_runtime_metrics = dict(initial_state.get("stage_runtime_metrics") or {})
        stage_runtime_metrics.update(state.get("stage_runtime_metrics") or {})
        return {
            **initial_state,
            "error_occurred": True,
            "initial_state": initial_state,
            "total_error_count": current_errors + 1,
            "stage_runtime_metrics": stage_runtime_metrics,
        }

    @safe_node
    def make_func_list(self, state: GraphState) -> GraphState:
        """Create function list from state."""
        # Evaluate error state and restore initial state if needed
        if state.get("error_occurred", False) and state.get("initial_state"):
            state = {**state.get("initial_state"), "error_occurred": False}

        func_names = self.get_func_names()
        initial_state = state

        func_list = create_functions_from_codes(state["generated_code"], func_names)

        return {
            **state,
            "func_list": func_list,
            "initial_state": initial_state,
            "error_occurred": False,
        }

    @safe_node
    def make_prompt(self, state: GraphState) -> GraphState:
        """Generate prompt for LLM from state."""
        from utils import code_to_equation

        x_cols = self.get_x_cols()
        func_names = ", ".join(self.get_func_names())

        # Get describe from state
        describe = state.get("describe", "")

        # Format Previous Attempt for Prompt (using previous_generation_ode_system instead of best_ode_system)
        # Get term evaluations from previous iteration metadata
        term_evaluations = None
        analysis_metadata = state.get("scientist_analysis_metadata")
        if analysis_metadata:
            term_evaluations = analysis_metadata.get("term_evaluations")

        # Apply Soft Forgetting Policy (Probabilistic Pardon)
        removed_terms_per_dim = state.get("removed_terms_per_dim")
        forbidden_terms = None
        resurrection_counts = state.get("resurrection_counts") or {}

        if removed_terms_per_dim:
            forbidden_terms = {}
            for dim_name, skeletons in removed_terms_per_dim.items():
                # Filter out skeletons based on forget probability
                kept_skeletons = []

                # Ensure resurrection count dict exists for this dimension
                if dim_name not in resurrection_counts:
                    resurrection_counts[dim_name] = {}

                for s in skeletons:
                    if random.random() > self.forget_probability:
                        # Keep in forbidden list (High Probability)
                        kept_skeletons.append(s)
                    else:
                        # "Forget" (Pardon/Resurrect) - Exclude from forbidden list (Low Probability)
                        # Log the resurrection event
                        LOGGER.info(
                            "[Resurrection] Term resurrected in %s: %r",
                            dim_name,
                            s,
                        )

                        # Increment count
                        current_count = resurrection_counts[dim_name].get(s, 0)
                        resurrection_counts[dim_name][s] = current_count + 1

                if kept_skeletons:
                    forbidden_terms[dim_name] = kept_skeletons

            # If all skeletons were forgotten for this prompt, set to None
            if not forbidden_terms:
                forbidden_terms = None

        # Format previous_attempt_str if Scientist is NOT used
        previous_attempt_str = ""
        use_scientist = state.get("use_scientist", False)
        if not use_scientist:
            prev_pair = state.get("previous_generation_ode_system")
            if prev_pair:
                lines = []
                codes = prev_pair.get("codes", {})
                params = prev_pair.get("dim_params", {})
                scores = prev_pair.get("dim_scores", {})

                lines.append("[Previous Attempt (Optimized Coefficients)]")

                for dim_name in sorted(codes.keys()):
                    code = codes.get(dim_name)
                    p_list = params.get(dim_name, [])
                    mse = scores.get(dim_name, float("inf"))

                    if code and len(p_list) > 0:
                        eq_str = code_to_equation(code, dim_name, p_list)
                        lines.append(f"{eq_str} (MSE: {mse:.6e})")
                    else:
                        lines.append(f"{dim_name}: No valid equation")

                previous_attempt_str = "\n".join(lines)

        prompt = make_sampler_ODE_prompt(
            x_cols=x_cols,
            func_names=func_names,
            max_params=self.max_params,
            # insight_list is now just the accumulated insight string
            insight_list=[
                (state.get("research_notebook") or {}).get("accumulated_insight", "")
            ],
            use_scientist=use_scientist,
            describe=describe,
            dim=self.dim,
            removed_terms_per_dim=forbidden_terms,
            term_evaluations=term_evaluations,
            previous_attempt_str=previous_attempt_str,
        )

        return {
            **state,
            "sampler_prompt": prompt,
            "resurrection_counts": resurrection_counts,
        }

    @safe_node
    def sampler(self, state: GraphState) -> GraphState:
        """Sample new functions from LLM based on current state."""
        message = HumanMessage(
            content=[{"type": "text", "text": state["sampler_prompt"]}]
        )

        # Dynamically create FunctionOutput class with term list schema
        FunctionOutput = create_function_output_class(
            dim=self.dim, num_equations=self.num_equations, max_params=self.max_params
        )

        llm = self.llm.with_structured_output(FunctionOutput, include_raw=True)
        response = llm.invoke([message])

        # Extract parsed result and raw usage
        result = response["parsed"].model_dump()
        raw_msg = response["raw"]

        # Log token usage
        token_usage_update = {}
        if hasattr(raw_msg, "usage_metadata"):
            usage = raw_msg.usage_metadata
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            LOGGER.info(
                "[Sampler Token Usage] Input: %s, Output: %s",
                input_tokens,
                output_tokens,
            )
            token_usage_update["sampler"] = {
                "input": input_tokens,
                "output": output_tokens,
            }

        # Merge with existing token usage (if any)
        current_usage = state.get("token_usage") or {}
        current_usage.update(token_usage_update)

        # We just store the raw result here, parsing happens in parse_candidates
        return {**state, "sampler_output": result, "token_usage": current_usage}

    @safe_node
    def parse_candidates(self, state: GraphState) -> GraphState:
        """Parse LLM output term lists into ODE system pairs with Python code.

        Converts term lists (now TermSuggestion objects) from LLM to full Python function code.
        Each term already includes params[i], we just need to combine them.
        """
        from utils import terms_to_function_code, validate_terms

        result = state["sampler_output"]
        func_names = self.get_func_names()

        # Get ode_pairs from LLM output
        ode_pairs = result.get("ode_pairs", [])

        raw_candidates = []  # List of ODE systems

        for ode_system in ode_pairs:
            codes = {}
            dim_n_params = {}  # func_name -> n_params (actual count used by code, can exceed MAX_PARAMS)
            ode_system_valid = True
            validation_errors = []
            term_reasonings = {}

            for func_name in func_names:
                term_suggestions = ode_system.get(func_name)
                if term_suggestions and isinstance(term_suggestions, list):
                    terms = []
                    reasonings = []
                    for ts in term_suggestions:
                        if isinstance(ts, dict):
                            terms.append(ts.get("term", ""))
                            reasonings.append(ts.get("reasoning", ""))
                        else:
                            terms.append(str(ts))
                            reasonings.append("")

                    is_valid, error_msg = validate_terms(
                        terms, func_name, self.dim, self.max_params
                    )
                    if is_valid:
                        code, n_params = terms_to_function_code(
                            terms, func_name, self.dim, self.max_params
                        )
                        codes[func_name] = code
                        dim_n_params[func_name] = n_params
                        term_reasonings[func_name] = reasonings
                    else:
                        ode_system_valid = False
                        validation_errors.append(error_msg)
                else:
                    ode_system_valid = False
                    validation_errors.append(f"{func_name}: missing or not a list")

            if ode_system_valid and len(codes) == len(func_names):
                raw_candidates.append(
                    {
                        "codes": codes,
                        "dim_n_params": dim_n_params,
                        "pair_reasoning": ode_system.get("pair_reasoning", ""),
                        "term_reasonings": term_reasonings,
                    }
                )
            elif validation_errors:
                # Log validation failures for debugging
                LOGGER.warning(
                    "Skipping invalid ODE system: %s", "; ".join(validation_errors)
                )

        return {
            **state,
            "raw_candidates": raw_candidates,  # List of validated ODE systems with generated code
        }

    @safe_node
    def evaluate_candidates(self, state: GraphState) -> GraphState:
        """Evaluate all ODE system pairs using DE and/or BFGS.

        Fully parallelized: all (ODE system, dimension) combinations evaluated simultaneously.
        """
        from multiprocessing import Pool, cpu_count

        raw_candidates = state["raw_candidates"]  # List of ODE systems
        func_names = self.get_func_names()
        target_cols = self.get_target_cols()
        use_de = state.get("use_differential_evolution", False)

        df_train_dict = self.df_train.to_dict()
        n_workers = PARALLEL_N_JOBS if PARALLEL_N_JOBS > 0 else cpu_count()

        # Flatten all evaluation tasks: (ode_system_idx, func_name, args)
        eval_tasks = []
        task_map = []  # To reconstruct results: [(ode_system_idx, func_name), ...]

        for ode_system_idx, ode_system in enumerate(raw_candidates):
            codes = ode_system["codes"]
            dim_n_params = ode_system.get("dim_n_params", {})
            for i, func_name in enumerate(func_names):
                code_str = codes.get(func_name, "")
                if not code_str:
                    continue
                n_params = dim_n_params.get(func_name, self.max_params)

                if i >= len(target_cols):
                    target_col = func_name
                else:
                    target_col = target_cols[i]
                if target_col not in self.df_train.columns:
                    LOGGER.warning(
                        "Target column %s not found, skipping", target_col
                    )
                    continue

                y_true = self.df_train[target_col].values.astype(np.float32)
                args = (
                    code_str,
                    func_name,
                    df_train_dict,
                    y_true,
                    n_params,
                    use_de,
                    self.de_tolerance,
                    self.bfgs_tolerance,
                    {"code": code_str},
                )
                eval_tasks.append(args)
                task_map.append((ode_system_idx, func_name))

        # Parallel evaluation of all (ODE system, dimension) combinations
        if eval_tasks:
            with Pool(processes=n_workers) as pool:
                results = pool.map(_evaluate_single_candidate, eval_tasks)
        else:
            results = []

        # Reconstruct results into ODE system structure
        ode_system_results = {
            i: {
                "dim_scores": {},
                "dim_params": {},
                "dim_codes": {},
                "dim_opt_methods": {},
            }
            for i in range(len(raw_candidates))
        }

        for (ode_system_idx, func_name), result in zip(task_map, results):
            n_params = (
                raw_candidates[ode_system_idx]
                .get("dim_n_params", {})
                .get(func_name, self.max_params)
            )
            ode_system_results[ode_system_idx]["dim_scores"][func_name] = result.get(
                "score", float("inf")
            )
            ode_system_results[ode_system_idx]["dim_params"][func_name] = result.get(
                "optimized_params", np.ones(n_params)
            )
            ode_system_results[ode_system_idx]["dim_codes"][func_name] = result.get(
                "code", ""
            )
            ode_system_results[ode_system_idx]["dim_opt_methods"][func_name] = (
                result.get("optimization_method", "unknown")
            )
            # Collect detailed optimization logs
            if "opt_details" in result:
                if "dim_opt_details" not in ode_system_results[ode_system_idx]:
                    ode_system_results[ode_system_idx]["dim_opt_details"] = {}
                ode_system_results[ode_system_idx]["dim_opt_details"][func_name] = (
                    result["opt_details"]
                )

        # Build evaluated_candidates list
        evaluated_candidates = []
        for ode_system_idx, ode_system in enumerate(raw_candidates):
            pr = ode_system_results[ode_system_idx]
            dim_scores = pr["dim_scores"]

            evaluated_candidates.append(
                {
                    "codes": ode_system.get("codes", {}),
                    "dim_scores": dim_scores,
                    "dim_params": pr["dim_params"],
                    "dim_opt_methods": pr["dim_opt_methods"],
                    "dim_opt_details": pr.get(
                        "dim_opt_details", {}
                    ),  # Include opt details
                    "pair_reasoning": ode_system.get("pair_reasoning", ""),
                    "term_reasonings": ode_system.get(
                        "term_reasonings", {}
                    ),  # Pass Sampler's term-level reasoning
                }
            )

        return {**state, "evaluated_candidates": evaluated_candidates}

    @safe_node
    def select_best(self, state: GraphState) -> GraphState:
        """Select best ODE system using Composite Best strategy.

        Instead of selecting a single best ODE system generated by the LLM,
        we construct a new 'Global Best' by independently selecting the best-performing
        equation for each dimension from the history (previous best) and current candidates.

        This ensures monotonic improvement per dimension.
        """
        import copy

        evaluated_candidates = state.get("evaluated_candidates", [])
        func_names = self.get_func_names()
        prev_best_ode_system = state.get("best_ode_system")

        if not evaluated_candidates:
            # Keep previous best if no new candidates
            return state

        # --- Composite Best Logic ---

        # 1. Initialize new_best_ode_system (Start with previous best if exists, else empty shell)
        if prev_best_ode_system:
            new_best_ode_system = copy.deepcopy(prev_best_ode_system)
        else:
            new_best_ode_system = {
                "codes": {},
                "dim_scores": {},
                "dim_params": {},
                "dim_opt_methods": {},
                "dim_opt_details": {},
                # Mixed reasoning isn't easily mergeable, so we might lose some context here.
                # We'll just keep the structure for now.
                "pair_reasoning": "Composite of best equations found so far.",
                "term_reasonings": {},
            }

        # 2. Define Candidate Pool (Previous Best + Current Batch)
        pool = evaluated_candidates[:]
        if prev_best_ode_system:
            # We include prev_best in the pool to explicitly re-evaluate/keep it
            # But practically we just iterate dimensions and check if any candidate beats what's currently in new_best_ode_system
            # (which is initialized to prev_best).
            pass

        # 3. Iterate Dimensions to find the absolute best equation for each
        for func_name in func_names:
            # Current best score for this func_name (from prev_best or inf)
            prev_dim_best_score = new_best_ode_system["dim_scores"].get(
                func_name, float("inf")
            )
            current_dim_best_score = prev_dim_best_score
            best_offer_cand = None

            # Check all new candidates
            for cand in evaluated_candidates:
                score = cand.get("dim_scores", {}).get(func_name, float("inf"))
                if score < current_dim_best_score:
                    current_dim_best_score = score
                    best_offer_cand = cand

            # If we found a better candidate for this dimension (best_offer_cand is not None)
            if best_offer_cand:
                # Log only meaningful score changes to avoid noisy near-tie updates.
                if should_log_score_improvement(
                    prev_dim_best_score, current_dim_best_score
                ):
                    LOGGER.info(
                        "[Global Best Update] %s MSE improved: %.12e -> %.12e",
                        func_name,
                        prev_dim_best_score,
                        current_dim_best_score,
                    )
                # Update new_best_ode_system with this winner
                new_best_ode_system["codes"][func_name] = best_offer_cand["codes"].get(
                    func_name, ""
                )
                new_best_ode_system["dim_scores"][func_name] = current_dim_best_score
                new_best_ode_system["dim_params"][func_name] = best_offer_cand[
                    "dim_params"
                ].get(func_name, [])
                new_best_ode_system["dim_opt_methods"][func_name] = best_offer_cand[
                    "dim_opt_methods"
                ].get(func_name, "unknown")

                # Copy opt details
                if "dim_opt_details" in best_offer_cand:
                    if "dim_opt_details" not in new_best_ode_system:
                        new_best_ode_system["dim_opt_details"] = {}
                    new_best_ode_system["dim_opt_details"][func_name] = best_offer_cand[
                        "dim_opt_details"
                    ].get(func_name, {})

                # Copy term reasoning if available
                if "term_reasonings" in best_offer_cand:
                    if "term_reasonings" not in new_best_ode_system:
                        new_best_ode_system["term_reasonings"] = {}
                    new_best_ode_system["term_reasonings"][func_name] = best_offer_cand[
                        "term_reasonings"
                    ].get(func_name, [])

        # --- Determine 'Current ODE System' (Batch Best) for Observation ---
        # The user likely wants to see the "Best of Batch" to compare with History.
        # We can apply the same Composite Logic restricted to the CURRENT batch.

        batch_best_ode_system = {
            "codes": {},
            "dim_scores": {},
            "dim_params": {},
            "dim_opt_methods": {},
            "dim_opt_details": {},
            "term_reasonings": {},
        }

        for func_name in func_names:
            best_score = float("inf")
            best_cand = None
            for cand in evaluated_candidates:
                score = cand.get("dim_scores", {}).get(func_name, float("inf"))
                if score < best_score:
                    best_score = score
                    best_cand = cand

            if best_cand:
                batch_best_ode_system["codes"][func_name] = best_cand["codes"].get(
                    func_name, ""
                )
                batch_best_ode_system["dim_scores"][func_name] = best_score
                batch_best_ode_system["dim_params"][func_name] = best_cand[
                    "dim_params"
                ].get(func_name, [])
                batch_best_ode_system["dim_opt_methods"][func_name] = best_cand[
                    "dim_opt_methods"
                ].get(func_name, "unknown")
                if "dim_opt_details" in best_cand:
                    if "dim_opt_details" not in batch_best_ode_system:
                        batch_best_ode_system["dim_opt_details"] = {}
                    batch_best_ode_system["dim_opt_details"][func_name] = best_cand[
                        "dim_opt_details"
                    ].get(func_name, {})
                if "term_reasonings" in best_cand:
                    if "term_reasonings" not in batch_best_ode_system:
                        batch_best_ode_system["term_reasonings"] = {}
                    batch_best_ode_system["term_reasonings"][func_name] = best_cand[
                        "term_reasonings"
                    ].get(func_name, [])

        # Build func_list, params_list, generated_code from new_best_ode_system (Composite Global Best)
        # NOTE: This means the output of the chain is the "Best So Far", not necessarily "Just Generated".
        # This is usually desired for iterative improvement.
        final_func_list = []
        final_params_list = []
        final_code_list = []

        for func_name in func_names:
            code = new_best_ode_system["codes"].get(func_name, "")
            params = new_best_ode_system["dim_params"].get(
                func_name, np.ones(self.max_params)
            )

            final_code_list.append(code)
            final_params_list.append(params)
            if code:
                final_func_list.append(make_function_from_code(code, func_name))
            else:
                final_func_list.append(None)

        # Calculate full scores dict (train, test_id, test_ood) for the new Composite Best
        df_dict = {
            "train": self.df_train,
            "test_id": self.df_test_id,
            "test_ood": self.df_test_ood,
        }

        target_cols = self.get_target_cols()

        score_dict = calculate_scores(
            func_list=final_func_list,
            params_list=final_params_list,
            df_dict=df_dict,
            dt_cols=target_cols,
        )

        return {
            **state,
            "func_list": final_func_list,
            "params_list": final_params_list,
            "generated_code": final_code_list,
            "score": score_dict,
            "best_ode_system": new_best_ode_system,  # Updated Global Best (Composite)
            "current_ode_system": batch_best_ode_system,  # Best of Current Batch (Composite)
            "previous_generation_ode_system": prev_best_ode_system,  # Previous Best (for comparison)
        }

    def check_error_and_route(self, state: GraphState) -> str:
        """Check if error occurred and route to restart or continue."""
        if state.get("error_occurred", False):
            return "restart"
        return "continue"

    @safe_node
    def analyze_and_record_observation(self, state: GraphState) -> GraphState:
        """Analyze experimental results and update the Research Notebook."""
        current_ode_system = state.get("current_ode_system")
        best_ode_system = state.get("best_ode_system")

        if not current_ode_system:
            return state

        # Get comparison targets
        # best_ode_system in state at this point is the Global Best from start of iteration (passed from main)
        global_best_ode_system = state.get("best_ode_system")
        previous_generation_ode_system = state.get(
            "previous_generation_ode_system"
        )  # Previous gen's best

        # 3-Way Comparison Logic
        global_improvement = False
        local_improvement = False

        if global_best_ode_system is None:
            global_improvement = True  # First iteration
            outcome_type = "GLOBAL_SUCCESS (First iteration - new best established)"
        elif is_better_than(current_ode_system, global_best_ode_system):
            global_improvement = True
            outcome_type = "GLOBAL_SUCCESS (New all-time best record!)"
        elif previous_generation_ode_system and is_better_than(
            current_ode_system, previous_generation_ode_system
        ):
            local_improvement = True
            outcome_type = "LOCAL_IMPROVEMENT (Better than previous generation, but not global best)"
        else:
            outcome_type = "REGRESSION (Worse than or equal to previous generation)"

        # Calculate term performance impacts using ablation study
        from utils import calculate_term_performance_impacts

        term_impacts = calculate_term_performance_impacts(
            current_codes=current_ode_system.get("codes", {}),
            current_params=current_ode_system.get("dim_params", {}),
            df_train=self.df_train,
            target_cols=self.get_target_cols(),
            threshold=0.05,  # 5% threshold
        )

        # Build observation for scientist with 3-way comparison data
        observation = {
            # Global Best (Target)
            "global_best_codes": global_best_ode_system.get("codes", {})
            if global_best_ode_system
            else {},
            "global_best_dim_scores": global_best_ode_system.get("dim_scores", {})
            if global_best_ode_system
            else {},
            "global_best_params": global_best_ode_system.get("dim_params", {})
            if global_best_ode_system
            else {},
            # Previous Generation (Baseline)
            "prev_gen_codes": previous_generation_ode_system.get("codes", {})
            if previous_generation_ode_system
            else {},
            "prev_gen_params": previous_generation_ode_system.get("dim_params", {})
            if previous_generation_ode_system
            else {},
            "prev_gen_dim_scores": previous_generation_ode_system.get("dim_scores", {})
            if previous_generation_ode_system
            else {},
            # Current (This Iteration)
            "current_codes": current_ode_system.get("codes", {}),
            "current_params": current_ode_system.get("dim_params", {}),
            "current_dim_scores": current_ode_system.get("dim_scores", {}),
            "pair_reasoning": current_ode_system.get("pair_reasoning", ""),
            # Sampler's term-level reasoning for Scientist to use
            "term_reasonings": current_ode_system.get("term_reasonings", {}),
            # Automated term performance impacts from ablation study
            "term_impacts": term_impacts,
        }

        # Call LLM for analysis and notebook record
        analysis_result = None
        ready_for_scientist = False
        obs_prompt = None
        new_notebook = state.get("research_notebook") or {"accumulated_insight": ""}

        # Initialize token usage for this node from state
        current_usage = state.get("token_usage") or {}

        if state.get("use_scientist", False) and observation:
            describe = state.get("describe", "")
            # Prepare notebook for context
            truncated_notebook = {
                "accumulated_insight": new_notebook.get("accumulated_insight", "")
            }

            analysis_result, obs_prompt, scientist_usage = (
                self._call_llm_for_analysis_and_record(
                    observations=[observation],
                    outcome_type=outcome_type,
                    describe=describe,
                    notebook=truncated_notebook,
                    removed_terms_per_dim=state.get("removed_terms_per_dim"),
                    current_iteration=state.get("current_iteration"),
                    total_iterations=state.get("total_iterations"),
                )
            )

            # Update token usage if available
            if scientist_usage:
                current_usage["scientist"] = scientist_usage

            if analysis_result:
                ready_for_scientist = True

                # Apply updates to notebook immediately
                updated_nb = dict(new_notebook)
                updated_nb["accumulated_insight"] = analysis_result.updated_insight
                new_notebook = updated_nb

        # Merge LLM evaluation with Automated Impact Analysis to determine Actions
        from utils import (
            merge_remove_list,
            determine_action,
            extract_terms_from_code_with_params,
            extract_term_skeleton,
        )

        removed_terms_per_dim = dict(state.get("removed_terms_per_dim") or {})
        final_analysis_metadata = None

        # Build previous eval map for state machine
        prev_term_evals = {}
        prev_metadata = state.get("scientist_analysis_metadata")
        if prev_metadata:
            prev_term_evals = prev_metadata.get("term_evaluations", {})

        if analysis_result:
            # Convert Pydantic model to dict to add extra fields
            final_analysis_metadata = analysis_result.model_dump()

            # Extract list from new schema and convert to dict for processing
            # Schema change: term_evaluations (dict) -> term_evaluations_list (list)
            term_evals_list = final_analysis_metadata.get("term_evaluations_list", [])
            term_evals_dict = {}
            for item in term_evals_list:
                func_name = item.get("function_name")
                evals = item.get("evaluations", [])
                if func_name:
                    term_evals_dict[func_name] = evals

            # Map impacts by (func_name, term_idx)
            # Impact list is like: {'x0_t': [{'term_idx': 0, 'impact': 'positive'}, ...]}

            # Use current codes/params to extract term strings by index
            # This is needed because LLM output no longer contains 'term' field
            current_codes = current_ode_system.get("codes", {})
            current_params = current_ode_system.get("dim_params", {})

            # Process each function's evaluations
            processed_evals_dict = {}

            for func_name, evals_list in term_evals_dict.items():
                impacts_list = term_impacts.get(func_name, [])

                # Create a map for quick lookup: term_idx -> impact info
                impact_map = {item["term_idx"]: item for item in impacts_list}

                # Build skeleton map for previous actions in this dimension
                dim_prev_evals = prev_term_evals.get(func_name, [])
                skeleton_to_prev_action = {}
                for pe in dim_prev_evals:
                    p_term = pe.get("term")
                    p_action = pe.get("action")
                    if p_term and p_action:
                        p_skeleton = extract_term_skeleton(p_term)
                        skeleton_to_prev_action[p_skeleton] = p_action

                # Extract term strings for this function
                code = current_codes.get(func_name, "")
                p_vals = current_params.get(func_name, [])
                term_list = extract_terms_from_code_with_params(code, p_vals)

                processed_evals = []
                # evals_list from LLM corresponds to terms in order.
                # LLM output order implies term index.

                for idx, eval_item in enumerate(evals_list):
                    # Inject term string (retrieved from code, not LLM)
                    term_str = ""
                    skeleton_str = ""
                    if idx < len(term_list):
                        term_str = term_list[idx]
                        # Use skeleton format for Scientist/User view (C instead of specific numbers)
                        skeleton_str = extract_term_skeleton(term_str)

                    # Store the skeleton (C*...) in metadata as requested
                    eval_item["term"] = skeleton_str

                    # Get semantic quality from LLM
                    semantic = eval_item.get("semantic_quality", "neutral")

                    # Get measured impact
                    imp_info = impact_map.get(idx)
                    if imp_info:
                        impact = imp_info["impact"]
                    else:
                        impact = "neutral"  # Default if not found

                    # Get previous action via skeleton mapping
                    term_skeleton = extract_term_skeleton(term_str) if term_str else ""
                    prev_action = skeleton_to_prev_action.get(term_skeleton)

                    # Determine Action deterministically
                    action = determine_action(semantic, impact, prev_action=prev_action)

                    # Update the item with system-determined values
                    eval_item["performance_impact"] = impact
                    eval_item["action"] = action

                    # Update remove list if needed
                    if action == "remove":
                        if term_str:
                            existing = removed_terms_per_dim.get(func_name, [])
                            removed_terms_per_dim[func_name] = merge_remove_list(
                                existing, [term_str]
                            )

                    processed_evals.append(eval_item)

                # Update the processed list for this function
                processed_evals_dict[func_name] = processed_evals

            # Save back to metadata in the OLD dict format for compatibility
            # We reconstruct 'term_evaluations' key which wasn't in LLM output but is needed by system
            final_analysis_metadata["term_evaluations"] = processed_evals_dict
            # Remove the list key to avoid confusion
            if "term_evaluations_list" in final_analysis_metadata:
                del final_analysis_metadata["term_evaluations_list"]

        return {
            **state,
            "prev_best_ode_system": best_ode_system,  # Save current best as prev for next iteration
            "research_notebook": new_notebook,
            "scientist_analysis_prompt": obs_prompt,
            "scientist_analysis_metadata": final_analysis_metadata,
            "global_improvement": global_improvement,
            "local_improvement": local_improvement,
            "removed_terms_per_dim": removed_terms_per_dim
            if removed_terms_per_dim
            else None,
            "token_usage": current_usage,  # Persist updated token usage
            "dim_term_details": term_impacts,  # Persist ablation logs
        }

    def _call_llm_for_analysis_and_record(
        self,
        observations: list,
        outcome_type: str,
        describe: str,
        notebook: dict,
        removed_terms_per_dim: dict = None,
        current_iteration: int = None,
        total_iterations: int = None,
    ):
        """Helper method to call LLM for observation analysis and notebook recording."""
        from prompt import make_analysis_and_record_prompt
        from with_structured_output import ExperimentAnalysis

        prompt = make_analysis_and_record_prompt(
            observations=observations,
            outcome_type=outcome_type,
            describe=describe,
            notebook=notebook,
            removed_terms_per_dim=removed_terms_per_dim,
            current_iteration=current_iteration,
            total_iterations=total_iterations,
        )

        try:
            response = self.scientist_llm.with_structured_output(
                ExperimentAnalysis, include_raw=True
            ).invoke(prompt)

            # Extract parsed result and raw usage
            analysis_result = response["parsed"]
            raw_msg = response["raw"]

            # Log token usage
            token_usage_stats = {}
            if hasattr(raw_msg, "usage_metadata"):
                usage = raw_msg.usage_metadata
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                LOGGER.info(
                    "[Scientist Token Usage] Input: %s, Output: %s",
                    input_tokens,
                    output_tokens,
                )
                token_usage_stats = {"input": input_tokens, "output": output_tokens}

            return analysis_result, prompt, token_usage_stats
        except Exception as e:
            LOGGER.exception("Error calling LLM for analysis and record: %s", e)
            return None, prompt, {}

    @safe_node
    def update_global_best(self, state: GraphState) -> GraphState:
        """Update Global Best ODE system if current ODE system is better (Dimension-wise Independent).

        Updates each dimension of the Global Best ODE system independently based on MSE scores.
        Global Best becomes a composite (chimera) of the best equations found across iterations.
        """
        import copy

        current_ode_system = state.get("current_ode_system")
        best_ode_system = state.get("best_ode_system")

        if not current_ode_system:
            return state

        # Initialize best_ode_system if None
        if best_ode_system is None:
            # First iteration: Global Best is just the current ODE system
            new_best_ode_system = copy.deepcopy(current_ode_system)

            # Initialize iteration tracking
            best_iteration_per_dim = {}
            current_iter = state.get("current_iteration", 0)
            if "dim_scores" in new_best_ode_system:
                for dim in new_best_ode_system["dim_scores"]:
                    best_iteration_per_dim[dim] = current_iter
        else:
            # Subsequent iterations: Update dimension-wise
            new_best_ode_system = copy.deepcopy(best_ode_system)
            best_iteration_per_dim = state.get("best_iteration_per_dim", {}).copy()
            current_iter = state.get("current_iteration", 0)

            curr_scores = current_ode_system.get("dim_scores", {})
            best_scores = new_best_ode_system.get("dim_scores", {})

            # Iterate over all dimensions present in current_ode_system
            for dim_name, curr_mse in curr_scores.items():
                best_mse = best_scores.get(dim_name, float("inf"))

                # Update if current MSE is strictly better (lower)
                # Independent update for each dimension
                if curr_mse < best_mse:
                    if should_log_score_improvement(best_mse, curr_mse):
                        LOGGER.info(
                            "[Global Best Update] %s MSE improved: "
                            "%.12e -> %.12e (Iter %s)",
                            dim_name,
                            best_mse,
                            curr_mse,
                            current_iter,
                        )

                    # Update all fields for this dimension
                    new_best_ode_system["dim_scores"][dim_name] = curr_mse

                    if (
                        "codes" in current_ode_system
                        and dim_name in current_ode_system["codes"]
                    ):
                        new_best_ode_system["codes"][dim_name] = current_ode_system[
                            "codes"
                        ][dim_name]

                    if (
                        "dim_params" in current_ode_system
                        and dim_name in current_ode_system["dim_params"]
                    ):
                        new_best_ode_system["dim_params"][dim_name] = (
                            current_ode_system["dim_params"][dim_name]
                        )

                    # Update iteration tracker
                    best_iteration_per_dim[dim_name] = current_iter

                    # Assuming opt_methods and other per-dim metadata might need update too if tracked
                    # But GraphState mainly tracks codes, params, scores for best_ode_system

        return {
            **state,
            "best_ode_system": new_best_ode_system,
            "best_iteration_per_dim": best_iteration_per_dim,
        }

    def link_nodes(self):
        """Link nodes in the state graph."""
        if hasattr(self, "graph") and len(self.graph.nodes) > 0:
            self.graph = StateGraph(GraphState)

        self.graph.add_node("make_func_list", self.make_func_list)
        self.graph.add_node("make_prompt", self.make_prompt)
        self.graph.add_node("sampler", self.sampler)
        self.graph.add_node("parse_candidates", self.parse_candidates)

        self.graph.add_node("evaluate_candidates", self.evaluate_candidates)
        self.graph.add_node("select_best", self.select_best)

        # Scientist nodes
        self.graph.add_node(
            "analyze_and_record_observation", self.analyze_and_record_observation
        )

        self.graph.add_node("update_global_best", self.update_global_best)

        # Connect edges
        self.graph.add_edge(START, "make_func_list")

        # make_func_list -> check error
        self.graph.add_conditional_edges(
            "make_func_list",
            self.check_error_and_route,
            {"restart": "make_func_list", "continue": "make_prompt"},
        )

        # make_prompt -> check error
        self.graph.add_conditional_edges(
            "make_prompt",
            self.check_error_and_route,
            {"restart": "make_func_list", "continue": "sampler"},
        )

        # sampler -> check error -> parse_candidates
        self.graph.add_conditional_edges(
            "sampler",
            self.check_error_and_route,
            {"restart": "make_func_list", "continue": "parse_candidates"},
        )

        # parse_candidates -> check error -> evaluate_candidates
        self.graph.add_conditional_edges(
            "parse_candidates",
            self.check_error_and_route,
            {"restart": "make_func_list", "continue": "evaluate_candidates"},
        )

        # evaluate_candidates -> check error -> select_best
        self.graph.add_conditional_edges(
            "evaluate_candidates",
            self.check_error_and_route,
            {"restart": "make_func_list", "continue": "select_best"},
        )

        self.graph.add_conditional_edges(
            "select_best",
            self.check_error_and_route,
            {"restart": "make_func_list", "continue": "analyze_and_record_observation"},
        )

        # analyze_and_record_observation -> update_global_best
        self.graph.add_edge("analyze_and_record_observation", "update_global_best")

        # update_global_best -> check error -> END
        self.graph.add_conditional_edges(
            "update_global_best",
            self.check_error_and_route,
            {"restart": "make_func_list", "continue": END},
        )

        self.app = self.graph.compile()
        return self.app
