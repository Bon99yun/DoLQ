"""
I/O utility functions for LLM ODE evolution.

This module contains file I/O, JSON serialization, and logging functions.
"""

import json
import logging
import os
import sys

from typing import Any, Dict, List
from pathlib import Path

import numpy as np
from utils import code_to_equation


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGGER_NAME = "dolq"
_CONSOLE_HANDLER_TAG = "_dolq_console_handler"
_FILE_HANDLER_TAG = "_dolq_file_handler"


def get_experiment_logger(name: str = None) -> logging.Logger:
    """Return the shared DoLQ experiment logger or a named child logger."""
    if not name:
        return logging.getLogger(LOGGER_NAME)
    if name == LOGGER_NAME or name.startswith(f"{LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def configure_experiment_logging(
    logs_dir: Path = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure consistent console and optional per-run file logging."""
    logger = get_experiment_logger()
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    if not any(
        getattr(handler, _CONSOLE_HANDLER_TAG, False) for handler in logger.handlers
    ):
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        setattr(console_handler, _CONSOLE_HANDLER_TAG, True)
        logger.addHandler(console_handler)

    if logs_dir is not None:
        logs_dir = Path(logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        run_log_path = logs_dir / "run.log"
        existing_file_paths = {
            Path(handler.baseFilename)
            for handler in logger.handlers
            if getattr(handler, _FILE_HANDLER_TAG, False)
            and hasattr(handler, "baseFilename")
        }
        if run_log_path not in existing_file_paths:
            file_handler = logging.FileHandler(run_log_path, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            setattr(file_handler, _FILE_HANDLER_TAG, True)
            logger.addHandler(file_handler)

    logging.captureWarnings(True)
    return logger


def write_json_atomic(
    path: Path,
    data: Any,
    *,
    ensure_ascii: bool = False,
    indent: int = 2,
    fsync: bool = True,
) -> None:
    """Write JSON through a temporary file and atomically replace the target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)
            if fsync:
                f.flush()
                os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def write_text_atomic(path: Path, text: str, *, fsync: bool = True) -> None:
    """Write text through a temporary file and atomically replace the target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(text)
            if fsync:
                f.flush()
                os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


LOGGER = get_experiment_logger(__name__)


def generate_logs_dir_name(
    use_var_desc: bool,
    use_differential_evolution: bool,
    start_time: str,
    use_scientist: bool = False,
    use_gt: bool = False,
    forget_prob: float = 0.1,
) -> str:
    """Generate log directory name based on experiment settings.

    Args:
        use_var_desc: Whether to use variable description
        use_differential_evolution: Whether to use Differential Evolution
        start_time: Experiment start time (YYYYMMDD_HHMMSS format)
        use_scientist: Whether to use scientist agent
        use_gt: Whether to use ground truth target

    Returns:
        Generated directory name string
    """
    flag_parts = []

    # Add gt prefix if using ground truth
    if use_gt:
        flag_parts.append("gt")


    if use_var_desc:
        flag_parts.append("desc")
    if use_differential_evolution:
        flag_parts.append("de")
    if use_scientist:
        flag_parts.append("scientist")

    # forget*: legacy token (unchanged) — str replace keeps compatibility with old runs
    prob_str = str(forget_prob).replace(".", "")
    flag_parts.append(f"forget{prob_str}")

    flag_str = "_".join(flag_parts)
    return f"{flag_str}_{start_time}"


def convert_to_serializable(value: Any) -> Any:
    """Convert Python objects to JSON-serializable format.

    Args:
        value: Any Python object

    Returns:
        JSON-serializable representation
    """
    if value is None:
        return None
    elif isinstance(value, (str, int, float, bool)):
        return value

    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, list):
        return [convert_to_serializable(item) for item in value]
    elif isinstance(value, dict):
        return {k: convert_to_serializable(v) for k, v in value.items()}
    elif callable(value):
        return str(value)
    else:
        return str(value)


def summarize_runtime_metrics(
    iteration_runtime_metrics_history=None, stage_runtime_metrics_history=None
) -> Dict[str, Any]:
    iteration_metrics = [
        metrics for metrics in (iteration_runtime_metrics_history or []) if metrics
    ]
    iteration_elapsed_seconds = [
        float(metrics.get("elapsed_seconds", 0.0) or 0.0)
        for metrics in iteration_metrics
    ]
    iteration_cpu_seconds = [
        float(metrics.get("process_cpu_seconds_delta", 0.0) or 0.0)
        for metrics in iteration_metrics
    ]
    iteration_count = len(iteration_metrics)
    total_iteration_elapsed = sum(iteration_elapsed_seconds)

    stage_summary = {}
    for stage_metrics in stage_runtime_metrics_history or []:
        for stage_name, metrics in (stage_metrics or {}).items():
            bucket = stage_summary.setdefault(
                stage_name,
                {
                    "count": 0,
                    "total_elapsed_seconds": 0.0,
                    "total_process_cpu_seconds_delta": 0.0,
                },
            )
            bucket["count"] += 1
            bucket["total_elapsed_seconds"] += float(
                metrics.get("elapsed_seconds", 0.0) or 0.0
            )
            bucket["total_process_cpu_seconds_delta"] += float(
                metrics.get("process_cpu_seconds_delta", 0.0) or 0.0
            )

    return {
        "iteration": {
            "count": iteration_count,
            "total_elapsed_seconds": total_iteration_elapsed,
            "average_elapsed_seconds": (
                total_iteration_elapsed / iteration_count if iteration_count else 0.0
            ),
            "total_process_cpu_seconds_delta": sum(iteration_cpu_seconds),
        },
        "stage": stage_summary,
    }


def save_result(
    result: Dict[str, Any],
    index: int,
    logs_dir: Path,
    iteration_json_dir: Path,
    problem_name: str,
) -> None:
    """Save evolution results as JSON.

    Args:
        result: Result dictionary from evolution
        index: Evolution iteration index
        logs_dir: Path to logs directory
        iteration_json_dir: Path to iteration_json directory
        problem_name: Problem name for file naming
    """
    try:
        # Prepare result for JSON (exclude non-serializable items)
        # Keys to exclude from JSON output (internal state or user requested removal)
        exclude_keys = {
            "func_list",
            "initial_state",
            "prev_best_ode_system",
            "previous_generation_pair",
            "global_improvement",
            "local_improvement",
        }

        # Prepare result for JSON
        result_for_json = {}
        if result is None:
            LOGGER.warning(
                "Result is None for iteration %s of %s; saving empty dict",
                index,
                problem_name,
            )
            result_for_json = {}
        else:
            for k, v in result.items():
                if k in exclude_keys:
                    continue

                # exclude removed_terms_per_dim if None (cleaner log)
                if k in ["removed_terms_per_dim"] and v is None:
                    continue

                result_for_json[k] = v

        if result is not None:
            for runtime_key in ["iteration_runtime_metrics", "stage_runtime_metrics"]:
                if runtime_key in result and result.get(runtime_key) is not None:
                    result_for_json[runtime_key] = result.get(runtime_key)

        # Warn if result_for_json is empty
        if not result_for_json:
            LOGGER.warning(
                "Result JSON is empty for iteration %s of %s",
                index,
                problem_name,
            )

        json_result = convert_to_serializable(result_for_json)
        save_path = iteration_json_dir / f"{problem_name}_{index}.json"

        write_json_atomic(save_path, json_result, ensure_ascii=False, indent=2)

    except Exception as e:
        LOGGER.exception(
            "Error saving result for iteration %s of %s: %s",
            index,
            problem_name,
            e,
        )
        # Try to save error info
        try:
            save_path = iteration_json_dir / f"{problem_name}_{index}.json"
            error_info = {
                "error": str(e),
                "iteration": index,
                "problem_name": problem_name,
            }
            write_json_atomic(save_path, error_info, ensure_ascii=False, indent=2)
        except OSError:
            LOGGER.exception(
                "Failed to write fallback error JSON for iteration %s of %s",
                index,
                problem_name,
            )


def update_generated_equations(
    logs_dir: Path,
    iteration: int,
    evaluated_candidates: List[Dict[str, Any]],
    best_ode_system: Dict[str, Any] = None,
    iteration_runtime_metrics: Dict[str, Any] = None,
    stage_runtime_metrics: Dict[str, Dict[str, Any]] = None,
) -> None:
    """Update generated_equations.json with all candidates from this iteration.

    This function is called after each evolution iteration to provide real-time logging.

    Args:
        logs_dir: Path to logs directory
        iteration: Current iteration number
        evaluated_candidates: List of all evaluated ODE system pairs from this iteration
        best_ode_system: Current global best ODE system (optional, for reference)
    """
    from datetime import datetime

    report_dir = logs_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    equations_file = report_dir / "generated_equations.json"
    default_data = {
        "description": "All generated ODE equations and their MSE scores per iteration",
        "iterations": [],
        "global_best": None,
    }

    # Load existing data or create new structure
    if equations_file.exists():
        try:
            with open(equations_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = default_data.copy()
    else:
        data = default_data.copy()

    # Build candidates list for this iteration
    candidates_data = []
    for idx, candidate in enumerate(evaluated_candidates or []):
        candidate_entry = {
            "candidate_index": idx,
            "equations": convert_to_serializable(candidate.get("codes", {})),
            "dim_scores": convert_to_serializable(candidate.get("dim_scores", {})),
            "params": convert_to_serializable(candidate.get("dim_params", {})),
            "optimization_methods": convert_to_serializable(
                candidate.get("dim_opt_methods", {})
            ),
            "reasoning": candidate.get("pair_reasoning", ""),
        }
        candidates_data.append(candidate_entry)

    best_candidate_scores = (
        convert_to_serializable(candidates_data[0].get("dim_scores", {}))
        if candidates_data
        else None
    )

    # Create iteration entry
    iteration_entry = {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "num_candidates": len(candidates_data),
        "best_candidate_scores": best_candidate_scores,
        "candidates": candidates_data,
        "iteration_runtime_metrics": convert_to_serializable(iteration_runtime_metrics),
        "stage_runtime_metrics": convert_to_serializable(stage_runtime_metrics),
    }

    # Update or append iteration
    existing_iterations = {
        entry["iteration"]: i for i, entry in enumerate(data["iterations"])
    }
    if iteration in existing_iterations:
        data["iterations"][existing_iterations[iteration]] = iteration_entry
    else:
        data["iterations"].append(iteration_entry)

    if best_ode_system:
        data["global_best"] = {
            "equations": convert_to_serializable(best_ode_system.get("codes", {})),
            "dim_scores": convert_to_serializable(
                best_ode_system.get("dim_scores", {})
            ),
            "params": convert_to_serializable(best_ode_system.get("dim_params", {})),
        }

    write_json_atomic(equations_file, data, ensure_ascii=False, indent=2)


def print_experiment_config(
    problem_name: str,
    dim: int,
    max_params: int,
    evolution_num: int,
    recursion_limit: int,
    timeout: int,
    max_retries: int,
    model_name: str,
    use_var_desc: bool,
    use_differential_evolution: bool,
    logs_dir: Path,
    start_time: str,
    use_scientist: bool = False,
    use_gt: bool = False,
    forget_prob: float = 0.1,
    de_tolerance: float = 1e-20,
    bfgs_tolerance: float = 1e-25,
    num_equations: int = 3,
) -> None:
    """Log experiment configuration to console and the per-run log file."""
    lines = [
        "=" * 60,
        "Evolution Experiment Configuration",
        "=" * 60,
        f"Problem name: {problem_name}",
        f"Dimension: {dim}D",
        f"Max parameters: {max_params}",
        f"Evolution iterations: {evolution_num}",
        f"Recursion limit: {recursion_limit}",
        f"Timeout: {timeout}s",
        f"Max retries: {max_retries}",
        f"Model name: {model_name}",
        f"Use variable description: {use_var_desc}",
        f"Use Differential Evolution: {use_differential_evolution}",
        f"Use scientist agent: {use_scientist}",
        f"Use ground truth targets: {use_gt}",
        f"Forget probability: {forget_prob}",
        f"DE tolerance: {de_tolerance}",
        f"BFGS tolerance: {bfgs_tolerance}",
        f"Num equations: {num_equations}",
        f"Log directory: {logs_dir}",
        f"Start time: {start_time}",
        "=" * 60,
    ]
    for line in lines:
        LOGGER.info(line)


def save_final_report(
    logs_dir: Path,
    problem_name: str,
    evolution_num: int,
    duration_str: str,
    total_error_count: int,
    config_info: Dict[str, Any],
    best_scores_per_dim: Dict[str, float],
    best_iteration_per_dim: Dict[str, int],
    best_code_per_dim: Dict[str, str],
    best_params_per_dim: Dict[str, List[float]],
    research_notebook: Dict[str, Any] = None,
    best_iteration: int = None,
    iteration_runtime_metrics_history=None,
    stage_runtime_metrics_history=None,
) -> None:
    """Save final experiment reports (JSON, Text, Buffer Dump).

    Args:
        logs_dir: Path to logs directory
        problem_name: Problem name
        evolution_num: Number of iterations
        duration_str: Formatted duration string
        total_error_count: Total accumulated errors
        config_info: Configuration dictionary
        best_scores_per_dim: Best scores per dimension
        best_iteration_per_dim: Best iteration per dimension
        best_code_per_dim: Best code per dimension
        best_params_per_dim: Best parameters per dimension
        research_notebook: Final state of Research Notebook
    """
    report_dir = logs_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    runtime_metrics_summary = summarize_runtime_metrics(
        iteration_runtime_metrics_history=iteration_runtime_metrics_history,
        stage_runtime_metrics_history=stage_runtime_metrics_history,
    )
    resource_usage = (config_info or {}).get("resource_usage", {})

    # 1. Prepare Final Report Dictionary
    final_report = {
        "problem_name": problem_name,
        "evolution_num": evolution_num,
        "total_duration": duration_str,
        "total_error_count": total_error_count,
        "config": config_info or {},
        "best_scores_per_dim": best_scores_per_dim,
        "best_iteration_per_dim": best_iteration_per_dim,
        "best_code_per_dim": best_code_per_dim,
        "best_params_per_dim": best_params_per_dim,
        # Pair-based architecture fields
        "best_iteration": best_iteration,
        # Scientist logging fields
        "research_notebook": research_notebook or {},
        "runtime_metrics_summary": runtime_metrics_summary,
        "resource_usage": resource_usage,
    }

    # 3. Save Final Report JSON
    write_json_atomic(
        report_dir / "final_report.json",
        convert_to_serializable(final_report),
        ensure_ascii=False,
        indent=4,
    )

    # 4. Save Final Report Text
    report_lines = []

    def add(line: str = "") -> None:
        report_lines.append(line)

    add(f"Final Evolution Report: {problem_name}")
    add("=======================================")

    # Write Config Info
    add("\n[Experiment Configuration]")
    add(f"  Total Duration: {duration_str}")
    add(f"  Total Errors Occurred: {total_error_count}")
    if config_info:
        for k, v in config_info.items():
            if k == "resource_usage":
                continue
            add(f"  {k}: {v}")

    # Write Scientist Info
    if research_notebook:
        add("\n[Scientist Report]")
        add("  Research Notebook Status:")
        add(
            f"    Structural Learnings: {len(research_notebook.get('structural_learnings', []))}"
        )
        add(
            "    Accumulated Insight Length: "
            f"{len(research_notebook.get('accumulated_insight', ''))} chars"
        )
        if research_notebook.get("next_experiment_suggestion"):
            add(
                "    Next Experiment Suggestion: "
                f"{research_notebook.get('next_experiment_suggestion')[:100]}..."
            )

    add("\n[Execution Environment]")
    if resource_usage:
        add(f"  Compute Device: {resource_usage.get('compute_device')}")
        add(f"  GPU Used: {resource_usage.get('gpu_used')}")
        add(f"  CPU Model: {resource_usage.get('cpu_model')}")
        add(f"  Logical CPU Count: {resource_usage.get('logical_cpu_count')}")
        add(f"  Available CPU Cores: {resource_usage.get('available_cpu_cores')}")
        add(f"  Parallel Worker Config: {resource_usage.get('parallel_n_jobs_config')}")
        add(
            "  Effective Parallel Workers: "
            f"{resource_usage.get('effective_parallel_workers')}"
        )
        add(
            f"  Total Memory: {resource_usage.get('total_memory_gib')} GiB "
            f"({resource_usage.get('total_memory_bytes')} bytes)"
        )
        add(
            f"  Peak Process Memory: {resource_usage.get('peak_process_memory_gib')} GiB "
            f"({resource_usage.get('peak_process_memory_bytes')} bytes)"
        )
        add(f"  OS: {resource_usage.get('platform')}")
        add(f"  Machine: {resource_usage.get('machine')}")
        add(
            f"  Python: {resource_usage.get('python_version')} "
            f"({resource_usage.get('python_executable')})"
        )
        add(f"  Conda Env: {resource_usage.get('conda_env')}")
        add(f"  NumPy Version: {resource_usage.get('numpy_version')}")
    else:
        add("  Resource usage metadata not available.")

    add("\n[Runtime Metrics]")
    iteration_summary = runtime_metrics_summary.get("iteration", {})
    add(f"  Iterations Tracked: {iteration_summary.get('count', 0)}")
    add(
        "  Total Iteration Wall Time: "
        f"{iteration_summary.get('total_elapsed_seconds', 0.0):.6f}s"
    )
    add(
        "  Average Iteration Wall Time: "
        f"{iteration_summary.get('average_elapsed_seconds', 0.0):.6f}s"
    )
    add(
        "  Total Iteration CPU Time: "
        f"{iteration_summary.get('total_process_cpu_seconds_delta', 0.0):.6f}s"
    )

    stage_summary = runtime_metrics_summary.get("stage", {})
    if stage_summary:
        add("  Stage Summary:")
        for stage_name in sorted(stage_summary.keys()):
            stage_metrics = stage_summary[stage_name]
            add(
                f"    {stage_name}: count={stage_metrics.get('count', 0)}, "
                f"wall={stage_metrics.get('total_elapsed_seconds', 0.0):.6f}s, "
                f"cpu={stage_metrics.get('total_process_cpu_seconds_delta', 0.0):.6f}s"
            )

    # Write Best ODE System Pair (Unified View)
    add("\n[Best ODE System Pair]")
    if best_iteration is not None:
        add(f"  Achieved at Iteration: {best_iteration}")
        add("  " + "=" * 40)

        # Sort dimensions for consistent output (x0, x1, ...)
        sorted_dims = sorted(best_code_per_dim.keys())

        for dim in sorted_dims:
            score = best_scores_per_dim.get(dim, float("inf"))
            code = best_code_per_dim.get(dim, "N/A")
            params = best_params_per_dim.get(dim, [])

            add(f"  Dimension: {dim}")
            add(f"    Score (MSE): {score:.6e}")
            add(f"    Parameters: {params}")
            add("    Code:")
            # Indent code block
            code_indented = "\n".join(["      " + line for line in code.split("\n")])
            add(code_indented)

            # Equation (Readable)
            try:
                eqn_str = code_to_equation(code, dim, params)
                add(f"    Equation: {eqn_str}")
            except Exception as e:
                add(f"    Equation: (Error transforming to equation: {e})")

            add("  " + "-" * 40)
    else:
        add("  No best ODE system recorded.")

    write_text_atomic(report_dir / "final_report.txt", "\n".join(report_lines) + "\n")
