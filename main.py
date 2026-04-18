"""
Main entry point for LLM-guided ODE evolution experiment.
"""

import os
import warnings
import argparse
import platform
import sys
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from tqdm import trange

from evolution import (
    evolution_chain,
    init_score,
    start_runtime_metrics,
    finish_runtime_metrics,
)
from config import (
    DEFAULT_NUM_EQUATIONS,
    DE_TOLERANCE,
    BFGS_TOLERANCE,
    REMOVED_TERMS_FORGET_PROBABILITY,
    PARALLEL_N_JOBS,
)
from init_func_str import *
from data_loader import load_dataframes, create_describe, create_df_dict
from io_utils import (
    configure_experiment_logging,
    generate_logs_dir_name,
    get_experiment_logger,
    save_result,
    print_experiment_config,
    save_final_report,
    update_generated_equations,
)
from compare import is_better_than

warnings.filterwarnings("ignore")
load_dotenv(".env")
LOGGER = get_experiment_logger(__name__)


def _read_linux_cpu_model() -> Optional[str]:
    cpuinfo_path = Path("/proc/cpuinfo")
    if not cpuinfo_path.exists():
        return None

    try:
        for line in cpuinfo_path.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                _, _, value = line.partition(":")
                return value.strip() or None
    except OSError:
        return None

    return None


def _get_total_memory_bytes() -> Optional[int]:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        if isinstance(page_size, int) and isinstance(phys_pages, int):
            return page_size * phys_pages
    except (ValueError, OSError, AttributeError):
        pass

    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        try:
            for line in meminfo_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
        except (OSError, ValueError):
            return None

    return None


def _get_peak_process_memory_bytes() -> Optional[int]:
    try:
        import resource

        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            return int(peak_rss)
        return int(peak_rss) * 1024
    except (ImportError, AttributeError, ValueError, OSError):
        return None


def _bytes_to_gib(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return round(value / (1024**3), 3)


def collect_execution_environment() -> Dict[str, Any]:
    logical_cpu_count = os.cpu_count()
    available_cpu_cores = None
    if hasattr(os, "sched_getaffinity"):
        try:
            available_cpu_cores = len(os.sched_getaffinity(0))
        except OSError:
            available_cpu_cores = None

    total_memory_bytes = _get_total_memory_bytes()
    effective_parallel_workers = (
        PARALLEL_N_JOBS if PARALLEL_N_JOBS > 0 else logical_cpu_count
    )

    return {
        "compute_device": "cpu",
        "gpu_used": False,
        "cpu_model": _read_linux_cpu_model() or platform.processor() or None,
        "logical_cpu_count": logical_cpu_count,
        "available_cpu_cores": available_cpu_cores,
        "parallel_n_jobs_config": PARALLEL_N_JOBS,
        "effective_parallel_workers": effective_parallel_workers,
        "total_memory_bytes": total_memory_bytes,
        "total_memory_gib": _bytes_to_gib(total_memory_bytes),
        "os_name": platform.system(),
        "os_release": platform.release(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "numpy_version": np.__version__,
        "peak_process_memory_bytes": None,
        "peak_process_memory_gib": None,
    }


def update_peak_process_memory(resource_usage: Dict[str, Any]) -> Dict[str, Any]:
    peak_memory_bytes = _get_peak_process_memory_bytes()
    return {
        **resource_usage,
        "peak_process_memory_bytes": peak_memory_bytes,
        "peak_process_memory_gib": _bytes_to_gib(peak_memory_bytes),
    }


def print_execution_environment(resource_usage: Dict[str, Any]) -> None:
    lines = [
        "[Execution Environment]",
        f"  Compute device: {resource_usage.get('compute_device')}",
        f"  GPU used: {resource_usage.get('gpu_used')}",
        f"  CPU model: {resource_usage.get('cpu_model')}",
        (
            f"  Logical CPU count: {resource_usage.get('logical_cpu_count')}"
            f" (available to process: {resource_usage.get('available_cpu_cores')})"
        ),
        (
            f"  Parallel workers: config={resource_usage.get('parallel_n_jobs_config')}, "
            f"effective={resource_usage.get('effective_parallel_workers')}"
        ),
        (
            f"  Total memory: {resource_usage.get('total_memory_gib')} GiB"
            f" ({resource_usage.get('total_memory_bytes')} bytes)"
        ),
        (
            f"  OS: {resource_usage.get('platform')} | "
            f"machine={resource_usage.get('machine')}"
        ),
        (
            f"  Python: {resource_usage.get('python_version')}"
            f" ({resource_usage.get('python_executable')})"
        ),
        (
            f"  Conda env: {resource_usage.get('conda_env')} | "
            f"NumPy: {resource_usage.get('numpy_version')}"
        ),
    ]
    for line in lines:
        LOGGER.info(line)


def get_configurable(config: RunnableConfig) -> dict:
    """Extract configurable dictionary from RunnableConfig."""
    configurable = {}
    if hasattr(config, "configurable"):
        configurable_attr = config.configurable
        if isinstance(configurable_attr, dict):
            configurable = configurable_attr
        elif configurable_attr is not None:
            try:
                configurable = dict(configurable_attr)
            except (TypeError, ValueError):
                configurable = {}
    elif hasattr(config, "get"):
        configurable = config.get("configurable", {})
    else:
        configurable = getattr(config, "configurable", {})
        if not isinstance(configurable, dict):
            try:
                configurable = dict(configurable) if configurable is not None else {}
            except (TypeError, ValueError):
                configurable = {}
    return configurable


def create_initial_state(
    evo: evolution_chain,
    df_dict: Dict[str, Any],
    init_func_str_list: List[str],
    max_params: int,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """Create initial state for evolution."""
    configurable = get_configurable(config)
    describe = configurable.get("describe", "")

    # Create initial params list
    initial_params_list = [np.ones(max_params) for _ in init_func_str_list]

    # Calculate initial scores
    score = init_score(evo, init_func_str_list, df_dict, initial_params_list)

    return {
        "generated_code": init_func_str_list,
        "params_list": initial_params_list,
        "score": score,
        "describe": describe,
        "use_var_desc": configurable.get("use_var_desc", False),
        "use_differential_evolution": configurable.get(
            "use_differential_evolution", False
        ),
        "total_error_count": 0,
    }


def run_evolution(
    evo: evolution_chain,
    initial_state: Dict[str, Any],
    config: RunnableConfig,
    evolution_num: int,
    logs_dir: Path,
    iteration_json_dir: Path,
    problem_name: str,
    use_scientist: bool = False,
    func_names: List[str] = None,
    config_info: Dict[str, Any] = None,
) -> List[Dict[str, Any]]:
    """Execute evolution process with ODE system-based tracking."""
    start_time_dt = datetime.now()
    result_list = [initial_state]

    # Track best ODE system (new architecture)
    best_ode_system: Optional[Dict[str, Any]] = None
    best_iteration: int = 0

    # For backward compatibility with save_final_report
    best_scores_per_dim: Dict[str, float] = {}
    best_code_per_dim: Dict[str, str] = {}
    best_params_per_dim: Dict[str, List[float]] = {}
    best_iteration_per_dim: Dict[str, int] = {}

    configurable = get_configurable(config)

    progress_is_interactive = sys.stderr.isatty()
    for index in trange(
        1,
        evolution_num + 1,
        desc="Evolution",
        disable=not progress_is_interactive,
        dynamic_ncols=progress_is_interactive,
    ):
        iteration_runtime_metrics = start_runtime_metrics()
        prev_state = result_list[index - 1]

        # Build input for evolution step (pass best_ode_system)
        input_data = {
            "generated_code": prev_state.get("generated_code"),
            "score": prev_state.get("score"),
            "params_list": prev_state.get("params_list"),
            "describe": configurable.get("describe", ""),
            "use_var_desc": configurable.get("use_var_desc", False),
            "use_differential_evolution": configurable.get(
                "use_differential_evolution", False
            ),
            "use_scientist": use_scientist,
            "total_error_count": prev_state.get("total_error_count", 0),
            "research_notebook": prev_state.get("research_notebook"),
            # Pair-based fields
            "best_ode_system": best_ode_system,
            "prev_best_ode_system": prev_state.get("prev_best_ode_system"),
            "current_ode_system": prev_state.get("current_ode_system"),
            "previous_generation_ode_system": prev_state.get(
                "previous_generation_ode_system"
            ),
            # Remove list (accumulated across iterations)
            "removed_terms_per_dim": prev_state.get("removed_terms_per_dim"),
            # Iteration tracking
            "current_iteration": index,
            "total_iterations": evolution_num,
            "scientist_analysis_metadata": prev_state.get(
                "scientist_analysis_metadata"
            ),
        }

        evo_node = evo.link_nodes()
        try:
            result = evo_node.invoke(input_data, config)
        except Exception as e:
            LOGGER.exception("[Evolution Error] iteration %s: %s", index, e)
            raise
        finally:
            iteration_runtime_metrics = finish_runtime_metrics(
                iteration_runtime_metrics
            )

        result["iteration_runtime_metrics"] = iteration_runtime_metrics
        result_list.append(result)

        # Update best ODE system from result
        # Check both explicit best_ode_system from result and current_ode_system (in case graph didn't update best)
        candidates_for_best = []
        if result.get("best_ode_system"):
            candidates_for_best.append(result["best_ode_system"])
        if result.get("current_ode_system"):
            candidates_for_best.append(result["current_ode_system"])

        for candidate in candidates_for_best:
            if is_better_than(candidate, best_ode_system):
                best_ode_system = candidate
                best_iteration = index

        # Save results
        LOGGER.info(
            "[iteration %s/%s] wall=%.3fs cpu=%.3fs started_at=%s ended_at=%s",
            index,
            evolution_num,
            iteration_runtime_metrics.get("elapsed_seconds"),
            iteration_runtime_metrics.get("process_cpu_seconds_delta"),
            iteration_runtime_metrics.get("started_at"),
            iteration_runtime_metrics.get("ended_at"),
        )
        save_result(result, index, logs_dir, iteration_json_dir, problem_name)

        # Update generated_equations.json in real-time
        update_generated_equations(
            logs_dir=logs_dir,
            iteration=index,
            evaluated_candidates=result.get("evaluated_candidates", []),
            best_ode_system=best_ode_system,
            iteration_runtime_metrics=iteration_runtime_metrics,
            stage_runtime_metrics=result.get("stage_runtime_metrics"),
        )

    # Extract per-dimension data from best_ode_system
    # best_iteration_per_dim is now tracked in state (GraphState)
    final_state = result_list[-1] if result_list else {}
    if best_ode_system:
        for func_name in func_names:
            best_scores_per_dim[func_name] = best_ode_system["dim_scores"].get(
                func_name, float("inf")
            )
            best_code_per_dim[func_name] = best_ode_system["codes"].get(func_name, "")
            best_params_per_dim[func_name] = best_ode_system["dim_params"].get(
                func_name, []
            )

        # Get from state if available, otherwise default to last iteration
        state_best_iter = final_state.get("best_iteration_per_dim", {})
        if state_best_iter:
            best_iteration_per_dim.update(state_best_iter)
        else:
            # Fallback (should not happen with new evolution.py)
            for func_name in func_names:
                best_iteration_per_dim[func_name] = best_iteration

    # Calculate duration
    end_time = datetime.now()
    duration = end_time - start_time_dt
    total_seconds = duration.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    duration_str = f"{hours}h {minutes}m {seconds}s"

    # Get total error count
    total_error_count = (
        result_list[-1].get("total_error_count", 0) if result_list else 0
    )
    if config_info is not None:
        resource_usage = config_info.get("resource_usage") or {}
        config_info["resource_usage"] = update_peak_process_memory(resource_usage)
    iteration_runtime_metrics_history = [
        entry.get("iteration_runtime_metrics")
        for entry in result_list[1:]
        if entry.get("iteration_runtime_metrics") is not None
    ]
    stage_runtime_metrics_history = [
        entry.get("stage_runtime_metrics")
        for entry in result_list[1:]
        if entry.get("stage_runtime_metrics") is not None
    ]

    # Save Final Report
    save_final_report(
        logs_dir=logs_dir,
        problem_name=problem_name,
        evolution_num=evolution_num,
        duration_str=duration_str,
        total_error_count=total_error_count,
        config_info=config_info,
        best_scores_per_dim=best_scores_per_dim,
        best_iteration_per_dim=best_iteration_per_dim,
        best_code_per_dim=best_code_per_dim,
        best_params_per_dim=best_params_per_dim,
        research_notebook=result_list[-1].get("research_notebook"),
        best_iteration=best_iteration,
        iteration_runtime_metrics_history=iteration_runtime_metrics_history,
        stage_runtime_metrics_history=stage_runtime_metrics_history,
    )

    LOGGER.info("Final report saved to %s", logs_dir)

    return result_list, None


def main(
    problem_name: str,
    max_params: int,
    dim: int,
    evolution_num: int,
    use_var_desc: bool = False,
    use_differential_evolution: bool = False,
    use_scientist: bool = False,
    recursion_limit: int = 12,
    timeout: int = 180,
    max_retries: int = 2,
    sampler_model_name: str = "google/gemini-2.5-flash-lite",
    scientist_model_name: str = "google/gemini-2.5-flash-lite",
    num_equations: int = DEFAULT_NUM_EQUATIONS,
    de_tolerance: float = DE_TOLERANCE,
    bfgs_tolerance: float = BFGS_TOLERANCE,
    use_gt: bool = False,
    forget_prob: float = REMOVED_TERMS_FORGET_PROBABILITY,
) -> None:
    """Main execution function."""
    configure_experiment_logging()
    LOGGER.info("Using device: cpu")
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load data
    df_train, df_test_id, df_test_ood = load_dataframes(problem_name, dim)
    df_dict = create_df_dict(df_train, df_test_id, df_test_ood)

    # Get variable description if needed
    describe = ""
    if use_var_desc:
        try:
            describe = create_describe(problem_name)
        except ValueError:
            LOGGER.warning("Variable description not found, continuing without it.")

    # Get initial function strings
    init_func_str_list = eval(f"init_func_str_{dim}D")

    # Initialize evolution chain
    evo = evolution_chain(
        df_train=df_train,
        df_test_id=df_test_id,
        df_test_ood=df_test_ood,
        max_params=max_params,
        dim=dim,
        num_equations=num_equations,
        de_tolerance=de_tolerance,
        bfgs_tolerance=bfgs_tolerance,
        use_gt=use_gt,
        sampler_model_name=sampler_model_name,
        scientist_model_name=scientist_model_name,
        forget_probability=forget_prob,
    )

    # Create config
    config = RunnableConfig(
        recursion_limit=recursion_limit,
        configurable={
            "timeout": timeout,
            "max_retries": max_retries,
            "describe": describe,
            "use_var_desc": use_var_desc,
            "use_differential_evolution": use_differential_evolution,
        },
    )

    # Generate log directory name
    logs_dir_name = generate_logs_dir_name(
        use_var_desc,
        use_differential_evolution,
        start_time,
        use_scientist=use_scientist,
        use_gt=use_gt,
        forget_prob=forget_prob,
    )

    # Sanitize model name for directory (replace / with _)
    model_name_safe = sampler_model_name.replace("/", "_")

    # Create directories: logs/problem_name/model_name/logs_dir_name/
    problem_logs_dir = Path("logs") / problem_name / model_name_safe
    problem_logs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = problem_logs_dir / logs_dir_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    iteration_json_dir = logs_dir / "iteration_json"
    iteration_json_dir.mkdir(parents=True, exist_ok=True)
    configure_experiment_logging(logs_dir)

    # Print experiment configuration
    print_experiment_config(
        problem_name,
        dim,
        max_params,
        evolution_num,
        recursion_limit,
        timeout,
        max_retries,
        sampler_model_name,
        use_var_desc,
        use_differential_evolution,
        logs_dir,
        start_time,
        use_scientist=use_scientist,
        use_gt=use_gt,
        forget_prob=forget_prob,
        de_tolerance=de_tolerance,
        bfgs_tolerance=bfgs_tolerance,
        num_equations=num_equations,
    )
    resource_usage = collect_execution_environment()
    print_execution_environment(resource_usage)

    # Create initial state
    initial_state = create_initial_state(
        evo, df_dict, init_func_str_list, max_params, config
    )

    config_info = {
        "use_var_desc": use_var_desc,
        "use_differential_evolution": use_differential_evolution,
        "use_scientist": use_scientist,
        "use_gt": use_gt,
        "problem_name": problem_name,
        "max_params": max_params,
        "dim": dim,
        "evolution_num": evolution_num,
        "recursion_limit": recursion_limit,
        "timeout": timeout,
        "max_retries": max_retries,
        "sampler_model_name": sampler_model_name,
        "scientist_model_name": scientist_model_name,
        "num_equations": num_equations,
        "de_tolerance": de_tolerance,
        "bfgs_tolerance": bfgs_tolerance,
        "forget_prob": forget_prob,
        "run_metadata": {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "process_id": os.getpid(),
            "log_directory": str(logs_dir),
            "iteration_json_directory": str(iteration_json_dir),
            "run_log": str(logs_dir / "run.log"),
        },
        "resource_usage": resource_usage,
    }

    # Execute evolution
    result_list, _ = run_evolution(
        evo,
        initial_state,
        config,
        evolution_num,
        logs_dir,
        iteration_json_dir,
        problem_name,
        use_scientist,
        evo.get_func_names(),
        config_info=config_info,
    )

    LOGGER.info("Evolution completed! Total %s results saved.", len(result_list))
    LOGGER.info("Log directory: %s", logs_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ODE evolution experiment")

    # Required parameters
    parser.add_argument(
        "--use_var_desc",
        type=lambda x: str(x).lower() in ["true", "1", "yes", "y"],
        default=False,
        help="Whether to use variable description (true/false, default: false)",
    )
    parser.add_argument(
        "--use_differential_evolution",
        type=lambda x: str(x).lower() in ["true", "1", "yes", "y"],
        default=True,
        help="Whether to use Differential Evolution for optimization (true/false, default: true)",
    )
    parser.add_argument(
        "--problem_name", type=str, required=True, help="Problem name (e.g., ID_02)"
    )
    parser.add_argument(
        "--max_params", type=int, required=True, help="Maximum number of parameters"
    )
    parser.add_argument("--dim", type=int, required=True, help="Dimension (1, 2, 3, 4)")
    parser.add_argument(
        "--evolution_num",
        type=int,
        required=True,
        help="Number of evolution iterations",
    )

    # Optional parameters
    parser.add_argument(
        "--recursion_limit", type=int, default=15, help="Recursion limit (default: 15)"
    )
    parser.add_argument(
        "--timeout", type=int, default=180, help="Timeout in seconds (default: 180)"
    )
    parser.add_argument(
        "--max_retries", type=int, default=2, help="Maximum retry count (default: 2)"
    )
    parser.add_argument(
        "--sampler_model_name",
        type=str,
        default="google/gemini-2.5-flash-lite",
        help="Model name for Sampler LLM (default: google/gemini-2.5-flash-lite)",
    )
    parser.add_argument(
        "--scientist_model_name",
        type=str,
        default="google/gemini-2.5-flash-lite",
        help="Model name for Scientist LLM (default: google/gemini-2.5-flash-lite)",
    )

    parser.add_argument(
        "--use_scientist",
        type=lambda x: str(x).lower() in ["true", "1", "yes", "y"],
        default=False,
        help="Whether to run scientist for insight generation (true/false, default: false)",
    )
    parser.add_argument(
        "--num_equations",
        type=int,
        default=DEFAULT_NUM_EQUATIONS,
        help=f"Number of candidate equations to generate (default: {DEFAULT_NUM_EQUATIONS})",
    )
    parser.add_argument(
        "--de_tolerance",
        type=float,
        default=DE_TOLERANCE,
        help="Tolerance for Differential Evolution (overrides config)",
    )
    parser.add_argument(
        "--bfgs_tolerance",
        type=float,
        default=BFGS_TOLERANCE,
        help="Tolerance for BFGS Optimization (overrides config)",
    )
    parser.add_argument(
        "--use_gt",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Use ground truth (x*_t_gt) as target instead of gradient-based (x*_t)",
    )
    parser.add_argument(
        "--forget_prob",
        type=float,
        default=REMOVED_TERMS_FORGET_PROBABILITY,
        help=f"Probability to forget removed terms for re-exploration (default: {REMOVED_TERMS_FORGET_PROBABILITY})",
    )

    args = parser.parse_args()

    main(
        use_var_desc=args.use_var_desc,
        use_differential_evolution=args.use_differential_evolution,
        use_scientist=args.use_scientist,
        problem_name=args.problem_name,
        max_params=args.max_params,
        dim=args.dim,
        evolution_num=args.evolution_num,
        recursion_limit=args.recursion_limit,
        timeout=args.timeout,
        max_retries=args.max_retries,
        sampler_model_name=args.sampler_model_name,
        scientist_model_name=args.scientist_model_name,
        num_equations=args.num_equations,
        de_tolerance=args.de_tolerance,
        bfgs_tolerance=args.bfgs_tolerance,
        use_gt=args.use_gt,
        forget_prob=args.forget_prob,
    )
