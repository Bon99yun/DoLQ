sampler_ODE_prompt = """
You are a helpful assistant tasked with discovering mathematical term structures for scientific systems. 
Complete the 'term_list' below, considering physical meaning and relationships of inputs.

# System Description
{describe}

{scientist_insight}
{goal_section}
{previous_attempt}

[Required Conditions (Violation Will Cause Errors)]
1. You can use: import numpy as np
2. Target System Context: Input variables are {x_cols_str}.
   - This system is {dim}-dimensional
   - Variables x{dim} and above do not exist.
3. Term Format: Propose terms WITHOUT coefficients. The system will automatically attach trainable parameters.
   - Correct: "x0", "np.sin(x0)", "x0*x1"
   - Incorrect: "params[0]*x0", "C*x0", "0.5*x0"
4. Term Complexity: You MAY use internal constants if they have physical meaning (e.g., frequency, phase).
   - Example: "np.sin(2*x0)" is allowed and encouraged if the factor 2 is significant.
   - Note: The system will still attach an outer trainable parameter (e.g., params[0]*np.sin(2*x0)).
5. Symbolic Constants: Do NOT use symbolic constants like 'g', 'k', 'm'. Use numerical values.
   - Correct: "9.81*x0" (if g=9.81 is known), "np.pi*x0"
   - Incorrect: "g*x0" (will cause NameError)
6. No duplicates: Equations identical to previous attempts are forbidden. Structural modifications are required.
7. Reasoning required: When proposing each term, provide a physical/mathematical reasoning based on the system description (desc).

{example_section}
"""




def format_scientist_insight_for_prompt(
    insight_list: list = None,
    use_scientist: bool = False,
    removed_terms_per_dim: dict = None,
    term_evaluations: dict = None
) -> str:
    """Format insight list for inclusion in prompt.
    
    Args:
        insight_list: List of insight strings (can be None or empty)
        use_scientist: Whether scientist feature is enabled
        removed_terms_per_dim: Dictionary of removed terms per dimension
        term_evaluations: Dictionary of term evaluations from previous iteration
    
    Returns:
        Formatted insight string for prompt
    """
    if not use_scientist:
        return ""
    
    if not insight_list and not removed_terms_per_dim and not term_evaluations:
        return "# Previous Scientist Insights:\nNo insights generated yet."
    
    insight_content = "\n".join(f"- {i}" for i in insight_list) if insight_list else "None"
    
    # Format term evaluations if available
    eval_list_str = "None"
    if term_evaluations:
        lines = []
        for func_name, evals in term_evaluations.items():
            func_lines = []
            for e in evals:
                if isinstance(e, dict):
                    term = e.get('term', '')
                    action = e.get('action', 'unknown').lower()
                    
                    # Skip 'remove' items as they appear in the ban list
                    if action == 'remove':
                        continue
                        
                    func_lines.append(f"  - {term} : {action.upper()}")
            
            if func_lines:
                lines.append(f"{func_name}:")
                lines.extend(func_lines)
                
        if lines:
            eval_list_str = "\n".join(lines)
    
    # Format removed terms if available
    remove_list_str = "None"
    if removed_terms_per_dim:
        lines = []
        for func_name, skeletons in removed_terms_per_dim.items():
            if skeletons:
                # remove duplicates and sort for deterministic output
                unique_skeletons = sorted(list(set(skeletons)))
                if unique_skeletons:
                    lines.append(f"  {func_name}: {', '.join(unique_skeletons)}")
        if lines:
            remove_list_str = "\n".join(lines)
    
    return f"""
### SCIENTIST AGENT GUIDANCE
The Scientist agent has analyzed previous experiments and provides the following guidance:

#### Accumulated Knowledge (Theory)
{insight_content}

#### Term-by-Term Evaluation (Previous Attempt Analysis)
Evaluation results for each term. keep = retain, hold = hold/modify, remove = eliminate:
{eval_list_str}

#### Removed Terms List (Ban List)
The following term structures have negatively impacted performance. **Do NOT propose them again**:
{remove_list_str}
"""


def make_sampler_ODE_prompt(
    x_cols, func_names, max_params, 
    insight_list=None, use_scientist=False,
    previous_attempt_str="", describe="", dim: int = 1,
    removed_terms_per_dim: dict = None,
    term_evaluations: dict = None
):
    """Generate ODE sampler prompt.
    
    Args:
        x_cols: List of x variable columns
        func_names: function names string (e.g., "x0_t, x1_t, x2_t")
        max_params: Maximum number of parameters
        insight_list: List of insight strings (optional)
        use_scientist: Whether to include scientist section
        previous_attempt_str: Formatted string of the previous attempt (optional)
        describe: System description string
        dim: Dimension of the system (1, 2, 3, or 4)
        removed_terms_per_dim: Dictionary of removed terms per dimension
        term_evaluations: Dictionary of term evaluations from previous iteration
    
    Returns:
        Generated prompt string
    """
    x_cols_str = ", ".join(x_cols)
    
    # Format scientist section
    scientist_str = format_scientist_insight_for_prompt(
        insight_list, use_scientist, removed_terms_per_dim, term_evaluations
    )
    
    # Prepare previous attempt section
    previous_attempt_section = ""
    if previous_attempt_str:
         previous_attempt_section = previous_attempt_str
    
    # Generate Goals section dynamically based on use_scientist
    goal_section = ""
    if use_scientist:
        goal_section = "**Goal:** Reflect the Scientist's insights and guidance in the equation structure."
    
    # Generate dimension-specific example section
    def generate_example_section(d: int) -> str:
        if d == 1:
            return """[Example (1D System)]
x0_t: ["x0", "np.sin(x0)"]"""
        elif d == 2:
            return """[Example (2D System)]
x0_t: ["x0", "x1"]
x1_t: ["x0", "np.sin(x1)"]"""
        elif d == 3:
            return """[Example (3D System)]
x0_t: ["x0", "x1"]
x1_t: ["x1", "x2"]
x2_t: ["x0", "x2"]"""
        else:  # d == 4
            return """[Example (4D System)]
x0_t: ["x0", "x1*x2", "x3"]"""
    
    example_section = generate_example_section(dim)
    
    # Generate final prompt
    return sampler_ODE_prompt.format(
        describe=describe,
        scientist_insight=scientist_str,
        goal_section=goal_section,
        x_cols=x_cols,
        x_cols_str=x_cols_str,
        max_params=max_params,
        dim=dim,
        example_section=example_section,
        previous_attempt=previous_attempt_section
    )




# Scientist insight generation prompt templates
hypothesis_system_comparison = """
=== Previous Experiment Results ===
[Global Best (Target)] ({global_best_mse_str})
{global_best_code_block}

[Previous Attempt] ({prev_gen_mse_str})
{prev_gen_code_block}

=== New Experiment Results ===
[Current Attempt] ({current_mse_str})
{current_code_block}
"""

def make_analysis_and_record_prompt(
    observations: list,
    outcome_type: str,
    notebook: dict,
    describe: str = "",
    removed_terms_per_dim: dict = None,
    current_iteration: int = None,
    total_iterations: int = None
) -> str:
    """Create prompt for analyzing observation and recording in notebook."""
    
    # 1. Format Observation (3-way comparison)
    observation_texts = []
    for obs in observations:
        # Helper to format code and params as readable equations
        def format_code_with_params(codes, params):
            from utils import code_to_equation
            
            lines = []
            for dim, code in codes.items():
                p_list = params.get(dim, [])
                if hasattr(p_list, 'tolist'):
                    p_list = p_list.tolist()
                
                if code and p_list:
                    # Convert to readable equation format
                    eq = code_to_equation(code, dim, p_list)
                    lines.append(eq)
                else:
                    lines.append(f"{dim} = (no data)")
            
            return "\n".join(lines)
        
        # Helper to format current attempt with integrated reasonings
        def format_current_with_reasonings(codes, params, term_reasonings):
            from utils import extract_terms_from_code_with_params
            
            lines = []
            for dim, code in codes.items():
                p_list = params.get(dim, [])
                if hasattr(p_list, 'tolist'):
                    p_list = p_list.tolist()
                
                if not code or not p_list:
                    lines.append(f"{dim} = (no data)")
                    continue
                
                # Use shared utility to get terms
                readable_terms_all = extract_terms_from_code_with_params(code, p_list)
                
                # Filter out the constant offset term (always ends with * 1) for evaluation
                readable_terms = [t for t in readable_terms_all if not t.strip().endswith('* 1')]
                
                if not readable_terms:
                    # If only constant term existed, show it or special msg?
                    # But usually we have other terms. If empty, it means only constant term.
                    # Let's show (constant only, ignored for evaluation)
                     if readable_terms_all:
                         lines.append(f"{dim} = (constant offset only, ignored)")
                         continue
                     else:
                        lines.append(f"{dim} = (parsing error)")
                        continue
                
                # Reconstruct full equation for display
                full_eq_rhs = " + ".join(readable_terms)
                full_eq_rhs = full_eq_rhs.replace("+ -", "- ")
                full_eq = f"{dim} = {full_eq_rhs}"
                lines.append(full_eq)
                
                # Add reasonings with specific term matching
                if dim in term_reasonings and term_reasonings[dim]:
                    lines.append("  [Sampler's Reasoning]")
                    for idx, reason in enumerate(term_reasonings[dim]):
                        if reason:
                            # Show actual term expression instead of 'TermN'
                            term_display = f"Term{idx+1}"
                            if idx < len(readable_terms):
                                term_display = readable_terms[idx]
                            
                            lines.append(f"  - {term_display}: {reason}")
                    lines.append("")
            
            return "\n".join(lines)

        # Format code blocks
        global_best_text = format_code_with_params(obs.get('global_best_codes', {}), obs.get('global_best_params', {})) or "None (first iteration)"
        prev_gen_text = format_code_with_params(obs.get('prev_gen_codes', {}), obs.get('prev_gen_params', {})) or "None (first iteration)"
        
        # Current attempt with integrated reasonings
        term_reasonings = obs.get('term_reasonings', {})
        current_text = format_current_with_reasonings(
            obs.get('current_codes', {}), 
            obs.get('current_params', {}),
            term_reasonings
        ) or "None"
        
        # Format MSEs
        def format_scores(scores):
            if not scores: return "N/A"
            # score is text like x0_t: 1.23e-04
            return ", ".join([f"{k}: {v:.6e}" for k, v in scores.items()])

        gb_mse_str = format_scores(obs.get('global_best_dim_scores', {}))
        pg_mse_str = format_scores(obs.get('prev_gen_dim_scores', {}))
        cu_mse_str = format_scores(obs.get('current_dim_scores', {}))
        
        observation_texts.append(hypothesis_system_comparison.format(
            global_best_mse_str=gb_mse_str,
            global_best_code_block=global_best_text,
            
            prev_gen_mse_str=pg_mse_str,
            prev_gen_code_block=prev_gen_text,

            current_mse_str=cu_mse_str,
            current_code_block=current_text
        ))
    
    observation_texts_str = "\n".join(observation_texts)
    
    # Format term impacts from ablation study
    def format_term_impacts(term_impacts: dict) -> str:
        """Format term impacts into readable text"""
        if not term_impacts:
            return "No measurement data"
        
        lines = []
        for func_name, impacts in term_impacts.items():
            lines.append(f"\n{func_name}:")
            for imp in impacts:
                idx = imp['term_idx']
                impact = imp['impact']
                change_rate = imp['change_rate'] * 100
                
                # Translate impact labels
                impact_label = {
                    'positive': 'Positive',
                    'neutral': 'Neutral',
                    'negative': 'Negative'
                }.get(impact, impact)
                
                # Get term string if available
                term_display = imp.get('term_str', f"Term{idx+1}")
                
                lines.append(f"  - {term_display}: {impact_label} (MSE change when removed: {change_rate:+.1f}%)")
        return "\n".join(lines)
    
    term_impacts_text = ""
    if observations and observations[0].get('term_impacts'):
        term_impacts_text = format_term_impacts(observations[0]['term_impacts'])
    else:
        term_impacts_text = "No measurement data"
    
    # 2. Format Context
    accumulated_insight = notebook.get('accumulated_insight', 'None (first iteration)')
    
    system_desc_str = describe if describe else "Not provided (pure data-driven discovery)"
    
    # 3. Format Remove List (per dimension)
    remove_list_lines = []
    if removed_terms_per_dim:
        for func_name, skeletons in removed_terms_per_dim.items():
            if skeletons:
                remove_list_lines.append(f"  {func_name}: {', '.join(skeletons)}")
    remove_list_str = "\n".join(remove_list_lines) if remove_list_lines else "None"
    
    # 4. Format iteration progress
    iteration_info = ""
    if current_iteration is not None and total_iterations is not None:
        iteration_info = f"\n**Progress**: Currently on iteration {current_iteration} of {total_iterations} total\n"

    # 5. Combine into Single Prompt
    return f"""You are a senior scientist specializing in ODE (differential equation) discovery.
Your role is to evaluate proposed mathematical terms and provide guidance to improve the term_list in the next iteration. Leverage your scientific knowledge to assess physical plausibility and suggest improvements.
Analyze the experiment results and perform **term-by-term evaluation**.
{iteration_info}

System Description: {system_desc_str}

=== Accumulated Insights ===
{accumulated_insight}

=== Removed Terms List (marked 'remove' in previous iterations) ===
*Do not propose terms with the same skeleton structure as those below.*
{remove_list_str}

{observation_texts_str}

[How to Perform Term-by-Term Evaluation]

For all terms in the current attempt, evaluate only the following 2 aspects:

1. Semantic Quality (semantic_quality)
Evaluate based on the system description (desc):
- good: Clearly aligns with the physical/mathematical meaning of the system (**Note: Maximum 3 per function**)
- neutral: Has some relevance but not essential
- bad: Unrelated to or contradicts the system

2. Reasoning (reasoning)
Explain your evaluation in 1-2 sentences. Focus on the system description (physical meaning).

[Important Notes]
- Use coefficient analysis: Terms with coefficients near 0 are removal candidates. If all coefficients are far from 0, optimization may be insufficient.
"""
