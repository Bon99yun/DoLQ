"""
Structured output schemas for LLM ODE evolution.

This module defines Pydantic models for structured LLM output.
Uses a factory function to dynamically create dimension-specific schemas.
"""

from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, create_model


class TermSuggestion(BaseModel):
    """Class containing a single term and its reasoning."""
    
    term: str = Field(
        ...,
        description="Mathematical term expression (e.g., 'x0', 'np.sin(x0)', '9.81'). Do NOT add coefficients like 'C*' or 'params[N]*'. Do NOT use symbolic constants like 'g' or 'k' (use numbers, e.g., 9.81)."
    )
    
    reasoning: str = Field(
        ...,
        description="Reasoning for proposing this term. Explain the physical/mathematical basis based on the system description (desc). 1-2 sentences."
    )


def _generate_x_args_str(dim: int) -> str:
    """Generate x arguments string for function signature.
    
    Args:
        dim: Dimension (1, 2, 3, or 4)
        
    Returns:
        String like 'x0', 'x0, x1', 'x0, x1, x2', etc.
    """
    return ", ".join([f"x{i}" for i in range(dim)])


def create_function_output_class(
    dim: int, 
    num_equations: int = 1,
    max_params: int = 8
) -> type[BaseModel]:
    """Dynamically create FunctionOutput class for given dimension.
    
    Generates ODE system pairs where all equations are designed together.
    LLM returns term lists instead of full Python code.
    
    Args:
        dim: Dimension (1, 2, 3, or 4)
        num_equations: Number of ODE system pairs to generate
        max_params: Maximum number of parameters (last one is constant term)
        
    Returns:
        A Pydantic BaseModel class with ode_pairs field
    """
    x_args_str = _generate_x_args_str(dim)
    
    # Generate dimension-specific examples using only available variables
    def generate_example(func_idx: int) -> str:
        """Generate example term list for a specific function using only valid variables."""
        if dim == 1:
            return '["x0", "np.sin(x0)"]'
        elif dim == 2:
            return '["x0", "x1", "x0*x1"]'
        elif dim == 3:
            return '["x0", "x1", "x2"]'
        else:  # dim == 4
            return '["x0", "x1*x2", "x3"]'
    
    # Create ODEPair class dynamically for this dimension
    pair_fields: dict[str, Any] = {}
    for i in range(dim):
        field_name = f"x{i}_t"
        example = generate_example(i)
        pair_fields[field_name] = (
            List[TermSuggestion],
            Field(
                description=f"Term suggestions for {field_name} with reasoning. "
                           f"Each suggestion contains: term (expression without coefficients) + reasoning (justification). "
                           f"Do NOT use 'C' or 'params[N]'. The system will automatically assign coefficients. "
                           f"Variables: {x_args_str} ONLY. "
                           f"Example term: {example}"
            )
        )
    pair_fields["pair_reasoning"] = (
        str,
        Field(description="Brief reasoning for this ODE system design (1-2 sentences).")
    )
    
    ODEPair = create_model(
        f"ODEPair_{dim}D",
        __doc__=f"Single ODE system for {dim}D with all equations designed together.",
        **pair_fields
    )
    
    # Create main FunctionOutput class with ode_pairs
    output_fields: dict[str, Any] = {}
    
    output_fields["ode_pairs"] = (
        List[ODEPair],
        Field(
            description=f"List of {num_equations} complete ODE system pairs. "
                       f"Design all {dim} equations together as a coupled system. "
                       f"Each ODE system should explore different mathematical hypotheses."
        )
    )
    
    model_class = create_model(
        f"FunctionOutput_{dim}D_{num_equations}Pairs",
        __doc__=f"ODE function output schema for {dim}D with {num_equations} system pairs",
        **output_fields
    )
    
    return model_class

class TermEvaluation(BaseModel):
    """Semantic and performance evaluation for a single term."""
    

    
    semantic_quality: str = Field(
        ...,
        description="Semantic validity. Choose one of 'good', 'neutral', 'bad'. "
                    "Evaluate how physically/mathematically valid this term is based on the system description (desc). "
                    "(WARNING) Maximum 3 'good' ratings are allowed per function."
    )
    
    # performance_impact and action are determined by system logic, not LLM
    
    reasoning: str = Field(
        ...,
        description="Reasoning for this evaluation. 1-2 sentences. Focus on the system description (physical meaning)."
    )



class FunctionTermEvaluation(BaseModel):
    """Term evaluations for a single function (e.g., x0_t)."""
    
    function_name: str = Field(
        ...,
        description="Name of the function (e.g., 'x0_t')."
    )
    
    evaluations: List[TermEvaluation] = Field(
        ...,
        description="List of evaluations for all terms in this function."
    )


class ExperimentAnalysis(BaseModel):
    """Term-by-term evaluation and overall insights for experiment results."""
    
    term_evaluations_list: List[FunctionTermEvaluation] = Field(
        ...,
        description="List of term evaluations per function. "
                    "Each item corresponds to one function (e.g., x0_t, x1_t). "
                    "All terms in the current attempt must be evaluated."
    )
    
    updated_insight: str = Field(
        ...,
        description="Summary of key insights so far. 2-3 sentences. "
                    "Synthesize the term-by-term evaluation results into overall findings."
    )

