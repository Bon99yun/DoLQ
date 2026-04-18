"""
Comparison logic for ODE candidates.

This module centralizes the logic for determining which ODE candidate is 'better'.
Now uses a priority-based comparison on dimension-wise MSEs.
"""

from typing import Dict, List, Optional

def get_dim_scores_values(candidate: dict) -> List[float]:
    """Extract list of dimension scores sorted by key to ensure consistent order.
    
    Args:
        candidate: Candidate dictionary
        
    Returns:
        List of MSE values. Returns [inf] if invalid.
    """
    dim_scores = candidate.get('dim_scores', {}) if candidate else {}
    if not dim_scores:
        return [float('inf')]
        
    # Sort keys to ensure consistent order (e.g. x0_t, x1_t)
    # pair1[i] must correspond to same dimension as pair2[i]
    return [dim_scores[k] for k in sorted(dim_scores)]


def compare_priority(pair1: List[float], pair2: List[float], epsilon: float = 1e-12) -> List[float]:
    """
    Compares two lists by sorting dimensions based on their minimum values.
    
    Returns:
        The 'better' list (smaller values preferred).
    """
    # If element counts differ (e.g., different dimensions) -> fallback to sum comparison
    # Using fallback as requested instead of raising an error.
    if len(pair1) != len(pair2):
        return pair1 if sum(pair1) < sum(pair2) else pair2
    
    # Compare dimensions in priority order: the dimension with the best
    # candidate value across the pair is most important and breaks ties first.
    sorted_indices = sorted(range(len(pair1)), key=lambda i: min(pair1[i], pair2[i]))
    for idx in sorted_indices:
        val1 = pair1[idx]
        val2 = pair2[idx]
        
        # If difference is >= epsilon, this dimension decides the winner
        if abs(val1 - val2) >= epsilon:
            return pair1 if val1 < val2 else pair2
            
    # If all dimensions are similar within epsilon, prefer keeping pair1.
    return pair1


def is_better_than(candidate_new: Optional[dict], candidate_old: Optional[dict]) -> bool:
    """Return True if candidate_new is strictly better than candidate_old.
    
    Uses compare_priority logic.
    """
    if not candidate_new:
        return False
    
    if not candidate_old:
        # New candidate exists, old does not -> New is better
        return True
        
    scores_new = get_dim_scores_values(candidate_new)
    scores_old = get_dim_scores_values(candidate_old)
    
    better_scores = compare_priority(scores_new, scores_old)
    return better_scores is scores_new and scores_new != scores_old


def select_best_candidate(candidates: List[dict]) -> Optional[dict]:
    """Select the single best candidate from a list of candidates using priority logic."""
    if not candidates:
        return None
    
    best_cand = candidates[0]
    for cand in candidates[1:]:
        if is_better_than(cand, best_cand):
            best_cand = cand
            
    return best_cand
