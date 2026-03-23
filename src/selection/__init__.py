# src/selection/__init__.py
"""
Channel selection methods
"""
from .base_selector import BaseChannelSelector
from .hrank import HRankSelector
from .random_selector import RandomSelector
from .gradient_based import GradientBasedSelector
from .weight_gradient_based import WeightGradientBasedSelector
from .median_based import MedianBasedSelector
from .taylor_expansion import TaylorExpansionSelector
from .activation_variance import ActivationVarianceSelector
from .weight_norm import WeightNormSelector

__all__ = [
    'BaseChannelSelector',
    'HRankSelector', 
    'RandomSelector',
    'GradientBasedSelector',
    'WeightGradientBasedSelector',
    'MedianBasedSelector',
    'TaylorExpansionSelector',
    'ActivationVarianceSelector',
    'WeightNormSelector',
    'get_selector'
]


def get_selector(method_name, seed=42):
    """
    Factory function to get selector by name
    
    Args:
        method_name: Selection method name
        seed: Random seed (only used for random selector)
        
    Available methods:
        - 'hrank': Feature map rank based
        - 'random': Random selection (baseline)
        - 'gradient_based': Feature gradient based
        - 'weight_gradient_based': Weight gradient based
        - 'median_based': Geometric median
        - 'taylor_expansion': First-order Taylor approximation
        - 'activation_variance': Activation variance based
        - 'weight_norm': L2 norm of weights (static)
        
    Returns:
        Selector instance
    """
    selectors = {
        'hrank': HRankSelector,
        'random': lambda: RandomSelector(seed=seed),
        'gradient_based': GradientBasedSelector,
        'weight_gradient_based': WeightGradientBasedSelector,
        'median_based': MedianBasedSelector,
        'taylor_expansion': TaylorExpansionSelector,
        'activation_variance': ActivationVarianceSelector,
        'weight_norm': WeightNormSelector,
    }
    
    method_name = method_name.lower()
    if method_name not in selectors:
        raise ValueError(
            f"Unknown selector: {method_name}. "
            f"Available: {list(selectors.keys())}"
        )
    
    selector_factory = selectors[method_name]
    return selector_factory() if callable(selector_factory) else selector_factory