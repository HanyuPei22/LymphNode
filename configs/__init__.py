# configs/__init__.py
"""
Configuration module
Separates path management from experiment configuration
"""

# Path configurations
from .paths import (
    PATHS,
    PROJECT_ROOT,
    BASE_PATH,
    get_model_path,
    get_pattern_path,
    get_watermark_data_path,
)

# Experiment configurations
from .experiment_configs import (
    ExperimentConfig,
    create_baseline_config,
    create_sparsity_config,
    create_data_efficiency_config,
)

__all__ = [
    # Paths
    'PATHS',
    'PROJECT_ROOT',
    'BASE_PATH',
    'get_model_path',
    'get_pattern_path',
    'get_watermark_data_path',
    
    # Experiment configs
    'ExperimentConfig',
    'create_baseline_config',
    'create_sparsity_config',
    'create_data_efficiency_config',
]