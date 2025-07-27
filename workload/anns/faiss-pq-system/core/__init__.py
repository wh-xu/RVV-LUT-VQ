"""
FAISS Product Quantization Core Module

Comprehensive framework for Product Quantization (PQ) and Residual Quantization (RQ) 
based approximate nearest neighbor search using FAISS library.

References:
    - Jegou, H., et al. (2011). Product quantization for nearest neighbor search. IEEE TPAMI.
    - Babenko, A., & Lempitsky, V. (2014). Additive quantization for extreme vector compression. CVPR.

Authors: FAISS PQ System Development Team
"""

# Core algorithm components
from .data_manager import DataManager
from .pq_builder import PQBuilder
from .pq_searcher import PQSearcher
from .evaluator import Evaluator

# Utility functions
from .utils import (
    load_config, setup_logging, create_model_name, ensure_dir,
    validate_parameters, print_system_info, get_timestamp, format_time,
    get_dataset_distance_metric
)

__version__ = "1.0.0"
__author__ = "FAISS PQ System Development Team"

__all__ = [
    "DataManager",
    "PQBuilder", 
    "PQSearcher",
    "Evaluator",
    "load_config",
    "setup_logging",
    "create_model_name",
    "ensure_dir",
] 