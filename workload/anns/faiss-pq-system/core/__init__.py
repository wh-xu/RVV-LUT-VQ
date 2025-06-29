"""
FAISS Product Quantization Core Module

This module contains core functionality for the PQ system:
- Data management (data_manager.py)
- PQ builder (pq_builder.py)
- PQ searcher (pq_searcher.py)
- Evaluator (evaluator.py)
- Utility functions (utils.py)
"""

from .data_manager import DataManager
from .pq_builder import PQBuilder
from .pq_searcher import PQSearcher
from .evaluator import Evaluator
from .utils import (
    load_config, setup_logging, create_model_name, ensure_dir,
    validate_parameters, print_system_info, get_timestamp, format_time
)

__version__ = "1.0.0"
__author__ = "FAISS PQ System"

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