"""
Utility Functions Module

Contains common utility functions for configuration loading, logging setup, file management, etc.
"""

import os
import yaml
import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import datetime


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Config file format error: {e}")


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    if log_file:
        ensure_dir(os.path.dirname(log_file))
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def ensure_dir(dir_path: Union[str, Path]) -> None:
    """Ensure directory exists, create if not"""
    if dir_path:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def create_model_name(dataset: str, algorithm: str, k: int, m: int, 
                     n_clusters: int = None, dist_metric: str = "l2") -> str:
    """Create model name string"""
    if algorithm == "ivf-pq" and n_clusters:
        return f"{dataset}_{algorithm}_k{k}_m{m}_clusters{n_clusters}_{dist_metric}"
    else:
        return f"{dataset}_{algorithm}_k{k}_m{m}_{dist_metric}"


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save data to JSON file"""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_fvecs(filename: str) -> np.ndarray:
    """Read .fvecs format file"""
    with open(filename, 'rb') as f:
        # Read dimension
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        
        # Calculate vector count
        file_size = os.path.getsize(filename)
        num_vectors = file_size // ((dim + 1) * 4)
        
        # Read all data
        data = np.fromfile(f, dtype=np.int32, count=(dim + 1) * num_vectors)
        data = data.reshape(-1, dim + 1)
        
        # Remove dimension column, convert to float32
        vectors = data[:, 1:].astype(np.float32)
        
    return vectors


def read_ivecs(filename: str) -> np.ndarray:
    """Read .ivecs format file"""
    with open(filename, 'rb') as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        
        file_size = os.path.getsize(filename)
        num_vectors = file_size // ((dim + 1) * 4)
        
        data = np.fromfile(f, dtype=np.int32, count=(dim + 1) * num_vectors)
        data = data.reshape(-1, dim + 1)
        
        vectors = data[:, 1:]
        
    return vectors


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors using FAISS built-in method"""
    try:
        import faiss
        normalized_vectors = vectors.copy().astype(np.float32)
        faiss.normalize_L2(normalized_vectors)
        return normalized_vectors
    except ImportError:
        # Fallback: manual normalization if FAISS not available
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return vectors / norms


def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def format_time(seconds: float) -> str:
    """Format time duration"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {secs:.2f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {secs:.2f}s"


def get_timestamp() -> str:
    """Get current timestamp string"""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def validate_parameters(algorithm: str, k_values: List[int], m_values: List[int],
                       dist_metric: str, n_clusters: int = None) -> None:
    """Validate parameter validity"""
    # Validate algorithm
    valid_algorithms = ["pq", "ivf-pq"]
    if algorithm not in valid_algorithms:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Supported: {valid_algorithms}")
    
    # Validate distance metric
    valid_metrics = ["l2", "angular", "ip", "cosine"]
    if dist_metric not in valid_metrics:
        raise ValueError(f"Unsupported distance metric: {dist_metric}. Supported: {valid_metrics}")
    
    # Validate K values
    for k in k_values:
        if k <= 0 or k > 65536:
            raise ValueError(f"Invalid K value: {k}. K should be in range 1-65536")
    
    # Validate M values
    for m in m_values:
        if m <= 0 or m > 128:
            raise ValueError(f"Invalid M value: {m}. M should be in range 1-128")
    
    # Validate cluster count for IVF
    if algorithm == "ivf-pq":
        if n_clusters is None or n_clusters <= 0:
            raise ValueError("IVF-PQ algorithm requires valid cluster count")


def print_system_info() -> None:
    """Print system information"""
    logger = logging.getLogger(__name__)
    
    try:
        import faiss
        if hasattr(faiss, '__version__'):
            logger.info(f"FAISS version: {faiss.__version__}")
        else:
            logger.info("FAISS installed (version info unavailable)")
    except ImportError:
        logger.warning("FAISS not installed")
    
    logger.info(f"NumPy version: {np.__version__}")
    logger.info(f"Memory usage: {get_memory_usage():.2f} MB")
    
    # Check FAISS GPU support
    try:
        import faiss
        if hasattr(faiss, 'get_num_gpus'):
            num_gpus = faiss.get_num_gpus()
            logger.info(f"Available GPUs: {num_gpus}")
        else:
            logger.info("GPU check: Not supported or version too old")
    except:
        logger.info("GPU check failed")


class ProgressTracker:
    """Progress tracker for long-running operations"""
    
    def __init__(self, total: int, description: str = "Processing"):
        """Initialize progress tracker"""
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.datetime.now()
        self.logger = logging.getLogger(__name__)
    
    def update(self, step: int = 1) -> None:
        """Update progress"""
        self.current += step
        percentage = (self.current / self.total) * 100
        
        elapsed = (datetime.datetime.now() - self.start_time).total_seconds()
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
            eta_str = format_time(eta)
        else:
            eta_str = "Unknown"
        
        self.logger.info(f"{self.description}: {self.current}/{self.total} "
                        f"({percentage:.1f}%) - ETA: {eta_str}")
    
    def finish(self) -> None:
        """Finish progress tracking"""
        elapsed = (datetime.datetime.now() - self.start_time).total_seconds()
        self.logger.info(f"{self.description} completed, time: {format_time(elapsed)}") 