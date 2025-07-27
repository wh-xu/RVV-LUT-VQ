"""
Data Manager Module

Handles dataset download, loading, validation and preprocessing
"""

import os
import logging
import requests
import tarfile
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from tqdm import tqdm

try:
    import h5py
except ImportError:
    h5py = None

from .utils import (
    load_config, ensure_dir, read_fvecs, read_ivecs, 
    normalize_vectors, ProgressTracker
)


class DataManager:
    """Dataset Manager for automatic download and loading"""
    
    def __init__(self, config_path: str = "configs/dataset_configs.yaml"):
        """Initialize data manager"""
        self.logger = logging.getLogger(__name__)
        self.config = load_config(config_path)
        self.datasets_config = self.config.get('datasets', {})
        
        # Ensure base dataset directory exists
        base_dataset_dir = "./Dataset"
        ensure_dir(base_dataset_dir)
        self.logger.info(f"Dataset base directory ensured: {base_dataset_dir}")
        
    def check_dataset_exists(self, dataset_name: str) -> bool:
        """Check if dataset exists locally"""
        if dataset_name not in self.datasets_config:
            self.logger.error(f"Unsupported dataset: {dataset_name}")
            return False
            
        dataset_config = self.datasets_config[dataset_name]
        local_path = dataset_config['local_path']
        
        is_hdf5 = dataset_config.get('params', {}).get('format') == 'hdf5'
        
        if is_hdf5:
            # HDF5 format: check single file and required keys
            required_files = dataset_config['files']
            hdf5_file = None
            required_keys = []
            
            for file_type, file_info in required_files.items():
                file_path = os.path.join(local_path, file_info['filename'])
                if hdf5_file is None:
                    hdf5_file = file_path
                required_keys.append(file_info['hdf5_key'])
            
            if not os.path.exists(hdf5_file):
                self.logger.info(f"Missing HDF5 file: {hdf5_file}")
                return False
            
            if h5py is None:
                self.logger.error("h5py required for HDF5 format")
                return False
                
            try:
                with h5py.File(hdf5_file, 'r') as f:
                    for key in set(required_keys):
                        if key not in f:
                            self.logger.info(f"Missing HDF5 key: {key}")
                            return False
            except Exception as e:
                self.logger.error(f"Error checking HDF5 file: {str(e)}")
                return False
        else:
            # fvecs format: check all required files
            required_files = dataset_config['files']
            for file_type, file_info in required_files.items():
                file_path = os.path.join(local_path, file_info['filename'])
                if not os.path.exists(file_path):
                    self.logger.info(f"Missing file: {file_path}")
                    return False
                
        self.logger.info(f"Dataset {dataset_name} exists")
        return True
    
    def download_dataset(self, dataset_name: str, force_download: bool = False) -> bool:
        """Download dataset if not exists"""
        if dataset_name not in self.datasets_config:
            self.logger.error(f"Unsupported dataset: {dataset_name}")
            return False
            
        if not force_download and self.check_dataset_exists(dataset_name):
            return True
            
        dataset_config = self.datasets_config[dataset_name]
        local_path = dataset_config['local_path']
        ensure_dir(local_path)
        
        self.logger.info(f"Starting download: {dataset_name}")
        
        try:
            is_hdf5 = dataset_config.get('params', {}).get('format') == 'hdf5'
            
            if is_hdf5:
                # HDF5: direct download
                first_file = list(dataset_config['files'].values())[0]
                download_url = first_file['url']
                filename = first_file['filename']
                
                target_path = os.path.join(local_path, filename)
                
                self.logger.info(f"Downloading HDF5 from {download_url}...")
                success = self._download_file(download_url, target_path)
                
                if not success:
                    self.logger.error("Download failed")
                    return False
                    
                self.logger.info(f"HDF5 file saved: {target_path}")
                
            else:
                # Traditional format: download and extract archive
                first_file = list(dataset_config['files'].values())[0]
                download_url = first_file['url']
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    archive_path = os.path.join(temp_dir, "dataset.tar.gz")
                    
                    self.logger.info(f"Downloading archive from {download_url}...")
                    success = self._download_file(download_url, archive_path)
                    
                    if not success:
                        self.logger.error("Download failed")
                        return False
                    
                    self.logger.info("Extracting files...")
                    with tarfile.open(archive_path, 'r:gz') as tar:
                        tar.extractall(temp_dir)
                    
                    self._organize_files(temp_dir, local_path, dataset_config['files'])
                
            self.logger.info(f"Dataset {dataset_name} downloaded")
            return True
            
        except Exception as e:
            self.logger.error(f"Download error: {str(e)}")
            return False
    
    def _download_file(self, url: str, filepath: str) -> bool:
        """Download single file with progress bar"""
        try:
            if url.startswith('ftp://'):
                import urllib.request
                
                self.logger.info(f"Using FTP: {url}")
                
                def progress_hook(block_num, block_size, total_size):
                    if hasattr(progress_hook, 'pbar'):
                        progress_hook.pbar.update(block_size)
                    else:
                        progress_hook.pbar = tqdm(
                            desc="Download",
                            total=total_size,
                            unit='B',
                            unit_scale=True,
                            unit_divisor=1024,
                        )
                        progress_hook.pbar.update(block_size)
                
                urllib.request.urlretrieve(url, filepath, progress_hook)
                
                if hasattr(progress_hook, 'pbar'):
                    progress_hook.pbar.close()
                
                return True
            else:
                # HTTP download
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(filepath, 'wb') as f, tqdm(
                    desc="Download",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                            
                return True
            
        except Exception as e:
            self.logger.error(f"Download failed: {str(e)}")
            return False
    
    def _organize_files(self, temp_dir: str, target_dir: str, files_config: Dict[str, Any]) -> None:
        """Organize extracted files"""
        for file_type, file_info in files_config.items():
            target_filename = file_info['filename']
            
            if 'archive_path' in file_info:
                src_path = os.path.join(temp_dir, file_info['archive_path'])
            else:
                # Search for file in extracted directory
                src_path = None
                for root, dirs, files in os.walk(temp_dir):
                    if target_filename in files:
                        src_path = os.path.join(root, target_filename)
                        break
            
            if src_path and os.path.exists(src_path):
                dst_path = os.path.join(target_dir, target_filename)
                os.rename(src_path, dst_path)
                self.logger.info(f"File saved: {dst_path}")
            else:
                self.logger.warning(f"File not found: {target_filename}")
    
    def load_dataset(self, dataset_name: str, dist_metric: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load dataset with automatic download"""
        if not self.check_dataset_exists(dataset_name):
            self.logger.info("Dataset not exists, downloading...")
            if not self.download_dataset(dataset_name):
                raise RuntimeError(f"Failed to get dataset: {dataset_name}")
        
        dataset_config = self.datasets_config[dataset_name]
        
        if dist_metric is None:
            dist_metric = dataset_config.get('params', {}).get('distance_metric', 'l2')
        
        self.logger.info(f"Loading dataset: {dataset_name} (distance: {dist_metric})")
        
        is_hdf5 = dataset_config.get('params', {}).get('format') == 'hdf5'
        
        if is_hdf5:
            base_vectors, query_vectors, learn_vectors, groundtruth = self._load_hdf5_dataset(dataset_config)
        else:
            base_vectors, query_vectors, learn_vectors, groundtruth = self._load_fvecs_dataset(dataset_config)
        
        # Vector normalization for inner product distance
        if dist_metric in ["angular", "ip", "cosine"]:
            self.logger.info(f"Normalizing vectors for {dist_metric} distance...")
            base_vectors = normalize_vectors(base_vectors)
            query_vectors = normalize_vectors(query_vectors)
            learn_vectors = normalize_vectors(learn_vectors)
        
        self._validate_data(base_vectors, query_vectors, learn_vectors, groundtruth, dataset_config)
        
        self.logger.info(f"Dataset loaded:")
        self.logger.info(f"  Base vectors: {base_vectors.shape}")
        self.logger.info(f"  Query vectors: {query_vectors.shape}")
        self.logger.info(f"  Learn vectors: {learn_vectors.shape}")
        self.logger.info(f"  Groundtruth: {groundtruth.shape}")
        
        return base_vectors, query_vectors, learn_vectors, groundtruth
    
    def _load_fvecs_dataset(self, dataset_config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load fvecs format dataset"""
        local_path = dataset_config['local_path']
        files_config = dataset_config['files']
        
        base_path = os.path.join(local_path, files_config['base_vectors']['filename'])
        query_path = os.path.join(local_path, files_config['query_vectors']['filename'])
        learn_path = os.path.join(local_path, files_config['learn_vectors']['filename'])
        gt_path = os.path.join(local_path, files_config['groundtruth']['filename'])
        
        self.logger.info("Loading base vectors...")
        base_vectors = read_fvecs(base_path)
        
        self.logger.info("Loading query vectors...")
        query_vectors = read_fvecs(query_path)
        
        self.logger.info("Loading learn vectors...")
        learn_vectors = read_fvecs(learn_path)
        
        self.logger.info("Loading groundtruth...")
        groundtruth = read_ivecs(gt_path)
        
        return base_vectors, query_vectors, learn_vectors, groundtruth
    
    def _load_hdf5_dataset(self, dataset_config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load HDF5 format dataset"""
        if h5py is None:
            raise ImportError("h5py required for HDF5 format")
        
        local_path = dataset_config['local_path']
        files_config = dataset_config['files']
        
        hdf5_file = os.path.join(local_path, files_config['base_vectors']['filename'])
        self.logger.info(f"Loading from HDF5: {hdf5_file}")
        
        with h5py.File(hdf5_file, 'r') as f:
            # Load base vectors
            base_key = files_config['base_vectors']['hdf5_key']
            self.logger.info(f"Loading base vectors (key: {base_key})...")
            base_vectors = np.array(f[base_key], dtype=np.float32)
            
            # Load query vectors
            query_key = files_config['query_vectors']['hdf5_key']
            self.logger.info(f"Loading query vectors (key: {query_key})...")
            query_vectors = np.array(f[query_key], dtype=np.float32)
            
            # Load learn vectors
            learn_key = files_config['learn_vectors']['hdf5_key']
            if learn_key == base_key:
                self.logger.info(f"Using all train data as learn vectors (count: {len(base_vectors):,})...")
                learn_vectors = base_vectors.copy()
            else:
                self.logger.info(f"Loading learn vectors (key: {learn_key})...")
                learn_vectors = np.array(f[learn_key], dtype=np.float32)
            
            # Load groundtruth
            gt_key = files_config['groundtruth']['hdf5_key']
            self.logger.info(f"Loading groundtruth (key: {gt_key})...")
            groundtruth = np.array(f[gt_key], dtype=np.int32)
        
        return base_vectors, query_vectors, learn_vectors, groundtruth
    
    def _validate_data(self, base_vectors: np.ndarray, query_vectors: np.ndarray,
                      learn_vectors: np.ndarray, groundtruth: np.ndarray,
                      dataset_config: Dict[str, Any]) -> None:
        """Validate data integrity"""
        expected_dim = dataset_config['dimension']
        expected_base = dataset_config['num_base']
        expected_query = dataset_config['num_query']
        expected_learn = dataset_config['num_learn']
        
        # Validate dimensions
        if base_vectors.shape[1] != expected_dim:
            raise ValueError(f"Base vector dimension error: expected {expected_dim}, got {base_vectors.shape[1]}")
        
        if query_vectors.shape[1] != expected_dim:
            raise ValueError(f"Query vector dimension error: expected {expected_dim}, got {query_vectors.shape[1]}")
            
        if learn_vectors.shape[1] != expected_dim:
            raise ValueError(f"Learn vector dimension error: expected {expected_dim}, got {learn_vectors.shape[1]}")
        
        # Validate counts (allow some tolerance)
        if abs(base_vectors.shape[0] - expected_base) > expected_base * 0.01:
            self.logger.warning(f"Base vector count mismatch: expected {expected_base}, got {base_vectors.shape[0]}")
        
        if abs(query_vectors.shape[0] - expected_query) > expected_query * 0.01:
            self.logger.warning(f"Query vector count mismatch: expected {expected_query}, got {query_vectors.shape[0]}")
            
        if abs(learn_vectors.shape[0] - expected_learn) > expected_learn * 0.01:
            self.logger.warning(f"Learn vector count mismatch: expected {expected_learn}, got {learn_vectors.shape[0]}")
        
        # Validate groundtruth
        if groundtruth.shape[0] != query_vectors.shape[0]:
            raise ValueError(f"Groundtruth count mismatch: {groundtruth.shape[0]} vs {query_vectors.shape[0]}")
        
        self.logger.info("Data validation passed")
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get dataset configuration info"""
        if dataset_name not in self.datasets_config:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
            
        return self.datasets_config[dataset_name]
    
    def list_available_datasets(self) -> List[str]:
        """List available dataset names"""
        return list(self.datasets_config.keys())
    
    def get_dataset_stats(self, dataset_name: str) -> Dict[str, Any]:
        """Get dataset statistics"""
        try:
            if not self.check_dataset_exists(dataset_name):
                return {"error": "Dataset not found"}
            
            dataset_config = self.datasets_config[dataset_name]
            local_path = dataset_config['local_path']
            
            is_hdf5 = dataset_config.get('params', {}).get('format') == 'hdf5'
            
            if is_hdf5:
                return self._get_hdf5_stats(dataset_config)
            else:
                return self._get_fvecs_stats(dataset_config)
                
        except Exception as e:
            return {"error": str(e)}
    
    def _get_hdf5_stats(self, dataset_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get HDF5 dataset statistics"""
        if h5py is None:
            return {"error": "h5py not available"}
        
        local_path = dataset_config['local_path']
        files_config = dataset_config['files']
        hdf5_file = os.path.join(local_path, files_config['base_vectors']['filename'])
        
        with h5py.File(hdf5_file, 'r') as f:
            base_key = files_config['base_vectors']['hdf5_key']
            query_key = files_config['query_vectors']['hdf5_key']
            learn_key = files_config['learn_vectors']['hdf5_key']
            
            base_shape = f[base_key].shape
            query_shape = f[query_key].shape
            learn_shape = f[learn_key].shape if learn_key != base_key else base_shape
            
            return {
                "dimension": base_shape[1],
                "num_base_vectors": base_shape[0],
                "num_query_vectors": query_shape[0],
                "num_learn_vectors": learn_shape[0],
                "data_type": str(f[base_key].dtype),
                "total_memory_mb": (base_shape[0] * base_shape[1] * 4 + 
                                  query_shape[0] * query_shape[1] * 4 +
                                  learn_shape[0] * learn_shape[1] * 4) / 1024 / 1024
            }
    
    def _get_fvecs_stats(self, dataset_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get fvecs dataset statistics"""
        local_path = dataset_config['local_path']
        files_config = dataset_config['files']
        
        base_path = os.path.join(local_path, files_config['base_vectors']['filename'])
        
        # Read first vector to get dimension
        with open(base_path, 'rb') as f:
            dim = np.frombuffer(f.read(4), dtype=np.int32)[0]
            
        # Get file sizes to estimate vector counts
        def get_vector_count(filepath):
            file_size = os.path.getsize(filepath)
            return file_size // ((dim + 1) * 4)
        
        base_count = get_vector_count(base_path)
        query_count = get_vector_count(os.path.join(local_path, files_config['query_vectors']['filename']))
        learn_count = get_vector_count(os.path.join(local_path, files_config['learn_vectors']['filename']))
        
        return {
            "dimension": dim,
            "num_base_vectors": base_count,
            "num_query_vectors": query_count,
            "num_learn_vectors": learn_count,
            "data_type": "float32",
            "total_memory_mb": (base_count * dim * 4 + 
                              query_count * dim * 4 +
                              learn_count * dim * 4) / 1024 / 1024
        }
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary files"""
        temp_dir = tempfile.gettempdir()
        for filename in os.listdir(temp_dir):
            if filename.startswith("dataset_"):
                file_path = os.path.join(temp_dir, filename)
                try:
                    os.remove(file_path)
                    self.logger.info(f"Cleaned up: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to clean up {file_path}: {str(e)}") 