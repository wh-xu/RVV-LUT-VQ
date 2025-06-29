"""
PQ Searcher Module

Handles loading trained PQ models and executing vector search
"""

import os
import logging
import time
import numpy as np
import faiss
from typing import Dict, Any, List, Tuple, Optional

from .utils import create_model_name, load_json, format_time, ensure_dir


class PQSearcher:
    """PQ Searcher for loading models and executing vector search"""
    
    def __init__(self, models_dir: str = "./models"):
        """Initialize PQ searcher"""
        self.logger = logging.getLogger(__name__)
        self.models_dir = models_dir
        self.loaded_models = {}  # Model cache
        
        if not os.path.exists(models_dir):
            ensure_dir(models_dir)
            self.logger.info(f"Models directory created: {models_dir}")
        else:
            self.logger.debug(f"Models directory exists: {models_dir}")
        
    def load_model(self, model_name: str) -> bool:
        """Load model into memory"""
        if model_name in self.loaded_models:
            self.logger.info(f"Model already in memory: {model_name}")
            return True
            
        model_path = os.path.join(self.models_dir, model_name)
        index_path = os.path.join(model_path, "index.faiss")
        metadata_path = os.path.join(model_path, "metadata.json")
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            self.logger.error(f"Model files not found: {model_name}")
            return False
            
        try:
            self.logger.info(f"Loading model: {model_name}")
            
            # Load FAISS index
            index = faiss.read_index(index_path)
            
            # Load metadata
            metadata = load_json(metadata_path)
            
            # Cache model
            self.loaded_models[model_name] = {
                "index": index,
                "metadata": metadata
            }
            
            self.logger.info(f"Model loaded successfully: {model_name}")
            self.logger.info(f"  Vector count: {index.ntotal}")
            self.logger.info(f"  Dimension: {metadata.get('dimension', 'Unknown')}")
            self.logger.info(f"  Algorithm: {metadata.get('algorithm', 'Unknown')}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {str(e)}")
            return False
    
    def search_model(self, model_name: str, query_vectors: np.ndarray, topk: int = 100,
                    rerank_k: Optional[int] = None) -> Dict[str, Any]:
        """Execute search on a specific model"""
        # Ensure model is loaded
        if not self.load_model(model_name):
            raise RuntimeError(f"Cannot load model: {model_name}")
        
        model_info = self.loaded_models[model_name]
        index = model_info["index"]
        metadata = model_info["metadata"]
        
        self.logger.info(f"Executing search - Model: {model_name}")
        self.logger.info(f"Query vectors: {query_vectors.shape[0]}")
        self.logger.info(f"TopK: {topk}")
        if rerank_k:
            self.logger.info(f"Rerank candidates: {rerank_k}")
        else:
            self.logger.info("Rerank: No")
        
        # Set search parameters
        if hasattr(index, 'nprobe'):
            original_nprobe = index.nprobe
            index.nprobe = min(32, index.nlist)  # Auto-set reasonable nprobe
            self.logger.info(f"Set nprobe: {index.nprobe}")
        
        # Execute search
        search_start = time.time()
        distances, indices = index.search(query_vectors, topk)
        search_time = time.time() - search_start
        
        # Restore original nprobe
        if hasattr(index, 'nprobe'):
            index.nprobe = original_nprobe
        
        result = {
            "indices": indices,
            "distances": distances,
            "search_time": search_time,
            "qps": query_vectors.shape[0] / search_time,
            "model_name": model_name
        }
        
        self.logger.info(f"Search completed, time: {format_time(search_time)}")
        self.logger.info(f"QPS: {result['qps']:.2f}")
        
        return result
    
    def search(self, model_name: str, query_vectors: np.ndarray, k: int = 100,
              rerank: int = 0, base_vectors: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Execute search (legacy method for compatibility)"""
        if not self.load_model(model_name):
            raise RuntimeError(f"Cannot load model: {model_name}")
        
        model_info = self.loaded_models[model_name]
        index = model_info["index"]
        metadata = model_info["metadata"]
        
        timing_info = {}
        
        if hasattr(index, 'nprobe'):
            original_nprobe = index.nprobe
            index.nprobe = min(32, index.nlist)
        
        search_start = time.time()
        
        if rerank > 0 and base_vectors is not None:
            distances, indices = self._search_with_rerank(
                index, query_vectors, k, base_vectors, metadata, rerank
            )
        else:
            distances, indices = index.search(query_vectors, k)
        
        search_time = time.time() - search_start
        timing_info["search_time"] = search_time
        timing_info["queries_per_second"] = query_vectors.shape[0] / search_time
        
        if hasattr(index, 'nprobe'):
            index.nprobe = original_nprobe
        
        return distances, indices, timing_info
    
    def _search_with_rerank(self, index, query_vectors: np.ndarray,
                           k: int, base_vectors: np.ndarray, 
                           metadata: Dict[str, Any], candidate_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search with reranking"""
        self.logger.info("Executing rerank search...")
        
        # Phase 1: PQ search for candidates
        candidate_k = min(candidate_k, base_vectors.shape[0])
        
        self.logger.info(f"Phase 1 candidate count: {candidate_k}")
        candidate_distances, candidate_indices = index.search(query_vectors, candidate_k)
        
        # Phase 2: Exact distance reranking
        self.logger.info("Phase 2 reranking...")
        final_distances = np.zeros((query_vectors.shape[0], k), dtype=np.float32)
        final_indices = np.zeros((query_vectors.shape[0], k), dtype=np.int64)
        
        dist_metric = metadata.get('dist_metric', 'l2')
        
        for i, query in enumerate(query_vectors):
            candidates_idx = candidate_indices[i]
            valid_mask = candidates_idx >= 0
            valid_candidates = candidates_idx[valid_mask]
            
            if len(valid_candidates) == 0:
                continue
                
            candidate_vectors = base_vectors[valid_candidates]
            
            if dist_metric == "l2":
                dists = np.sum((query[np.newaxis, :] - candidate_vectors) ** 2, axis=1)
            else:
                dists = -np.dot(candidate_vectors, query)
            
            sorted_indices = np.argsort(dists)[:k]
            final_distances[i, :len(sorted_indices)] = dists[sorted_indices]
            final_indices[i, :len(sorted_indices)] = valid_candidates[sorted_indices]
        
        return final_distances, final_indices
    
    def batch_search(self, model_names: List[str], query_vectors: np.ndarray,
                    k: int = 100, rerank: int = 0,
                    base_vectors: Optional[np.ndarray] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str, float]]]:
        """Batch search multiple models"""
        results = {}
        
        for model_name in model_names:
            try:
                self.logger.info(f"Searching model: {model_name}")
                distances, indices, timing_info = self.search(
                    model_name, query_vectors, k, rerank, base_vectors
                )
                results[model_name] = (distances, indices, timing_info)
                
            except Exception as e:
                self.logger.error(f"Search failed for model {model_name}: {str(e)}")
                results[model_name] = None
        
        return results
    
    def get_model_list(self) -> List[str]:
        """Get available model list"""
        models = []
        
        if not os.path.exists(self.models_dir):
            return models
        
        for model_name in os.listdir(self.models_dir):
            model_path = os.path.join(self.models_dir, model_name)
            
            if not os.path.isdir(model_path):
                continue
                
            index_path = os.path.join(model_path, "index.faiss")
            metadata_path = os.path.join(model_path, "metadata.json")
            
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                models.append(model_name)
        
        return sorted(models)
    
    def filter_models(self, dataset: str = None, algorithm: str = None,
                     k_values: List[int] = None, m_values: List[int] = None,
                     dist_metric: str = None) -> List[str]:
        """Filter models by conditions"""
        all_models = self.get_model_list()
        filtered_models = []
        
        for model_name in all_models:
            try:
                model_path = os.path.join(self.models_dir, model_name)
                metadata_path = os.path.join(model_path, "metadata.json")
                metadata = load_json(metadata_path)
                
                if dataset and metadata.get('dataset') != dataset:
                    continue
                    
                if algorithm and metadata.get('algorithm') != algorithm:
                    continue
                    
                if k_values and metadata.get('k') not in k_values:
                    continue
                    
                if m_values and metadata.get('m') not in m_values:
                    continue
                    
                if dist_metric and metadata.get('dist_metric') != dist_metric:
                    continue
                
                filtered_models.append(model_name)
                
            except Exception as e:
                self.logger.warning(f"Failed to read model metadata {model_name}: {str(e)}")
        
        return filtered_models
    
    def get_loaded_models(self) -> List[str]:
        """Get loaded model list"""
        return list(self.loaded_models.keys())
    
    def unload_model(self, model_name: str) -> None:
        """Unload model from memory"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            self.logger.info(f"Model unloaded: {model_name}")
        else:
            self.logger.warning(f"Model not in memory: {model_name}")
    
    def clear_cache(self) -> None:
        """Clear model cache"""
        self.loaded_models.clear()
        self.logger.info("Model cache cleared")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage info"""
        usage = {
            "total_models": len(self.loaded_models),
            "total_memory_mb": 0.0,
            "models": {}
        }
        
        for model_name, model_info in self.loaded_models.items():
            try:
                index = model_info["index"]
                if hasattr(index, 'ntotal') and hasattr(index, 'd'):
                    estimated_mb = index.ntotal * index.d * 4 / 1024 / 1024  # Assume float32
                    usage["models"][model_name] = estimated_mb
                    usage["total_memory_mb"] += estimated_mb
                    
            except Exception as e:
                self.logger.warning(f"Failed to calculate memory usage for model {model_name}: {str(e)}")
        
        return usage 