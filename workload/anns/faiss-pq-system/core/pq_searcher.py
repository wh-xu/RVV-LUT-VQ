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
        
    def _find_model_path(self, model_name: str) -> Optional[str]:
        """查找模型路径 - 支持按dataset分组的目录结构"""
        # 尝试多种路径查找模型
        possible_paths = []
        
        # 1. 直接路径（兼容旧结构）
        possible_paths.append(os.path.join(self.models_dir, model_name))
        
        # 2. 按dataset分组的路径（新结构）
        # 从model_name中提取dataset信息
        if "_" in model_name:
            parts = model_name.split("_")
            if len(parts) >= 2:
                # 假设第一部分是dataset名
                dataset_name = parts[0]
                simplified_name = "_".join(parts[1:])
                possible_paths.append(os.path.join(self.models_dir, dataset_name, simplified_name))
        
        # 3. 搜索所有dataset文件夹
        if os.path.exists(self.models_dir):
            for item in os.listdir(self.models_dir):
                item_path = os.path.join(self.models_dir, item)
                if os.path.isdir(item_path):
                    # 尝试在这个dataset文件夹中查找
                    possible_model_path = os.path.join(item_path, model_name)
                    possible_paths.append(possible_model_path)
                    
                    # 也尝试简化名称
                    if "_" in model_name:
                        simplified = "_".join(model_name.split("_")[1:])
                        possible_paths.append(os.path.join(item_path, simplified))
        
        # 查找存在的路径
        for model_path in possible_paths:
            index_path = os.path.join(model_path, "index.faiss")
            metadata_path = os.path.join(model_path, "metadata.json")
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                return model_path
        
        return None
        
    def load_model(self, model_name: str) -> bool:
        """Load model into memory - 支持按dataset分组的目录结构"""
        if model_name in self.loaded_models:
            self.logger.info(f"Model already in memory: {model_name}")
            return True
            
        # 尝试多种路径查找模型
        model_path = self._find_model_path(model_name)
        if not model_path:
            self.logger.error(f"Model files not found: {model_name}")
            return False
            
        index_path = os.path.join(model_path, "index.faiss")
        metadata_path = os.path.join(model_path, "metadata.json")
            
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
        
        # Calculate QPS with zero protection
        qps = query_vectors.shape[0] / search_time if search_time > 0 else float('inf')
        
        result = {
            "indices": indices,
            "distances": distances,
            "search_time": search_time,
            "qps": qps,
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
        # Calculate QPS with zero protection
        timing_info["queries_per_second"] = query_vectors.shape[0] / search_time if search_time > 0 else float('inf')
        
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
    
    def get_models_info(self, model_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        获取模型信息用于重构误差计算
        
        Args:
            model_names: 模型名称列表
            
        Returns:
            模型信息字典 {model_name: {index: faiss_index, metadata: metadata_dict}}
        """
        models_info = {}
        
        for model_name in model_names:
            try:
                # 确保模型已加载
                if not self.load_model(model_name):
                    self.logger.warning(f"Failed to load model for info: {model_name}")
                    continue
                
                # 获取模型信息
                model_info = self.loaded_models[model_name]
                models_info[model_name] = {
                    "index": model_info["index"],
                    "metadata": model_info["metadata"]
                }
                
            except Exception as e:
                self.logger.error(f"Failed to get info for model {model_name}: {str(e)}")
                continue
        
        return models_info
    
    def get_model_list(self) -> List[str]:
        """Get available model list - 支持按dataset分组的目录结构"""
        models = []
        
        if not os.path.exists(self.models_dir):
            return models
        
        # 遍历models目录下的所有项目
        for item in os.listdir(self.models_dir):
            item_path = os.path.join(self.models_dir, item)
            
            if not os.path.isdir(item_path):
                continue
            
            # 检查是否是直接的模型文件夹（兼容旧结构）
            index_path = os.path.join(item_path, "index.faiss")
            metadata_path = os.path.join(item_path, "metadata.json")
            
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                models.append(item)
            else:
                # 检查是否是dataset文件夹（新结构）
                # 遍历dataset文件夹内的模型
                try:
                    for model_name in os.listdir(item_path):
                        model_path = os.path.join(item_path, model_name)
                        
                        if not os.path.isdir(model_path):
                            continue
                            
                        model_index_path = os.path.join(model_path, "index.faiss")
                        model_metadata_path = os.path.join(model_path, "metadata.json")
                        
                        if os.path.exists(model_index_path) and os.path.exists(model_metadata_path):
                            # 构建完整的模型名称
                            full_model_name = f"{item}_{model_name}"
                            models.append(full_model_name)
                except PermissionError:
                    # 跳过无权限访问的文件夹
                    continue
        
        return sorted(models)
    
    def filter_models(self, dataset: str = None, algorithm: str = None,
                     k_values: List[int] = None, m_values: List[int] = None,
                     dist_metric: str = None) -> List[str]:
        """Filter models by conditions"""
        all_models = self.get_model_list()
        filtered_models = []
        
        for model_name in all_models:
            try:
                # 使用新的路径查找方法
                model_path = self._find_model_path(model_name)
                if not model_path:
                    continue
                    
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