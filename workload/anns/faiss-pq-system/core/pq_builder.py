"""
PQ Builder Module

Handles Product Quantization model training, building and saving
"""

import os
import pickle
import logging
import time
import numpy as np
import faiss
from typing import Dict, Any, List, Tuple, Optional
from itertools import product

from .utils import (
    create_model_name, ensure_dir, save_json, 
    ProgressTracker, format_time, get_memory_usage
)


class PQBuilder:
    """PQ Model Builder for training and saving FAISS indices"""
    
    def __init__(self, models_dir: str = "./models"):
        """Initialize PQ builder"""
        self.logger = logging.getLogger(__name__)
        self.models_dir = models_dir
        ensure_dir(models_dir)
        self.logger.info(f"Models directory ensured: {models_dir}")
        
    def build_models(self, base_vectors: np.ndarray, learn_vectors: np.ndarray,
                    dataset_name: str, algorithm: str, k_values: List[int],
                    m_values: List[int], dist_metric: str = "l2",
                    n_clusters: int = 256) -> List[Dict[str, Any]]:
        """Build multiple PQ models with grid search"""
        self.logger.info(f"Starting PQ model grid search")
        self.logger.info(f"Dataset: {dataset_name}")
        self.logger.info(f"Algorithm: {algorithm}")
        self.logger.info(f"K values: {k_values}")
        self.logger.info(f"M values: {m_values}")
        self.logger.info(f"Distance metric: {dist_metric}")
        if algorithm == "ivf-pq":
            self.logger.info(f"Clusters: {n_clusters}")
        
        param_combinations = list(product(k_values, m_values))
        total_combinations = len(param_combinations)
        self.logger.info(f"Total {total_combinations} parameter combinations")
        
        results = []
        tracker = ProgressTracker(total_combinations, "Building PQ models")
        
        for i, (k, m) in enumerate(param_combinations):
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Building model {i+1}/{total_combinations}: k={k}, m={m}")
            self.logger.info(f"{'='*50}")
            
            try:
                start_time = time.time()
                
                # Check if model already exists
                model_name = create_model_name(dataset_name, algorithm, k, m, n_clusters, dist_metric)
                
                # 按dataset分组存储 - 创建dataset子文件夹
                dataset_dir = os.path.join(self.models_dir, dataset_name)
                ensure_dir(dataset_dir)
                
                # 简化模型名称（移除dataset前缀，避免重复）
                simplified_model_name = create_model_name("", algorithm, k, m, n_clusters, dist_metric).lstrip("_")
                model_path = os.path.join(dataset_dir, simplified_model_name)
                # 不需要在这里创建目录，因为稍后会创建
                
                if self._model_exists(model_path):
                    self.logger.info(f"Model exists, skipping: {model_name}")
                    results.append({
                        "model_name": model_name,
                        "k": k,
                        "m": m,
                        "status": "Exists",
                        "build_time": 0.0
                    })
                    tracker.update()
                    continue
                
                # Build single model
                result = self._build_single_model(
                    base_vectors, learn_vectors, dataset_name, algorithm,
                    k, m, dist_metric, n_clusters
                )
                
                build_time = time.time() - start_time
                result["build_time"] = build_time
                results.append(result)
                
                self.logger.info(f"Model built, time: {format_time(build_time)}")
                
            except Exception as e:
                self.logger.error(f"Failed to build model (k={k}, m={m}): {str(e)}")
                results.append({
                    "model_name": f"{dataset_name}_{algorithm}_k{k}_m{m}_{dist_metric}",
                    "k": k,
                    "m": m,
                    "status": f"Failed: {str(e)}",
                    "build_time": 0.0
                })
            
            tracker.update()
        
        tracker.finish()
        self.logger.info(f"\nGrid search completed, successfully built {len([r for r in results if r['status'] == 'Success'])} models")
        
        return results
    
    def _build_single_model(self, base_vectors: np.ndarray, learn_vectors: np.ndarray,
                           dataset_name: str, algorithm: str, k: int, m: int,
                           dist_metric: str, n_clusters: int) -> Dict[str, Any]:
        """Build single PQ model"""
        dimension = base_vectors.shape[1]
        
        # Check parameter validity - only for algorithms that require dimension divisibility
        if algorithm in ["pq", "ivf-pq", "4bitfastscan"]:
            if dimension % m != 0:
                raise ValueError(f"Vector dimension {dimension} not divisible by M={m} for {algorithm} algorithm")
        # RQ algorithms don't require dimension % m == 0
        elif algorithm in ["rq", "rq_fastscan"]:
            self.logger.info(f"RQ algorithm: using M={m} (no dimension divisibility requirement)")
        
        model_name = create_model_name(dataset_name, algorithm, k, m, n_clusters, dist_metric)
        
        # 按dataset分组存储 - 创建dataset子文件夹
        dataset_dir = os.path.join(self.models_dir, dataset_name)
        ensure_dir(dataset_dir)
        
        # 简化模型名称（移除dataset前缀，避免重复）
        simplified_model_name = create_model_name("", algorithm, k, m, n_clusters, dist_metric).lstrip("_")
        model_path = os.path.join(dataset_dir, simplified_model_name)
        ensure_dir(model_path)  # 确保模型路径存在
        
        self.logger.info(f"Building model: {model_name}")
        self.logger.info(f"Base vectors: {base_vectors.shape}")
        self.logger.info(f"Learn vectors: {learn_vectors.shape}")
        
        use_inner_product = dist_metric in ["angular", "cosine", "ip"]
        self.logger.info(f"Distance metric: {dist_metric}, inner product: {use_inner_product}")
        
        # Create index
        if algorithm == "pq":
            index = self._create_pq_index(dimension, m, k, use_inner_product)
        elif algorithm == "ivf-pq":
            index = self._create_ivf_pq_index(dimension, m, k, n_clusters, use_inner_product)
        elif algorithm == "4bitfastscan":
            index = self._create_pq_fastscan_index(dimension, m, k, use_inner_product)
        elif algorithm == "rq":
            index = self._create_rq_index(dimension, m, k, use_inner_product)
        elif algorithm == "rq_fastscan":
            # Check if K value is compatible with RQ FastScan
            if k != 16:
                self.logger.warning(f"⚠️  RQ FastScan only supports K=16 (4-bit quantization), but K={k} was requested")
                self.logger.warning(f"🔄 Automatically converting to standard RQ algorithm for K={k}")
                self.logger.info(f"💡 Note: RQ FastScan is optimized for K=16. For other K values, standard RQ provides better flexibility")
                # Convert to standard RQ algorithm
                algorithm = "rq"
                index = self._create_rq_index(dimension, m, k, use_inner_product)
            else:
                index = self._create_rq_fastscan_index(dimension, m, k, use_inner_product)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Train index with algorithm-specific optimizations
        self.logger.info("Training index...")
        train_start = time.time()
        
        # 使用全部训练数据进行训练，不限制样本数量
        if algorithm in ["rq", "rq_fastscan"]:
            self.logger.info(f"Using all {len(learn_vectors):,} training samples for RQ algorithm")
            # 使用全部训练数据，不进行内存优化
            index.train(learn_vectors)
        else:
            # Standard training for PQ and other algorithms
            index.train(learn_vectors)
        
        train_time = time.time() - train_start
        self.logger.info(f"Training completed, time: {format_time(train_time)}")
        
        # Add base vectors to index
        self.logger.info("Adding base vectors to index...")
        add_start = time.time()
        
        index.add(base_vectors)
        
        add_time = time.time() - add_start
        self.logger.info(f"Adding vectors completed, time: {format_time(add_time)}")
        
        # Save model
        self._save_model(index, model_path, {
            "dataset": dataset_name,
            "algorithm": algorithm,
            "k": k,
            "m": m,
            "n_clusters": n_clusters,
            "dist_metric": dist_metric,
            "dimension": dimension,
            "num_base_vectors": base_vectors.shape[0],
            "num_learn_vectors": learn_vectors.shape[0],
            "train_time": train_time,
            "add_time": add_time,
            "memory_usage_mb": get_memory_usage()
        })
        
        return {
            "model_name": model_name,
            "k": k,
            "m": m,
            "algorithm": algorithm,
            "status": "Success",
            "train_time": train_time,
            "add_time": add_time,
            "total_vectors": base_vectors.shape[0]
        }
    
    def _create_pq_index(self, dimension: int, m: int, k: int, use_inner_product: bool):
        """Create PQ index"""
        subvector_dim = dimension // m
        
        index = faiss.IndexPQ(dimension, m, int(np.log2(k)))
        
        self.logger.info(f"Created PQ index: dim={dimension}, M={m}, K={k}")
        self.logger.info(f"Subvector dimension: {subvector_dim}")
        self.logger.info(f"Using inner product: {use_inner_product}")
        
        return index
    
    def _create_ivf_pq_index(self, dimension: int, m: int, k: int, 
                            n_clusters: int, use_inner_product: bool):
        """Create IVF-PQ index"""
        # Create quantizer based on distance metric
        if use_inner_product:
            quantizer = faiss.IndexFlatIP(dimension)
            self.logger.info("Using inner product quantizer")
        else:
            quantizer = faiss.IndexFlatL2(dimension)
            self.logger.info("Using L2 quantizer")
        
        index = faiss.IndexIVFPQ(quantizer, dimension, n_clusters, m, int(np.log2(k)))
        
        self.logger.info(f"Created IVF-PQ index: dim={dimension}, M={m}, K={k}, clusters={n_clusters}")
        
        return index
    
    def _create_pq_fastscan_index(self, dimension: int, m: int, k: int, use_inner_product: bool):
        """Create PQ FastScan index optimized for 4-bit quantization"""
        # For FastScan, we typically use 4-bit quantization (k=16)
        if k > 16:
            self.logger.warning(f"FastScan works best with 4-bit (k=16), but k={k} was specified")
        
        # Create IndexPQFastScan - optimized for SIMD operations
        nbits = 4 if k <= 16 else int(np.log2(k))
        index = faiss.IndexPQFastScan(dimension, m, nbits)
        
        # Set distance metric
        if use_inner_product:
            index.metric_type = faiss.METRIC_INNER_PRODUCT
            self.logger.info("Using inner product metric for FastScan")
        else:
            index.metric_type = faiss.METRIC_L2
            self.logger.info("Using L2 metric for FastScan")
        
        self.logger.info(f"Created PQ FastScan index: dim={dimension}, M={m}, nbits={nbits}")
        self.logger.info(f"FastScan optimizations: 4-bit quantization, SIMD acceleration")
        self.logger.info(f"Expected memory reduction: ~{100*(1-nbits/32):.1f}% vs float32")
        
        return index
    
    def _create_rq_index(self, dimension: int, m: int, k: int, use_inner_product: bool):
        """Create Residual Quantizer (RQ) index with optimized settings"""
        # Convert K to nbits for RQ (K = 2^nbits)
        if k <= 0 or (k & (k - 1)) != 0:  # Check if k is power of 2
            raise ValueError(f"K value {k} must be a power of 2 for RQ algorithm")
        
        nbits = int(np.log2(k))
        
        # Create IndexResidualQuantizer with optimized parameters
        index = faiss.IndexResidualQuantizer(dimension, m, nbits)
        
        # Configure RQ-specific parameters for better performance
        if hasattr(index, 'rq'):
            # Set beam search size based on K value for better accuracy/speed tradeoff
            optimal_beam_size = min(max(16, k // 4), 64)  # Adaptive beam size
            index.rq.max_beam_size = optimal_beam_size
            
            # 不使用训练样本限制，使用全部训练数据以获得最佳精度
            self.logger.info(f"RQ configuration: beam_size={optimal_beam_size}, using all training data")
        
        # Set distance metric
        if use_inner_product:
            index.metric_type = faiss.METRIC_INNER_PRODUCT
            self.logger.info("Using inner product metric for RQ")
        else:
            index.metric_type = faiss.METRIC_L2
            self.logger.info("Using L2 metric for RQ")
        
        # Log comprehensive information
        self.logger.info(f"Created Residual Quantizer index: dim={dimension}, M={m}, K={k}, nbits={nbits}")
        self.logger.info(f"RQ advantages: Better accuracy than PQ, additive quantization structure")
        
        # Estimate training time more accurately
        complexity_factor = m * nbits * 0.5  # More realistic time estimation
        estimated_minutes = max(1, int(complexity_factor))
        self.logger.info(f"Estimated training time: ~{estimated_minutes} minutes for large datasets")
        
        return index
    
    def _create_rq_fastscan_index(self, dimension: int, m: int, k: int, use_inner_product: bool):
        """Create Residual Quantizer FastScan index with enhanced configuration"""
        # Check if K is supported by FastScan (powers of 2, typically 4-bit to 8-bit)
        if k <= 0 or (k & (k - 1)) != 0:  # Check if k is power of 2
            raise ValueError(f"K value {k} must be a power of 2 for RQ FastScan algorithm")
        
        nbits = int(np.log2(k))
        
        # RQ FastScan ONLY supports 4-bit quantization (K=16) according to FAISS documentation
        if nbits == 4:  # K=16 is the ONLY supported configuration
            self.logger.info(f"✅ Using RQ FastScan: K={k} (4-bit) - FAISS-supported configuration")
            
            try:
                # Create IndexResidualQuantizerFastScan
                index = faiss.IndexResidualQuantizerFastScan(dimension, m, nbits)
                
                # Configure FastScan-specific optimizations if available
                if hasattr(index, 'rq'):
                    # Set conservative beam size for FastScan (smaller than standard RQ)
                    optimal_beam_size = min(max(8, k // 8), 32)  # Smaller beam for speed
                    index.rq.max_beam_size = optimal_beam_size
                    self.logger.info(f"RQ FastScan configuration: beam_size={optimal_beam_size} (optimized for speed)")
                
                # Set distance metric
                if use_inner_product:
                    index.metric_type = faiss.METRIC_INNER_PRODUCT
                    self.logger.info("Using inner product metric for RQ FastScan")
                else:
                    index.metric_type = faiss.METRIC_L2
                    self.logger.info("Using L2 metric for RQ FastScan")
                
                # Log detailed performance information
                self.logger.info(f"Created RQ FastScan index: dim={dimension}, M={m}, K={k}, nbits={nbits}")
                self.logger.info(f"🚀 FastScan advantages: SIMD acceleration, ~2-4x faster than standard RQ")
                self.logger.info(f"📊 Memory optimization: 4-bit quantization, ~87% memory reduction vs float32")
                self.logger.info("💡 4-bit quantization: Optimal speed/memory tradeoff for RQ FastScan")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to create RQ FastScan index: {str(e)}")
                self.logger.error(f"FAISS Error details: {repr(e)}")
                self.logger.warning("🔄 Falling back to standard RQ algorithm")
                # Avoid infinite recursion by calling standard RQ directly
                return self._create_rq_index(dimension, m, k, use_inner_product)
            
        else:
            # Fall back to standard RQ for ALL other K values
            self.logger.warning(f"⚠️  RQ FastScan ONLY supports K=16 (4-bit), but K={k} ({nbits}-bit) was requested")
            self.logger.warning("🔄 Falling back to standard RQ algorithm for non-K=16 configurations")
            self.logger.info("💡 For RQ FastScan, use K=16. For other K values, standard RQ provides better compatibility")
            # Avoid infinite recursion by calling standard RQ directly
            return self._create_rq_index(dimension, m, k, use_inner_product)
            
        return index
    
    def _save_model(self, index, model_path: str, metadata: Dict[str, Any]) -> None:
        """Save model and metadata"""
        self.logger.info(f"Saving model to: {model_path}")
        
        # Save FAISS index
        index_path = os.path.join(model_path, "index.faiss")
        faiss.write_index(index, index_path)
        self.logger.info(f"Index saved: {index_path}")
        
        # Save metadata
        metadata_path = os.path.join(model_path, "metadata.json")
        
        # Add index file size info
        metadata["index_size_mb"] = os.path.getsize(index_path) / 1024 / 1024
        
        save_json(metadata, metadata_path)
        self.logger.info(f"Metadata saved: {metadata_path}")
        
        self.logger.info("Model saved successfully")
    
    def _model_exists(self, model_path: str) -> bool:
        """Check if model exists"""
        index_path = os.path.join(model_path, "index.faiss")
        metadata_path = os.path.join(model_path, "metadata.json")
        
        return os.path.exists(index_path) and os.path.exists(metadata_path)
    
    def list_built_models(self) -> List[Dict[str, Any]]:
        """List all built models - 支持按dataset分组的目录结构"""
        models = []
        
        if not os.path.exists(self.models_dir):
            return models
        
        # 遍历models目录下的所有项目
        for item_name in os.listdir(self.models_dir):
            item_path = os.path.join(self.models_dir, item_name)
            
            if not os.path.isdir(item_path):
                continue
            
            # 检查是否是直接的模型文件夹（兼容旧结构）
            if self._model_exists(item_path):
                try:
                    metadata_path = os.path.join(item_path, "metadata.json")
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        import json
                        metadata = json.load(f)
                    
                    models.append({
                        "model_name": item_name,
                        "path": item_path,
                        "dataset_folder": None,  # 直接在models下
                        **metadata
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Failed to read model metadata {item_name}: {str(e)}")
            
            # 检查是否是dataset文件夹（新结构）
            else:
                # 遍历dataset文件夹内的模型
                for model_name in os.listdir(item_path):
                    model_path = os.path.join(item_path, model_name)
                    
                    if not os.path.isdir(model_path):
                        continue
                        
                    if self._model_exists(model_path):
                        try:
                            metadata_path = os.path.join(model_path, "metadata.json")
                            with open(metadata_path, 'r', encoding='utf-8') as f:
                                import json
                                metadata = json.load(f)
                            
                            # 构建完整的模型名称
                            full_model_name = f"{item_name}_{model_name}"
                            
                            models.append({
                                "model_name": full_model_name,
                                "simplified_name": model_name,
                                "path": model_path,
                                "dataset_folder": item_name,
                                **metadata
                            })
                            
                        except Exception as e:
                            self.logger.warning(f"Failed to read model metadata {item_name}/{model_name}: {str(e)}")
        
        return models
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get model information - 支持按dataset分组的目录结构"""
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
            if self._model_exists(model_path):
                try:
                    metadata_path = os.path.join(model_path, "metadata.json")
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        import json
                        metadata = json.load(f)
                    
                    # Add file size info
                    index_path = os.path.join(model_path, "index.faiss")
                    if os.path.exists(index_path):
                        metadata["index_size_mb"] = os.path.getsize(index_path) / 1024 / 1024
                    
                    # Add path info
                    metadata["model_path"] = model_path
                    
                    return metadata
                    
                except Exception as e:
                    self.logger.error(f"Failed to read model metadata at {model_path}: {str(e)}")
                    continue
        
        # 模型未找到
        self.logger.warning(f"Model not found: {model_name}")
        return None
    
    def delete_model(self, model_name: str) -> bool:
        """Delete model"""
        model_path = os.path.join(self.models_dir, model_name)
        
        if not os.path.exists(model_path):
            self.logger.warning(f"Model does not exist: {model_name}")
            return False
        
        try:
            import shutil
            shutil.rmtree(model_path)
            self.logger.info(f"Model deleted: {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete model {model_name}: {str(e)}")
            return False
    
    def get_build_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get build summary"""
        total_models = len(results)
        successful_models = len([r for r in results if r['status'] == 'Success'])
        failed_models = len([r for r in results if r['status'] != 'Success' and r['status'] != 'Exists'])
        existing_models = len([r for r in results if r['status'] == 'Exists'])
        
        total_build_time = sum([r.get('build_time', 0) for r in results])
        
        summary = {
            "total_models": total_models,
            "successful_models": successful_models,
            "failed_models": failed_models,
            "existing_models": existing_models,
            "total_build_time": total_build_time,
            "average_build_time": total_build_time / max(1, successful_models),
            "success_rate": successful_models / max(1, total_models - existing_models) * 100
        }
        
        return summary 