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
                model_path = os.path.join(self.models_dir, model_name)
                
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
        
        # Check parameter validity
        if dimension % m != 0:
            raise ValueError(f"Vector dimension {dimension} not divisible by M={m}")
        
        model_name = create_model_name(dataset_name, algorithm, k, m, n_clusters, dist_metric)
        model_path = os.path.join(self.models_dir, model_name)
        ensure_dir(model_path)
        
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
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Train index
        self.logger.info("Training index...")
        train_start = time.time()
        
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
        """List all built models"""
        models = []
        
        if not os.path.exists(self.models_dir):
            return models
        
        for model_name in os.listdir(self.models_dir):
            model_path = os.path.join(self.models_dir, model_name)
            
            if not os.path.isdir(model_path):
                continue
                
            if self._model_exists(model_path):
                try:
                    metadata_path = os.path.join(model_path, "metadata.json")
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        import json
                        metadata = json.load(f)
                    
                    models.append({
                        "model_name": model_name,
                        "path": model_path,
                        **metadata
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Failed to read model metadata {model_name}: {str(e)}")
        
        return models
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get model information"""
        model_path = os.path.join(self.models_dir, model_name)
        
        if not self._model_exists(model_path):
            return None
        
        try:
            metadata_path = os.path.join(model_path, "metadata.json")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                import json
                metadata = json.load(f)
            
            # Add file size info
            index_path = os.path.join(model_path, "index.faiss")
            if os.path.exists(index_path):
                metadata["index_size_mb"] = os.path.getsize(index_path) / 1024 / 1024
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to get model info {model_name}: {str(e)}")
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