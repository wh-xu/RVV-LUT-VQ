"""
Evaluator Module

Calculates evaluation metrics for search results such as recall@k and saves results
"""

import os
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

from .utils import ensure_dir, get_timestamp, format_time


class Evaluator:
    """Evaluator for computing search performance metrics"""
    
    def __init__(self, results_dir: str = "./results"):
        """Initialize evaluator"""
        self.logger = logging.getLogger(__name__)
        self.results_dir = results_dir
        ensure_dir(results_dir)
        self.logger.info(f"Results directory ensured: {results_dir}")
        
    def evaluate_search_results(self, search_results: Dict[str, Any],
                               groundtruth: np.ndarray, recall_k: List[int] = [1, 10, 100]) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate search results
        
        Args:
            search_results: Search results dictionary {model_name: result_dict}
            groundtruth: Ground truth labels
            recall_k: List of k values for recall@k
            
        Returns:
            Evaluation results dictionary
        """
        self.logger.info("Starting evaluation...")
        
        evaluation_results = {}
        
        for model_name, result in search_results.items():
            if result is None:
                self.logger.warning(f"Skipping failed model: {model_name}")
                continue
                
            # Extract data from new dictionary format with safety checks
            if not isinstance(result, dict):
                self.logger.warning(f"Skipping model {model_name}: invalid result format")
                continue
                
            required_keys = ["indices", "distances", "search_time", "qps"]
            missing_keys = [key for key in required_keys if key not in result]
            if missing_keys:
                self.logger.warning(f"Skipping model {model_name}: missing keys {missing_keys}")
                continue
                
            indices = result["indices"]
            distances = result["distances"]
            timing_info = {
                "search_time": result["search_time"],
                "queries_per_second": result["qps"]
            }
            
            self.logger.info(f"Evaluating model: {model_name}")
            
            # Calculate recall metrics
            recall_scores = self._calculate_recall(indices, groundtruth, recall_k)
            
            # Calculate other metrics
            other_metrics = self._calculate_other_metrics(indices, groundtruth, distances)
            
            evaluation_results[model_name] = {
                "recall@1": recall_scores.get("recall@1", 0.0),
                "recall@10": recall_scores.get("recall@10", 0.0),
                "recall@100": recall_scores.get("recall@100", 0.0),
                "search_time": timing_info["search_time"],
                "qps": timing_info["queries_per_second"],
                "mean_rank": other_metrics.get("mean_rank", 0.0),
                "hit_rate": other_metrics.get("hit_rate", 0.0)
            }
            
            # Log results
            self._log_evaluation_results(model_name, recall_scores, timing_info)
        
        self.logger.info("Evaluation completed")
        return evaluation_results
    
    def evaluate_with_reconstruction_error(self, search_results: Dict[str, Any], 
                                         models_info: Dict[str, Any],
                                         query_vectors: np.ndarray,
                                         groundtruth: np.ndarray, 
                                         recall_k: List[int] = [1, 10, 100]) -> Dict[str, Dict[str, Any]]:
        """
        评估搜索结果，包含重构误差计算
        
        Args:
            search_results: 搜索结果字典 {model_name: result_dict}
            models_info: 模型信息字典 {model_name: model_info}
            query_vectors: 查询向量数组
            groundtruth: 真实标签
            recall_k: recall@k 的 k 值列表
            
        Returns:
            评估结果字典，包含重构误差指标
        """
        self.logger.info("Starting evaluation with reconstruction error...")
        
        evaluation_results = {}
        
        for model_name, result in search_results.items():
            if result is None:
                self.logger.warning(f"Skipping failed model: {model_name}")
                continue
                
            # Extract data from search results
            indices = result["indices"]
            distances = result["distances"]
            timing_info = {
                "search_time": result["search_time"],
                "queries_per_second": result["qps"]
            }
            
            self.logger.info(f"Evaluating model with reconstruction error: {model_name}")
            
            # Calculate recall metrics
            recall_scores = self._calculate_recall(indices, groundtruth, recall_k)
            
            # Calculate other metrics
            other_metrics = self._calculate_other_metrics(indices, groundtruth, distances)
            
            # Calculate reconstruction error if model info is available
            reconstruction_metrics = {}
            if model_name in models_info:
                model_info = models_info[model_name]
                reconstruction_metrics = self._calculate_reconstruction_error(query_vectors, model_info)
            else:
                self.logger.warning(f"Model info not found for {model_name}, skipping reconstruction error")
                reconstruction_metrics = {
                    "mse": None,
                    "mae": None,
                    "max_error": None,
                    "support_status": "no_model_info"
                }
            
            # Combine all metrics
            evaluation_results[model_name] = {
                "recall@1": recall_scores.get("recall@1", 0.0),
                "recall@10": recall_scores.get("recall@10", 0.0),
                "recall@100": recall_scores.get("recall@100", 0.0),
                "search_time": timing_info["search_time"],
                "qps": timing_info["queries_per_second"],
                "mean_rank": other_metrics.get("mean_rank", 0.0),
                "hit_rate": other_metrics.get("hit_rate", 0.0),
                "reconstruction_mse": reconstruction_metrics.get("mse"),
                "reconstruction_mae": reconstruction_metrics.get("mae"),
                "reconstruction_max_error": reconstruction_metrics.get("max_error"),
                "reconstruction_support_status": reconstruction_metrics.get("support_status", "unknown")
            }
            
            # Log results with reconstruction error
            self._log_evaluation_results_with_reconstruction(
                model_name, recall_scores, timing_info, reconstruction_metrics
            )
        
        self.logger.info("Evaluation with reconstruction error completed")
        return evaluation_results
    
    def save_evaluation_results(self, evaluation_results: Dict[str, Dict[str, Any]], 
                               dataset_name: str, output_filename: str = None) -> Dict[str, str]:
        """Save evaluation results to files"""
        timestamp = get_timestamp()
        result_dir = os.path.join(self.results_dir, f"{dataset_name}_results_{timestamp}")
        ensure_dir(result_dir)
        
        # Save CSV
        csv_filename = output_filename if output_filename else f"evaluation_results_{timestamp}.csv"
        csv_path = os.path.join(result_dir, csv_filename)
        
        rows = []
        for model_name, metrics in evaluation_results.items():
            row = {"model_name": model_name}
            row.update(metrics)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        
        # Save report
        report_path = os.path.join(result_dir, f"evaluation_report_{timestamp}.txt")
        report_content = self._generate_simple_report(evaluation_results, dataset_name)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"Results saved to: {result_dir}")
        
        return {
            "csv_path": csv_path,
            "report_path": report_path,
            "result_dir": result_dir
        }
    
    def _generate_simple_report(self, evaluation_results: Dict[str, Dict[str, Any]], 
                               dataset_name: str) -> str:
        """Generate simple evaluation report"""
        lines = []
        lines.append("=" * 80)
        lines.append(f"FAISS PQ Evaluation Report - {dataset_name}")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Dataset: {dataset_name}")
        lines.append(f"Models evaluated: {len(evaluation_results)}")
        lines.append("")
        
        # Check if any models have reconstruction error metrics
        has_reconstruction = any(
            'reconstruction_mse' in metrics for metrics in evaluation_results.values()
        )
        
        for model_name, metrics in evaluation_results.items():
            lines.append(f"Model: {model_name}")
            lines.append("-" * 40)
            
            # Search performance metrics
            lines.append("Search Performance:")
            for key, value in metrics.items():
                if key.startswith('reconstruction_'):
                    continue  # Handle reconstruction metrics separately
                    
                if isinstance(value, float):
                    if 'recall' in key or 'hit_rate' in key:
                        lines.append(f"  {key}: {value:.4f}")
                    elif 'time' in key:
                        lines.append(f"  {key}: {value:.3f}s")
                    else:
                        lines.append(f"  {key}: {value:.2f}")
                else:
                    lines.append(f"  {key}: {value}")
            
            # Reconstruction error metrics (if available)
            if has_reconstruction:
                lines.append("")
                lines.append("Reconstruction Error:")
                
                support_status = metrics.get('reconstruction_support_status', 'unknown')
                lines.append(f"  Support Status: {support_status}")
                
                if support_status not in ['no_api', 'zero_code_size', 'code_size_error', 'error', 'no_model_info']:
                    if metrics.get('reconstruction_mse') is not None:
                        lines.append(f"  MSE: {metrics['reconstruction_mse']:.6f}")
                    if metrics.get('reconstruction_mae') is not None:
                        lines.append(f"  MAE: {metrics['reconstruction_mae']:.6f}")
                    if metrics.get('reconstruction_max_error') is not None:
                        lines.append(f"  Max Error: {metrics['reconstruction_max_error']:.6f}")
                        
                    if support_status == 'limited_4bit':
                        lines.append("  Note: Reconstruction error values are approximate due to 4-bit quantization")
                else:
                    lines.append(f"  Status: Not supported or error occurred")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def get_best_metrics(self, evaluation_results: Dict[str, Dict[str, Any]]) -> Dict[str, Tuple[str, float]]:
        """Find best performing models for each metric"""
        best_metrics = {}
        
        if not evaluation_results:
            return best_metrics
        
        # Get all metric names
        metric_names = set()
        for metrics in evaluation_results.values():
            metric_names.update(metrics.keys())
        
        for metric_name in metric_names:
            if metric_name == "model_name":
                continue
                
            best_model = None
            best_value = None
            
            for model_name, metrics in evaluation_results.items():
                if metric_name in metrics:
                    value = metrics[metric_name]
                    if isinstance(value, (int, float)):
                        # For recall and hit_rate, higher is better
                        # For search_time, lower is better
                        # For qps, higher is better
                        if best_value is None:
                            best_model = model_name
                            best_value = value
                        elif metric_name == "search_time":
                            if value < best_value:
                                best_model = model_name
                                best_value = value
                        else:
                            if value > best_value:
                                best_model = model_name
                                best_value = value
            
            if best_model is not None:
                best_metrics[metric_name] = (best_model, best_value)
        
        return best_metrics
    
    def _calculate_recall(self, retrieved_indices: np.ndarray, groundtruth: np.ndarray,
                         recall_k: List[int]) -> Dict[str, float]:
        """Calculate recall@k metrics"""
        num_queries = retrieved_indices.shape[0]
        recall_scores = {}
        
        for k in recall_k:
            if k > retrieved_indices.shape[1]:
                self.logger.warning(f"Cannot calculate recall@{k}: only {retrieved_indices.shape[1]} results returned, but {k} requested")
                recall_scores[f"recall@{k}"] = 0.0
                continue
                
            correct_queries = 0
            
            for i in range(num_queries):
                retrieved_k = retrieved_indices[i, :k]
                closest_true_neighbor = groundtruth[i, 0]
                
                if closest_true_neighbor in retrieved_k:
                    correct_queries += 1
            
            recall_scores[f"recall@{k}"] = correct_queries / num_queries
        
        return recall_scores
    
    def _calculate_other_metrics(self, retrieved_indices: np.ndarray, groundtruth: np.ndarray,
                               distances: np.ndarray) -> Dict[str, float]:
        """Calculate additional evaluation metrics"""
        metrics = {}
        
        mean_rank = self._calculate_mean_rank(retrieved_indices, groundtruth)
        metrics["mean_rank"] = mean_rank
        
        mrr = self._calculate_mrr(retrieved_indices, groundtruth)
        metrics["mrr"] = mrr
        
        hit_rate = self._calculate_hit_rate(retrieved_indices, groundtruth)
        metrics["hit_rate"] = hit_rate
        
        if distances is not None:
            metrics["mean_distance"] = float(np.mean(distances[:, 0]))
            metrics["std_distance"] = float(np.std(distances[:, 0]))
        
        return metrics
    
    def _calculate_reconstruction_error(self, query_vectors: np.ndarray, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算重构误差 - 基于FAISS真实API能力
        
        Args:
            query_vectors: 查询向量数组
            model_info: 模型信息字典，包含 index 和 metadata
            
        Returns:
            dict: 包含重构误差指标，如果不支持则返回None值
        """
        try:
            index = model_info["index"]
            metadata = model_info["metadata"]
            algorithm = metadata.get("algorithm", "unknown")
            
            # 内存优化：大查询集采样
            if len(query_vectors) > 1000:
                sample_indices = np.random.choice(len(query_vectors), 1000, replace=False)
                sample_vectors = query_vectors[sample_indices].astype(np.float32)
            else:
                sample_vectors = query_vectors.astype(np.float32)
                
            # 检查是否支持重构
            if not hasattr(index, 'sa_encode') or not hasattr(index, 'sa_decode'):
                self.logger.warning(f"算法 {algorithm} 不支持编解码接口")
                return {
                    "mse": None,
                    "mae": None, 
                    "max_error": None,
                    "support_status": "no_api"
                }
                
            # 检查代码大小
            try:
                code_size = index.sa_code_size()
                if code_size == 0:
                    return {
                        "mse": None,
                        "mae": None,
                        "max_error": None,
                        "support_status": "zero_code_size"
                    }
            except Exception as e:
                self.logger.warning(f"无法获取代码大小: {e}")
                return {
                    "mse": None,
                    "mae": None,
                    "max_error": None,
                    "support_status": "code_size_error"
                }
            
            # 算法特定处理
            if "fastscan" in algorithm.lower() or "4bit" in algorithm.lower():
                # FastScan变体：计算但标记为近似值
                support_status = "limited_4bit"
                self.logger.info(f"算法 {algorithm} 为FastScan变体，重构误差为近似值")
            else:
                support_status = "full_support"
                
            # 执行编解码
            start_time = time.time()
            
            # Step 1: 编码
            codes = index.sa_encode(sample_vectors)
            
            # Step 2: 解码  
            reconstructed = index.sa_decode(codes)
            
            encode_decode_time = time.time() - start_time
            
            # Step 3: 计算误差指标
            mse = np.mean((sample_vectors - reconstructed) ** 2)
            mae = np.mean(np.abs(sample_vectors - reconstructed))
            max_error = np.max(np.abs(sample_vectors - reconstructed))
            
            # 记录性能信息
            self.logger.info(f"重构误差计算完成 - 算法: {algorithm}, "
                            f"样本数: {len(sample_vectors)}, "
                            f"编解码时间: {encode_decode_time:.3f}s")
            
            return {
                "mse": float(mse),
                "mae": float(mae),
                "max_error": float(max_error),
                "support_status": support_status,
                "sample_size": len(sample_vectors),
                "encode_decode_time": encode_decode_time,
                "code_size_bytes": int(code_size)
            }
            
        except Exception as e:
            self.logger.error(f"重构误差计算失败: {e}")
            return {
                "mse": None,
                "mae": None,
                "max_error": None,
                "support_status": "error",
                "error_message": str(e)
            }
    
    def _calculate_mean_rank(self, retrieved_indices: np.ndarray, groundtruth: np.ndarray) -> float:
        """Calculate mean rank of closest true neighbor"""
        num_queries = retrieved_indices.shape[0]
        total_rank = 0.0
        valid_queries = 0
        
        for i in range(num_queries):
            true_neighbor = groundtruth[i, 0]
            rank_positions = np.where(retrieved_indices[i] == true_neighbor)[0]
            
            if len(rank_positions) > 0:
                total_rank += rank_positions[0] + 1
                valid_queries += 1
        
        return total_rank / max(1, valid_queries)
    
    def _calculate_mrr(self, retrieved_indices: np.ndarray, groundtruth: np.ndarray) -> float:
        """Calculate Mean Reciprocal Rank"""
        num_queries = retrieved_indices.shape[0]
        rr_sum = 0.0
        
        for i in range(num_queries):
            true_neighbors = set(groundtruth[i, :min(10, groundtruth.shape[1])])
            
            for rank, retrieved_id in enumerate(retrieved_indices[i]):
                if retrieved_id in true_neighbors:
                    rr_sum += 1.0 / (rank + 1)
                    break
        
        return rr_sum / num_queries
    
    def _calculate_hit_rate(self, retrieved_indices: np.ndarray, groundtruth: np.ndarray) -> float:
        """Calculate hit rate (proportion of queries with at least one correct result)"""
        num_queries = retrieved_indices.shape[0]
        hits = 0
        
        for i in range(num_queries):
            true_neighbors = set(groundtruth[i, :min(10, groundtruth.shape[1])])
            retrieved_set = set(retrieved_indices[i])
            
            if len(true_neighbors.intersection(retrieved_set)) > 0:
                hits += 1
        
        return hits / num_queries
    
    def _log_evaluation_results(self, model_name: str, recall_scores: Dict[str, float],
                              timing_info: Dict[str, float]) -> None:
        """Log evaluation results"""
        self.logger.info(f"Model {model_name} evaluation results:")
        
        for metric, score in recall_scores.items():
            self.logger.info(f"  {metric}: {score:.4f}")
        
        if "search_time" in timing_info:
            self.logger.info(f"  Search time: {format_time(timing_info['search_time'])}")
        
        if "queries_per_second" in timing_info:
            self.logger.info(f"  QPS: {timing_info['queries_per_second']:.2f}")
    
    def _log_evaluation_results_with_reconstruction(self, model_name: str, recall_scores: Dict[str, float],
                                                     timing_info: Dict[str, float],
                                                     reconstruction_metrics: Dict[str, Any]) -> None:
        """Log evaluation results including reconstruction error"""
        self.logger.info(f"Model {model_name} evaluation results:")
        
        for metric, score in recall_scores.items():
            self.logger.info(f"  {metric}: {score:.4f}")
        
        if "search_time" in timing_info:
            self.logger.info(f"  Search time: {format_time(timing_info['search_time'])}")
        
        if "queries_per_second" in timing_info:
            self.logger.info(f"  QPS: {timing_info['queries_per_second']:.2f}")
        
        # Log reconstruction error metrics
        if reconstruction_metrics.get("mse") is not None:
            self.logger.info(f"  Reconstruction MSE: {reconstruction_metrics['mse']:.6f}")
        if reconstruction_metrics.get("mae") is not None:
            self.logger.info(f"  Reconstruction MAE: {reconstruction_metrics['mae']:.6f}")
        if reconstruction_metrics.get("max_error") is not None:
            self.logger.info(f"  Reconstruction Max Error: {reconstruction_metrics['max_error']:.6f}")
        
        support_status = reconstruction_metrics.get("support_status", "unknown")
        self.logger.info(f"  Reconstruction Support: {support_status}")
        
        if "sample_size" in reconstruction_metrics:
            self.logger.info(f"  Reconstruction Sample Size: {reconstruction_metrics['sample_size']}")
        if "encode_decode_time" in reconstruction_metrics:
            self.logger.info(f"  Encode/Decode Time: {reconstruction_metrics['encode_decode_time']:.3f}s")
    
    def _determine_result_folder(self, evaluation_results: Dict[str, Dict[str, Any]]) -> str:
        """Determine result folder name"""
        model_names = list(evaluation_results.keys())
        
        if not model_names:
            timestamp = get_timestamp()
            return f"empty_search_{timestamp}"
        
        if len(model_names) == 1:
            return model_names[0]
        
        common_prefix = self._find_common_prefix(model_names)
        timestamp = get_timestamp()
        
        if common_prefix and len(common_prefix) > 5:
            common_prefix = common_prefix.rstrip('_')
            return f"{common_prefix}_{timestamp}"
        else:
            first_model_params = self._parse_model_name(model_names[0])
            dataset = first_model_params.get("dataset", "unknown")
            algorithm = first_model_params.get("algorithm", "unknown")
            return f"{dataset}_{algorithm}_batch_{timestamp}"
    
    def _find_common_prefix(self, model_names: List[str]) -> str:
        """Find common prefix of model names"""
        if not model_names:
            return ""
        
        min_length = min(len(name) for name in model_names)
        
        common_prefix = ""
        for i in range(min_length):
            char = model_names[0][i]
            if all(name[i] == char for name in model_names):
                common_prefix += char
            else:
                break
        
        return common_prefix
    

    
    def _parse_model_name(self, model_name: str) -> Dict[str, Any]:
        """Parse model name to extract parameters"""
        params = {}
        
        try:
            parts = model_name.split("_")
            
            if len(parts) >= 2:
                params["dataset"] = parts[0]
                params["algorithm"] = parts[1]
            
            for part in parts:
                if part.startswith("k") and len(part) > 1:
                    params["k"] = int(part[1:])
                elif part.startswith("m") and len(part) > 1:
                    params["m"] = int(part[1:])
                elif part.startswith("clusters") and len(part) > 8:
                    params["n_clusters"] = int(part[8:])
                elif part in ["l2", "angular", "ip", "cosine"]:
                    params["dist_metric"] = part
                    
        except Exception as e:
            self.logger.warning(f"Failed to parse model name {model_name}: {str(e)}")
        
        return params
    

 