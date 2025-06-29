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
                
            # Extract data from new dictionary format
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
        
        for model_name, metrics in evaluation_results.items():
            lines.append(f"Model: {model_name}")
            lines.append("-" * 40)
            
            for key, value in metrics.items():
                if isinstance(value, float):
                    if 'recall' in key or 'hit_rate' in key:
                        lines.append(f"  {key}: {value:.4f}")
                    elif 'time' in key:
                        lines.append(f"  {key}: {value:.3f}s")
                    else:
                        lines.append(f"  {key}: {value:.2f}")
                else:
                    lines.append(f"  {key}: {value}")
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
    

 