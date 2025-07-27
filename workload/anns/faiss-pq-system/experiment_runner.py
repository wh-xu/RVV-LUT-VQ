#!/usr/bin/env python3
"""
自动化实验运行脚本
完整的PQ vs RQ性能对比实验
"""

import os
import sys
import logging
import time
import math
import csv
import subprocess
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """实验配置类"""
    dataset: str
    algorithm: str
    k: int
    m: int
    target_bits: int
    actual_bits: int
    ideal_m: float
    dimension: int

class ExperimentRunner:
    """自动化实验运行器"""
    
    def __init__(self):
        self.datasets = ["glove", "sift1m", "gist1m", "deep10m"]
        self.k_values = [4, 8, 16, 32, 64, 128, 256]
        self.target_bits = [32, 64, 128]
        
        # 数据集维度映射
        self.dataset_dimensions = {
            "glove": 100,
            "sift1m": 128, 
            "gist1m": 960,
            "deep10m": 96
        }
        
        # 算法策略映射
        self.algorithm_strategy = {
            "glove": "pq",      # 需要维度整除
            "sift1m": "pq",     # 需要维度整除
            "gist1m": "rq",     # 维度灵活
            "deep10m": "rq"     # 维度灵活
        }
        
        self.results = {
            "build_results": [],
            "search_results": [],
            "errors": []
        }
    
    def get_valid_m_values(self, dataset: str, dimension: int) -> List[int]:
        """获取有效的M值列表"""
        if dataset in ["glove", "sift1m"]:
            # PQ算法需要维度能被M整除
            return [m for m in range(1, dimension + 1) if dimension % m == 0]
        else:
            # RQ算法M值可以任意选择
            return list(range(1, dimension + 1))
    
    def calculate_optimal_m(self, k: int, target_bits: int, dataset: str, dimension: int, algorithm: str) -> Tuple[int, float, int]:
        """计算最优M值"""
        # 根据算法调整ideal_m计算
        if algorithm == "4bitfastscan":
            # 4BitFastScan固定4-bit
            ideal_m = target_bits / 4
        else:
            # 其他算法使用log2(K)
            ideal_m = target_bits / math.log2(k)
            
        valid_m_values = self.get_valid_m_values(dataset, dimension)
        
        if dataset in ["glove", "sift1m"]:
            # PQ算法：选择最接近的有效因子
            chosen_m = min(valid_m_values, key=lambda m: abs(m - ideal_m))
        else:
            # RQ算法：四舍五入到最近整数
            chosen_m = round(ideal_m)
            chosen_m = max(1, min(chosen_m, dimension))  # 确保在合理范围内
        
        # 根据算法计算实际bit数
        if algorithm == "4bitfastscan":
            # 4BitFastScan固定使用4-bit
            actual_bits = chosen_m * 4
        else:
            # 其他算法使用log2(K)
            actual_bits = chosen_m * math.log2(k)
        
        return chosen_m, ideal_m, int(actual_bits)
    
    def select_algorithm(self, dataset: str, k: int) -> str:
        """选择算法"""
        base_algorithm = self.algorithm_strategy[dataset]
        
        # K=16时优先使用FastScan
        if k == 16:
            if base_algorithm == "pq":
                return "4bitfastscan"
            elif base_algorithm == "rq":
                return "rq_fastscan"
        
        return base_algorithm
    
    def generate_experiment_configs(self, dataset: str) -> List[ExperimentConfig]:
        """生成实验配置"""
        configs = []
        dimension = self.dataset_dimensions[dataset]
        
        logger.info(f"📋 生成{dataset}数据集实验配置 (维度={dimension})")
        
        for target_bits in self.target_bits:
            for k in self.k_values:
                algorithm = self.select_algorithm(dataset, k)
                chosen_m, ideal_m, actual_bits = self.calculate_optimal_m(k, target_bits, dataset, dimension, algorithm)
                
                config = ExperimentConfig(
                    dataset=dataset,
                    algorithm=algorithm,
                    k=k,
                    m=chosen_m,
                    target_bits=target_bits,
                    actual_bits=actual_bits,
                    ideal_m=ideal_m,
                    dimension=dimension
                )
                
                configs.append(config)
                logger.info(f"  K={k}, target={target_bits}bits, ideal_M={ideal_m:.2f}, chosen_M={chosen_m}, actual={actual_bits}bits, alg={algorithm}")
        
        logger.info(f"✅ 生成{len(configs)}个实验配置")
        return configs
    
    def build_single_model(self, config: ExperimentConfig) -> Dict[str, Any]:
        """构建单个模型"""
        logger.info(f"🔨 构建模型: {config.dataset}_{config.algorithm}_k{config.k}_m{config.m}")
        
        cmd = [
            "python", "main.py",
            "--mode", "build",
            "--dataset", config.dataset,
            "--algorithm", config.algorithm,
            "--k_values", str(config.k),
            "--m_values", str(config.m)
        ]
        
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1小时超时
            
            if result.returncode != 0:
                error_msg = f"模型构建失败: {config.dataset}_{config.algorithm}_k{config.k}_m{config.m}"
                logger.error(f"❌ {error_msg}")
                logger.error(f"错误输出: {result.stderr}")
                raise RuntimeError(error_msg)
            
            build_time = time.time() - start_time
            
            # 生成模型名称
            # 根据数据集确定正确的距离度量
            distance_metrics = {
                "glove": "ip",
                "sift1m": "l2", 
                "gist1m": "l2",
                "deep10m": "ip"
            }
            dist_metric = distance_metrics.get(config.dataset, "l2")
            model_name = f"{config.dataset}_{config.algorithm}_k{config.k}_m{config.m}_{dist_metric}"
            
            result_data = {
                "dataset": config.dataset,
                "algorithm": config.algorithm,
                "k": config.k,
                "m": config.m,
                "target_bits": config.target_bits,
                "actual_bits": config.actual_bits,
                "ideal_m": config.ideal_m,
                "chosen_m": config.m,
                "dimension": config.dimension,
                "status": "Success",
                "build_time": build_time,
                "model_name": model_name
            }
            
            logger.info(f"✅ 构建成功: {model_name} (耗时: {build_time:.1f}s)")
            return result_data
            
        except subprocess.TimeoutExpired:
            error_msg = f"模型构建超时: {config.dataset}_{config.algorithm}_k{config.k}_m{config.m}"
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"模型构建异常: {str(e)}"
            logger.error(f"❌ {error_msg}")
            raise
    
    def run_search_test(self, dataset: str) -> List[Dict[str, Any]]:
        """运行搜索测试"""
        logger.info(f"🔍 执行搜索测试: {dataset}")
        
        cmd = [
            "python", "main.py", 
            "--mode", "search",
            "--dataset", dataset,
            "--recall_k", "1,5,10,100",
            
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30分钟超时
            
            if result.returncode != 0:
                error_msg = f"搜索测试失败: {dataset}"
                logger.error(f"❌ {error_msg}")
                logger.error(f"错误输出: {result.stderr}")
                raise RuntimeError(error_msg)
            
            logger.info(f"✅ 搜索测试完成: {dataset}")
            
            # 解析搜索结果
            # 这里需要从main.py的输出中解析结果，或者修改main.py输出JSON格式
            search_results = self.parse_search_output(result.stdout, dataset)
            return search_results
            
        except subprocess.TimeoutExpired:
            error_msg = f"搜索测试超时: {dataset}"
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"搜索测试异常: {str(e)}"
            logger.error(f"❌ {error_msg}")
            raise
    
    def parse_search_output(self, output: str, dataset: str) -> List[Dict[str, Any]]:
        """解析搜索输出结果"""
        results = []
        
        try:
            # 解析搜索结果的简化版本
            # 查找性能指标行
            lines = output.split('\n')
            
            current_model = None
            for line in lines:
                line = line.strip()
                
                # 查找模型名称
                if '📊' in line and dataset in line:
                    current_model = line.split('📊')[1].strip()
                
                # 查找性能指标
                if current_model and any(metric in line for metric in ['recall@', 'search_time:', 'QPS:']):
                    if 'recall@1:' in line:
                        recall_1 = float(line.split(':')[1].strip())
                    elif 'recall@10:' in line:
                        recall_10 = float(line.split(':')[1].strip())
                    elif 'recall@100:' in line:
                        recall_100 = float(line.split(':')[1].strip())
                    elif 'search_time:' in line:
                        search_time = float(line.split(':')[1].strip().replace('s', ''))
                    elif 'QPS:' in line:
                        qps = float(line.split(':')[1].strip())
                        
                        # 收集一个模型的完整结果
                        if current_model:
                            model_result = {
                                "model_name": current_model,
                                "dataset": dataset,
                                "recall@1": locals().get('recall_1', 0.0),
                                "recall@10": locals().get('recall_10', 0.0),
                                "recall@100": locals().get('recall_100', 0.0),
                                "search_time": locals().get('search_time', 0.0),
                                "qps": qps
                            }
                            results.append(model_result)
                            current_model = None
            
            logger.info(f"✅ 成功解析 {len(results)} 个模型的搜索结果")
            return results
            
        except Exception as e:
            logger.warning(f"⚠️ 搜索结果解析失败: {str(e)}")
            logger.warning("返回空结果列表")
            return []
    
    def save_dataset_results(self, dataset: str, build_results: List[Dict], search_results: List[Dict]):
        """保存单个数据集的结果"""
        results_dir = f"results/{dataset}"
        
        # 保存构建结果
        build_csv_path = f"{results_dir}/build_results.csv"
        self.save_csv(build_results, build_csv_path)
        
        # 保存搜索结果
        search_csv_path = f"{results_dir}/search_results.csv"
        self.save_csv(search_results, search_csv_path)
        
        # 生成摘要
        summary_data = self.generate_dataset_summary(dataset, build_results, search_results)
        summary_csv_path = f"{results_dir}/summary.csv"
        self.save_csv([summary_data], summary_csv_path)
        
        logger.info(f"💾 {dataset}数据集结果已保存到 {results_dir}/")
    
    def save_csv(self, data: List[Dict], filepath: str):
        """保存CSV文件"""
        if not data:
            logger.warning(f"⚠️ 数据为空，跳过保存: {filepath}")
            return
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        
        logger.info(f"💾 CSV文件已保存: {filepath}")
    
    def generate_dataset_summary(self, dataset: str, build_results: List[Dict], search_results: List[Dict]) -> Dict[str, Any]:
        """生成数据集摘要"""
        return {
            "dataset": dataset,
            "total_models": len(build_results),
            "successful_builds": len([r for r in build_results if r["status"] == "Success"]),
            "total_build_time": sum(r["build_time"] for r in build_results),
            "dimension": build_results[0]["dimension"] if build_results else 0,
            "algorithms_used": list(set(r["algorithm"] for r in build_results)),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def run_dataset_experiment(self, dataset: str):
        """运行单个数据集的完整实验"""
        logger.info(f"🚀 开始{dataset}数据集实验")
        start_time = time.time()
        
        try:
            # 生成实验配置
            configs = self.generate_experiment_configs(dataset)
            
            # 构建所有模型
            build_results = []
            for i, config in enumerate(configs, 1):
                logger.info(f"📊 进度: {i}/{len(configs)} - {dataset}")
                result = self.build_single_model(config)
                build_results.append(result)
            
            # 运行搜索测试
            search_results = self.run_search_test(dataset)
            
            # 保存结果
            self.save_dataset_results(dataset, build_results, search_results)
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ {dataset}数据集实验完成! 耗时: {elapsed_time/60:.1f}分钟")
            
            return build_results, search_results
            
        except Exception as e:
            logger.error(f"❌ {dataset}数据集实验失败: {str(e)}")
            logger.error("🛑 根据错误处理策略，停止整个实验")
            raise
    
    def run_complete_experiment(self):
        """运行完整实验"""
        logger.info("🎯 开始完整的PQ vs RQ性能对比实验")
        logger.info(f"📋 实验规模: {len(self.datasets)}个数据集 × {len(self.k_values)}个K值 × {len(self.target_bits)}个编码长度 = {len(self.datasets) * len(self.k_values) * len(self.target_bits)}个模型")
        
        total_start_time = time.time()
        all_build_results = []
        all_search_results = []
        
        try:
            for dataset in self.datasets:
                build_results, search_results = self.run_dataset_experiment(dataset)
                all_build_results.extend(build_results)
                all_search_results.extend(search_results)
            
            # 保存最终汇总结果
            self.save_final_results(all_build_results, all_search_results)
            
            total_time = time.time() - total_start_time
            logger.info(f"🎉 完整实验成功完成! 总耗时: {total_time/3600:.2f}小时")
            
        except Exception as e:
            logger.error(f"❌ 实验失败: {str(e)}")
            logger.error("🛑 实验已停止")
            sys.exit(1)
    
    def save_final_results(self, all_build_results: List[Dict], all_search_results: List[Dict]):
        """保存最终汇总结果"""
        # 保存完整构建结果
        self.save_csv(all_build_results, "final_results/all_build_results.csv")
        
        # 保存完整搜索结果
        self.save_csv(all_search_results, "final_results/all_search_results.csv")
        
        # 生成实验摘要
        summary = self.generate_experiment_summary(all_build_results, all_search_results)
        self.save_csv([summary], "final_results/experiment_summary.csv")
        
        logger.info("💾 最终结果已保存到 final_results/")
    
    def generate_experiment_summary(self, build_results: List[Dict], search_results: List[Dict]) -> Dict[str, Any]:
        """生成实验摘要"""
        return {
            "total_models": len(build_results),
            "successful_models": len([r for r in build_results if r["status"] == "Success"]),
            "total_build_time_hours": sum(r["build_time"] for r in build_results) / 3600,
            "datasets_tested": len(self.datasets),
            "algorithms_tested": list(set(r["algorithm"] for r in build_results)),
            "k_values_tested": self.k_values,
            "target_bits_tested": self.target_bits,
            "experiment_date": time.strftime("%Y-%m-%d"),
            "experiment_duration_hours": time.time() / 3600  # 需要实际计算
        }

def main():
    """主函数"""
    print("🚀 启动自动化PQ vs RQ性能对比实验")
    print("=" * 80)
    
    runner = ExperimentRunner()
    runner.run_complete_experiment()

if __name__ == "__main__":
    main() 