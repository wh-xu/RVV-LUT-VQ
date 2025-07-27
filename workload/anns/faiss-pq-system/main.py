#!/usr/bin/env python3
"""
FAISS Product Quantization Main Program

Workflow:
    1. Build: python main.py --mode build --dataset glove --algorithm pq --k_values 16,64 --m_values 5,10
    2. Search: python main.py --mode search --topk 100 --log_level INFO
    3. List: python main.py --list-models
"""

import argparse
import logging
import sys
import os
from typing import List, Dict, Any, Optional

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import (
    DataManager, PQBuilder, PQSearcher, Evaluator,
    load_config, setup_logging, validate_parameters, print_system_info,
    get_dataset_distance_metric
)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="FAISS Product Quantization System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --list-datasets
    python main.py --mode build --dataset glove --k_values 16,64 --m_values 5,10
    python main.py --mode search --dataset glove --topk 100 --recall_k 1,10,100
    python main.py --list-models
        """
    )
    
    # Main parameters
    parser.add_argument("--mode", type=str, choices=["build", "search"],
                       help="Mode: build or search")
    
    # Build mode parameters
    parser.add_argument("--dataset", type=str,
                       help="Dataset name (required for build mode)")
    
    parser.add_argument("--algorithm", type=str, default="pq", choices=["pq", "ivf-pq", "4bitfastscan", "rq", "rq_fastscan"],
                       help="Algorithm type (default: pq)")
    
    parser.add_argument("--k_values", type=str, default="8,16,64,128,256",
                       help="K values, comma-separated (default: 8,16,64,128,256)")
    
    parser.add_argument("--m_values", type=str, default="8",
                       help="M values, comma-separated (default: 8)")
    
    parser.add_argument("--n_clusters", type=int, default=256,
                       help="IVF clusters (default: 256)")
    
    # Search mode parameters
    parser.add_argument("--model-name", type=str,
                       help="Model name for direct search")
    
    parser.add_argument("--topk", type=int, default=100,
                       help="Top-K neighbors (default: 100)")
    
    parser.add_argument("--rerank", type=int, default=0,
                       help="Rerank candidates (0=no rerank, >0=rerank size)")
    
    parser.add_argument("--recall_k", type=str, default="1,10,100",
                       help="Recall@K values (default: 1,10,100)")
    
    parser.add_argument("--preload", type=str, default="True",
                       help="Preload option (default: True)")
    
    parser.add_argument("--output", type=str, default=None,
                       help="Output filename")
    
    parser.add_argument("--no-reconerr", action="store_true",
                       help="Disable reconstruction error calculation (faster but less detailed)")
    
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Log level (default: INFO)")
    
    # Special operations
    parser.add_argument("--list-models", action="store_true",
                       help="List built models")
    
    parser.add_argument("--list-datasets", action="store_true",
                       help="List available datasets")
    
    parser.add_argument("--check-dataset", type=str,
                       help="Check dataset status")
    
    parser.add_argument("--download-dataset", type=str,
                       help="Download dataset")
    
    parser.add_argument("--config", type=str, default="configs/default_config.yaml",
                       help="Config file path")
    
    return parser.parse_args()


def parse_list_parameter(param_str: str) -> List[int]:
    """Parse comma-separated parameter list"""
    return [int(x.strip()) for x in param_str.split(",") if x.strip()]


def main():
    """Main function"""
    args = parse_arguments()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("FAISS Product Quantization System Starting")
    logger.info("=" * 80)
    
    print_system_info()
    
    try:
        # Special operations
        if args.list_models:
            list_built_models()
            return
        
        if args.list_datasets:
            list_available_datasets()
            return
        
        if args.check_dataset:
            check_dataset_status(args.check_dataset)
            return
        
        if args.download_dataset:
            download_dataset_manually(args.download_dataset)
            return
        
        # Validate required parameters
        if not args.mode:
            logger.error("Must specify mode --mode (build/search)")
            sys.exit(1)
        
        recall_k = parse_list_parameter(args.recall_k)
        
        # Build mode validation
        if args.mode == "build":
            if not args.dataset:
                logger.error("Build mode requires --dataset parameter")
                logger.info("Available datasets: sift1m, glove, gist1m, deep10m")
                logger.info("Example: python main.py --mode build --dataset sift1m --k_values 16,64 --m_values 8")
                sys.exit(1)
                
            k_values = parse_list_parameter(args.k_values)
            m_values = parse_list_parameter(args.m_values)
            
            # Get default distance metric from dataset config
            data_manager = DataManager()
            dataset_info = data_manager.get_dataset_info(args.dataset)
            dist_metric = get_dataset_distance_metric(args.dataset, dataset_info)
            
            # Validate parameters and get final algorithm (may be auto-corrected)
            final_algorithm = validate_parameters(args.algorithm, k_values, m_values, 
                              dist_metric, args.n_clusters)
            
            # Update algorithm if it was auto-corrected
            if final_algorithm != args.algorithm:
                logger.info(f"Algorithm updated: {args.algorithm} → {final_algorithm}")
                args.algorithm = final_algorithm
        else:
            k_values = []
            m_values = []
            dist_metric = 'l2'  # Default for search mode
        
        logger.info(f"Parameters:")
        logger.info(f"  Mode: {args.mode}")
        
        if args.mode == "build":
            logger.info(f"  Dataset: {args.dataset}")
            logger.info(f"  Algorithm: {args.algorithm}")
            logger.info(f"  K values: {k_values}")
            logger.info(f"  M values: {m_values}")
            logger.info(f"  Distance metric: {dist_metric} (auto-selected)")
            if args.algorithm == "ivf-pq":
                logger.info(f"  Clusters: {args.n_clusters}")
        else:
            if args.dataset:
                logger.info(f"  Dataset: {args.dataset}")
            logger.info(f"  TopK: {args.topk}")
            if args.rerank > 0:
                logger.info(f"  Rerank candidates: {args.rerank}")
            else:
                logger.info("  Rerank: No")
        
        # Run main logic
        if args.mode == "build":
            run_build_mode(args, k_values, m_values)
        elif args.mode == "search":
            run_auto_search_mode(args, recall_k)
        
        logger.info("Program completed!")
        
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
        sys.exit(1)


def list_built_models():
    """List built models"""
    logger = logging.getLogger(__name__)
    
    builder = PQBuilder()
    models = builder.list_built_models()
    
    if not models:
        logger.info("No built models found")
        return
    
    logger.info(f"Built models ({len(models)} total):")
    logger.info("-" * 80)
    
    for model in models:
        logger.info(f"Model: {model['model_name']}")
        logger.info(f"  Dataset: {model.get('dataset', 'Unknown')}")
        logger.info(f"  Algorithm: {model.get('algorithm', 'Unknown')}")
        logger.info(f"  Parameters: K={model.get('k', 'Unknown')}, M={model.get('m', 'Unknown')}")
        logger.info(f"  Clusters: {model.get('n_clusters', 'N/A')}")
        logger.info(f"  Distance: {model.get('dist_metric', 'Unknown')}")
        logger.info(f"  Vectors: {model.get('num_base_vectors', 'Unknown')}")
        logger.info(f"  Dimension: {model.get('dimension', 'Unknown')}")
        
        if 'index_size_mb' in model:
            logger.info(f"  Size: {model['index_size_mb']:.2f} MB")
        
        logger.info("")


def list_available_datasets():
    """List available datasets"""
    logger = logging.getLogger(__name__)
    
    data_manager = DataManager()
    datasets = data_manager.list_available_datasets()
    
    logger.info(f"Available datasets ({len(datasets)} total):")
    logger.info("-" * 80)
    
    for dataset_name in datasets:
        try:
            dataset_info = data_manager.get_dataset_info(dataset_name)
            exists = data_manager.check_dataset_exists(dataset_name)
            status = "Downloaded" if exists else "Not downloaded"
            
            logger.info(f"\nDataset: {dataset_name}")
            logger.info(f"  Name: {dataset_info.get('name', 'Unknown')}")
            logger.info(f"  Description: {dataset_info.get('description', 'No description')}")
            logger.info(f"  Dimension: {dataset_info.get('dimension', 'Unknown')}")
            logger.info(f"  Base vectors: {dataset_info.get('num_base', 'Unknown'):,}")
            logger.info(f"  Query vectors: {dataset_info.get('num_query', 'Unknown'):,}")
            logger.info(f"  Format: {dataset_info.get('params', {}).get('format', 'Unknown')}")
            logger.info(f"  Distance: {dataset_info.get('params', {}).get('distance_metric', 'Unknown')}")
            logger.info(f"  Status: {status}")
            
            if not exists:
                files_list = list(dataset_info.get('files', {}).values())
                if files_list:
                    first_file = files_list[0]
                    download_url = first_file.get('url', 'Unknown')
                    logger.info(f"  URL: {download_url}")
                else:
                    logger.info("  URL: Not configured")
            else:
                logger.info("  Status: Exists")
                stats = data_manager.get_dataset_stats(dataset_name)
                if "error" not in stats:
                    logger.info("\nDetails:")
                    logger.info(f"  Dimension: {stats.get('dimension', 'Unknown')}")
                    logger.info(f"  Base vectors: {stats.get('num_base_vectors', 'Unknown'):,}")
                    logger.info(f"  Query vectors: {stats.get('num_query_vectors', 'Unknown'):,}")
                    logger.info(f"  Learn vectors: {stats.get('num_learn_vectors', 'Unknown'):,}")
                    logger.info(f"  Data type: {stats.get('data_type', 'Unknown')}")
                    logger.info(f"  Memory usage: {stats.get('total_memory_mb', 0):.2f} MB")
                else:
                    logger.warning(f"  Failed to get stats: {stats['error']}")
        except Exception as e:
            logger.error(f"Failed to get dataset {dataset_name} info: {str(e)}")


def check_dataset_status(dataset_name: str):
    """Check dataset status"""
    logger = logging.getLogger(__name__)
    
    data_manager = DataManager()
    
    try:
        dataset_info = data_manager.get_dataset_info(dataset_name)
        exists = data_manager.check_dataset_exists(dataset_name)
        
        logger.info(f"Dataset status: {dataset_name}")
        logger.info("-" * 60)
        logger.info(f"  Name: {dataset_info.get('name', 'Unknown')}")
        logger.info(f"  Path: {dataset_info.get('local_path', 'Unknown')}")
        
        if exists:
            logger.info("  Status: Exists")
            
            stats = data_manager.get_dataset_stats(dataset_name)
            if "error" not in stats:
                logger.info("\nDetails:")
                logger.info(f"  Dimension: {stats.get('dimension', 'Unknown')}")
                logger.info(f"  Base vectors: {stats.get('num_base_vectors', 'Unknown'):,}")
                logger.info(f"  Query vectors: {stats.get('num_query_vectors', 'Unknown'):,}")
                logger.info(f"  Learn vectors: {stats.get('num_learn_vectors', 'Unknown'):,}")
                logger.info(f"  Data type: {stats.get('data_type', 'Unknown')}")
                logger.info(f"  Memory usage: {stats.get('total_memory_mb', 0):.2f} MB")
            else:
                logger.warning(f"  Failed to get stats: {stats['error']}")
        else:
            logger.info("  Status: Not exists")
            files_list = list(dataset_info.get('files', {}).values())
            if files_list:
                first_file = files_list[0]
                download_url = first_file.get('url', 'Unknown')
                logger.info(f"  URL: {download_url}")
            else:
                logger.info("  URL: Not configured")
            logger.info("\nDownload with:")
            logger.info(f"  python main.py --download-dataset {dataset_name}")
            
    except ValueError as e:
        logger.error(f"Unsupported dataset: {dataset_name}")
        logger.info("\nUse this to see available datasets:")
        logger.info("  python main.py --list-datasets")
    except Exception as e:
        logger.error(f"Failed to check dataset: {str(e)}")


def download_dataset_manually(dataset_name: str):
    """Download dataset manually"""
    logger = logging.getLogger(__name__)
    
    data_manager = DataManager()
    
    try:
        logger.info(f"Downloading dataset: {dataset_name}")
        
        success = data_manager.download_dataset(dataset_name, force_download=True)
        
        if success:
            logger.info(f"Dataset {dataset_name} downloaded successfully!")
            
            if data_manager.check_dataset_exists(dataset_name):
                logger.info("Dataset verification passed")
                
                stats = data_manager.get_dataset_stats(dataset_name)
                if "error" not in stats:
                    logger.info("\nDataset info:")
                    logger.info(f"  Dimension: {stats.get('dimension', 'Unknown')}")
                    logger.info(f"  Base vectors: {stats.get('num_base_vectors', 'Unknown'):,}")
                    logger.info(f"  Query vectors: {stats.get('num_query_vectors', 'Unknown'):,}")
                    logger.info(f"  Memory usage: {stats.get('total_memory_mb', 0):.2f} MB")
            else:
                logger.warning("Download completed but verification failed")
        else:
            logger.error(f"Dataset {dataset_name} download failed")
            
    except ValueError as e:
        logger.error(f"Unsupported dataset: {dataset_name}")
        logger.info("\nUse this to see available datasets:")
        logger.info("  python main.py --list-datasets")
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")


def run_build_mode(args: argparse.Namespace, k_values: List[int], m_values: List[int]):
    """Run build mode"""
    logger = logging.getLogger(__name__)
    logger.info("Starting build mode")
    
    # Initialize components
    data_manager = DataManager()
    builder = PQBuilder()
    
    # Get default distance metric from dataset config
    dataset_info_config = data_manager.get_dataset_info(args.dataset)
    dist_metric = get_dataset_distance_metric(args.dataset, dataset_info_config)
    
    # Load dataset
    logger.info("Loading dataset...")
    base_vectors, query_vectors, learn_vectors, groundtruth = data_manager.load_dataset(
        args.dataset, dist_metric
    )
    
    # Build models
    logger.info("Starting PQ model building...")
    build_results = builder.build_models(
        base_vectors=base_vectors,
        learn_vectors=learn_vectors,
        dataset_name=args.dataset,
        algorithm=args.algorithm,
        k_values=k_values,
        m_values=m_values,
        dist_metric=dist_metric,
        n_clusters=args.n_clusters
    )
    
    # Output build summary
    summary = builder.get_build_summary(build_results)
    logger.info("\nBuild summary:")
    logger.info(f"  Total models: {summary['total_models']}")
    logger.info(f"  Successfully built: {summary['successful_models']}")
    logger.info(f"  Already exists: {summary['existing_models']}")
    logger.info(f"  Build failed: {summary['failed_models']}")
    logger.info(f"  Total time: {summary['total_build_time']:.2f}s")
    logger.info(f"  Average time: {summary['average_build_time']:.2f}s")
    logger.info(f"  Success rate: {summary['success_rate']:.1f}%")


def run_auto_search_mode(args: argparse.Namespace, recall_k: List[int]):
    """Run auto search mode"""
    logger = logging.getLogger(__name__)
    logger.info("Starting search mode")
    
    # Initialize components
    searcher = PQSearcher()
    evaluator = Evaluator()
    builder = PQBuilder()
    data_manager = DataManager()
    
    # Check if models exist
    all_model_names = searcher.get_model_list()
    
    if not all_model_names:
        logger.error("❌ Error: No built models found!")
        logger.info("")
        logger.info("Please run build mode first:")
        logger.info("  python main.py --mode build --dataset glove --k_values 16,64,128 --m_values 5,10")
        logger.info("")
        logger.info("Or see available commands:")
        logger.info("  python main.py --help")
        sys.exit(1)
    
    # Filter models by user-specified dataset
    if args.dataset:
        dataset_name = args.dataset
        logger.info(f"🎯 User specified dataset: {dataset_name}")
        
        # Filter matching models
        model_names = []
        for model_name in all_model_names:
            model_info = builder.get_model_info(model_name)
            if model_info and model_info.get('dataset') == dataset_name:
                model_names.append(model_name)
        
        if not model_names:
            logger.error(f"❌ Error: No models found for dataset '{dataset_name}'!")
            logger.info(f"\nAvailable models:")
            for model_name in all_model_names:
                model_info = builder.get_model_info(model_name)
                model_dataset = model_info.get('dataset', 'Unknown') if model_info else 'Unknown'
                logger.info(f"  {model_name} (dataset: {model_dataset})")
            logger.info(f"\nUse one of these commands:")
            logger.info(f"  python main.py --mode search  # Search all models")
            logger.info(f"  python main.py --mode build --dataset {dataset_name} --k_values 16,64 --m_values 8")
            sys.exit(1)
            
        # Get dataset distance metric
        dataset_info = data_manager.get_dataset_info(dataset_name)
        dist_metric = dataset_info.get('params', {}).get('distance_metric', 'l2')
        
        logger.info(f"✅ Found {len(model_names)} matching models, starting search...")
        logger.info(f"📊 Target dataset: {dataset_name} (distance metric: {dist_metric})")
    else:
        # No dataset specified, use all models
        model_names = all_model_names
        logger.info(f"✅ Found {len(model_names)} models, starting search...")
        
        # Try to infer dataset from first model
        first_model_info = builder.get_model_info(model_names[0])
        if first_model_info:
            dataset_name = first_model_info.get('dataset', 'unknown')
            try:
                dataset_info = data_manager.get_dataset_info(dataset_name)
                dist_metric = dataset_info.get('params', {}).get('distance_metric', 'l2')
            except ValueError:
                # Dataset not in config, use model's distance metric or default
                dist_metric = first_model_info.get('dist_metric', 'l2')
                logger.warning(f"Dataset {dataset_name} not in config, using distance metric from model: {dist_metric}")
        else:
            dataset_name = 'unknown'
            dist_metric = 'l2'
    
    # Load dataset
    logger.info(f"Loading dataset: {dataset_name}")
    try:
        base_vectors, query_vectors, learn_vectors, groundtruth = data_manager.load_dataset(
            dataset_name, dist_metric
        )
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {str(e)}")
        sys.exit(1)
    
    logger.info(f"Query vectors: {len(query_vectors)} vectors")
    logger.info(f"Search parameters: TopK={args.topk}, Rerank={'Yes' if args.rerank > 0 else 'No'}")
    
    # Run search
    logger.info("Executing search...")
    search_results = {}
    
    for model_name in model_names:
        try:
            # Search
            result = searcher.search_model(
                model_name=model_name,
                query_vectors=query_vectors,
                topk=args.topk,
                rerank_k=args.rerank if args.rerank > 0 else None
            )
            search_results[model_name] = result
        except Exception as e:
            logger.error(f"Search failed for model {model_name}: {str(e)}")
            continue
    
    if not search_results:
        logger.error("❌ All searches failed!")
        sys.exit(1)
    
    # Evaluate results
    logger.info("Evaluating results...")
    
    if not args.no_reconerr:
        logger.info("Computing reconstruction error (this may take a while)...")
        # Get models info for reconstruction error calculation
        models_info = searcher.get_models_info(list(search_results.keys()))
        
        evaluation_results = evaluator.evaluate_with_reconstruction_error(
            search_results=search_results,
            models_info=models_info,
            query_vectors=query_vectors,
            groundtruth=groundtruth,
            recall_k=recall_k
        )
    else:
        logger.info("Reconstruction error calculation disabled for faster evaluation")
        evaluation_results = evaluator.evaluate_search_results(
            search_results=search_results,
            groundtruth=groundtruth,
            recall_k=recall_k
        )
    
    # Save results
    output_dir = evaluator.save_evaluation_results(
        evaluation_results=evaluation_results,
        dataset_name=dataset_name,
        output_filename=args.output
    )
    
    # Display results
    logger.info(f"\n🎯 Search results:")
    logger.info("=" * 80)
    
    for model_name, metrics in evaluation_results.items():
        logger.info(f"\n📊 {model_name}")
        for k in recall_k:
            if f'recall@{k}' in metrics:
                logger.info(f"   recall@{k}: {metrics[f'recall@{k}']:.4f}")
        if 'search_time' in metrics:
            logger.info(f"   search_time: {metrics['search_time']:.2f}s")
        if 'qps' in metrics:
            logger.info(f"   QPS: {metrics['qps']:.2f}")
    
    # Show best performance
    best_metrics = evaluator.get_best_metrics(evaluation_results)
    logger.info(f"\n🏆 Best performance:")
    for metric_name, (model_name, value) in best_metrics.items():
        logger.info(f"  {metric_name}: {model_name} ({value:.4f})")
    
    logger.info(f"\n✅ Search completed!")
    logger.info("Results saved:")
    logger.info(f"  📄 CSV: {output_dir['csv_path']}")
    logger.info(f"  📋 Report: {output_dir['report_path']}")


if __name__ == "__main__":
    main() 