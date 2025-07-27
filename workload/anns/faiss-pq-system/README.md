# FAISS Product Quantization Vector Retrieval System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FAISS](https://img.shields.io/badge/FAISS-1.7.1+-green.svg)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

High-performance product quantization (PQ) vector retrieval system based on FAISS, supporting large-scale vector similarity search, automatic dataset download, and comprehensive performance evaluation. **Zero-configuration GitHub deployment!**

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify installation
python main.py --list-datasets

# 3. Train models (recommend GloVe first, faster download)
python main.py --mode build --dataset glove --k_values 16,64 --m_values 5,10

# 4. Performance evaluation
python main.py --mode search --dataset glove --topk 100 --recall_k 1,10,100
```

## 📖 Experiment Overview

This is a comprehensive **Product Quantization (PQ) vs Residual Quantization (RQ) performance comparison study** based on FAISS library, comparing the performance of PQ and RQ quantization algorithms across different datasets and parameter configurations.

### 🎯 Research Objectives

1. **Algorithm Performance Comparison**: Compare PQ vs RQ algorithms in terms of retrieval accuracy (Recall) and search speed
2. **Parameter Optimization Study**: Find optimal K and M value combinations for different datasets
3. **Encoding Efficiency Analysis**: Compare algorithm performance under same encoding length constraints
4. **Practical Application Guidance**: Provide data-driven recommendations for algorithm and parameter selection

## 📊 Supported Datasets

| Dataset | Dimension | Vectors | Download Size | Use Case |
|---------|-----------|---------|---------------|----------|
| **GloVe** | 100 | 1.18M | ~463MB | Word embeddings |
| **SIFT1M** | 128 | 1M | ~500MB | Image features |
| **GIST1M** | 960 | 1M | ~3.6GB | Image descriptors |
| **Deep10M** | 96 | 10M | ~3.6GB | Deep learning features |

## 🎯 Core Algorithms

### Algorithm Types

| Algorithm | Type | Features | Use Case |
|-----------|------|----------|----------|
| **PQ** | Product Quantization | Standard 8-bit quantization, high accuracy | Small to medium datasets |
| **IVF-PQ** | Inverted File + PQ | Inverted index accelerates large-scale search | Large datasets |
| **4BitFastScan** | 4-bit optimized PQ | SIMD acceleration, extreme speed | Speed-critical scenarios |
| **RQ** | Residual Quantization | Residual quantization, higher accuracy | High accuracy requirements |
| **RQ FastScan** | 4-bit optimized RQ | Balance of speed and accuracy | Balanced applications |

### Algorithm Usage Examples

```bash
# PQ algorithm
python main.py --mode build --dataset sift1m --algorithm pq --k_values 16,64 --m_values 8,16

# IVF-PQ algorithm
python main.py --mode build --dataset deep10m --algorithm ivf-pq --k_values 64 --m_values 8 --n_clusters 1024

# 4BitFastScan algorithm (4-bit optimized)
python main.py --mode build --dataset glove --algorithm 4bitfastscan --k_values 16 --m_values 5,10

# RQ algorithm (K values must be powers of 2)
python main.py --mode build --dataset sift1m --algorithm rq --k_values 16,32,64 --m_values 4,8

# RQ FastScan algorithm (optimized for speed)
python main.py --mode build --dataset glove --algorithm rq_fastscan --k_values 16 --m_values 4,8
```

## 🧪 Experimental Design

### Parameter Design Strategy

The experiment uses a **target encoding length driven** parameter design approach:

#### Core Parameters
- **K values**: [4, 8, 16, 32, 64, 128, 256] - Number of cluster centers per subspace
- **M values**: Intelligently selected based on dataset dimension and target encoding length
- **Target encoding lengths**: [32, 64, 128] bits

#### Intelligent M Value Selection

```
Encoding Length = M × log₂(K)
Ideal M = Target Encoding Length / log₂(K)
```

**For PQ algorithms** (GloVe, SIFT1M): M must be a factor of dataset dimension
**For RQ algorithms** (GIST1M, Deep10M): M can be chosen arbitrarily for greater flexibility

### Experiment Scale

- **Total datasets**: 4
- **Algorithm types**: 5
- **K value range**: 7 different values
- **Encoding length targets**: 3 levels
- **Expected models**: 84 training models
- **Search tests**: Complete evaluation for each dataset

## 📈 Typical Performance

**SIFT1M Dataset Results:**
- PQ K=64, M=16: Recall@10=0.85, Search time=2.9s
- PQ K=256, M=32: Recall@10=0.98, Search time=6.0s
- **4BitFastScan K=16, M=16: Recall@10=0.83, Search time=0.8s** (🚀 4x faster)
- **RQ K=64, M=8: Recall@10=0.89, Search time=3.2s** (🎯 Higher accuracy)
- **RQ FastScan K=16, M=8: Recall@10=0.87, Search time=1.1s** (⚡ Speed + Accuracy)

## 📁 Auto-created Structure

```
faiss-pq-system/
├── 📋 Experiment Framework
│   ├── experiment_runner.py      # One-click automated experiment execution
│   ├── main.py                   # Core program entry
│   └── configs/                  # Configuration files
├── 🧠 Core Algorithms
│   ├── core/data_manager.py      # Dataset management
│   ├── core/pq_builder.py        # Model training (PQ, RQ, FastScan support)
│   ├── core/pq_searcher.py       # Search engine
│   └── core/evaluator.py         # Performance evaluation
├── 📊 Experiment Results
│   ├── models/                   # Trained models
│   ├── results/                  # Evaluation results
│   └── final_results/            # Final summary
└── 📝 Documentation
    └── README.md                 # This comprehensive guide
```

## 🔍 Reconstruction Error Analysis

**New Feature!** The system supports **Reconstruction Error** analysis, measuring quantization quality by comparing original vectors with reconstructed versions.

### 📈 Supported Metrics

- **MSE (Mean Squared Error)**: Average squared difference between original and reconstructed vectors
- **MAE (Mean Absolute Error)**: Average absolute difference 
- **Max Error**: Maximum absolute difference across all dimensions

### 🛠️ Algorithm Support Status

| Algorithm | Support Level | Notes |
|-----------|---------------|-------|
| **PQ** | ✅ Full Support | Standard 8-bit PQ, most reliable reconstruction |
| **IVF-PQ** | ✅ Full Support | Encodes residual vectors, accurate reconstruction |
| **4BitFastScan** | ⚠️ Limited Support | 4-bit quantization, approximate values |
| **RQ** | ✅ Full Support | Residual quantization, excellent reconstruction |
| **RQ FastScan** | ⚠️ Limited Support | 4-bit optimization, reconstruction limited |

### 💡 Usage Example

```python
from core.evaluator import Evaluator
from core.pq_searcher import PQSearcher

# Load models and get model info
searcher = PQSearcher(models_dir="./models")
models_info = searcher.get_models_info(["model1", "model2"])

# Evaluate with reconstruction error
evaluator = Evaluator(results_dir="./results")
results = evaluator.evaluate_with_reconstruction_error(
    search_results=search_results,
    models_info=models_info,
    query_vectors=query_vectors,
    groundtruth=groundtruth
)
```

## 📚 Complete API Reference

### 🗂️ Core Classes

#### DataManager - Dataset Management

```python
from core import DataManager

# Initialize
data_manager = DataManager(config_path="configs/dataset_configs.yaml")

# Key methods
exists = data_manager.check_dataset_exists("sift1m")
success = data_manager.download_dataset("sift1m", force_download=False)
base_vectors, query_vectors, learn_vectors, groundtruth = data_manager.load_dataset("sift1m", "l2")
dataset_info = data_manager.get_dataset_info("sift1m")
stats = data_manager.get_dataset_stats("sift1m")
datasets = data_manager.list_available_datasets()
```

#### PQBuilder - Model Training

```python
from core import PQBuilder

# Initialize
builder = PQBuilder(models_dir="./models")

# Build models
results = builder.build_models(
    base_vectors=base_vectors,
    learn_vectors=learn_vectors,
    dataset_name="sift1m",
    algorithm="pq",           # pq, ivf-pq, 4bitfastscan, rq, rq_fastscan
    k_values=[16, 64, 256],   # Cluster center counts
    m_values=[8, 16],         # Subvector counts
    dist_metric="l2",         # l2, ip, angular, cosine
    n_clusters=256            # IVF clusters (IVF-PQ only)
)

# List and manage models
models = builder.list_built_models()
model_info = builder.get_model_info("sift1m_pq_k64_m8_l2")
summary = builder.get_build_summary(results)
```

#### PQSearcher - Vector Search

```python
from core import PQSearcher

# Initialize
searcher = PQSearcher(models_dir="./models")

# Search operations
success = searcher.load_model("sift1m_pq_k64_m8_l2")
result = searcher.search_model(
    model_name="sift1m_pq_k64_m8_l2",
    query_vectors=query_vectors,
    topk=100,
    rerank_k=None
)

# Model management
model_names = searcher.get_model_list()
loaded_models = searcher.get_loaded_models()
models_info = searcher.get_models_info(["model1", "model2"])
searcher.clear_cache()
```

#### Evaluator - Performance Evaluation

```python
from core import Evaluator

# Initialize
evaluator = Evaluator(results_dir="./results")

# Evaluation methods
results = evaluator.evaluate_search_results(
    search_results=search_results,
    groundtruth=groundtruth,
    recall_k=[1, 10, 100]
)

# With reconstruction error
results = evaluator.evaluate_with_reconstruction_error(
    search_results=search_results,
    models_info=models_info,
    query_vectors=query_vectors,
    groundtruth=groundtruth
)

# Save and analyze results
output_paths = evaluator.save_evaluation_results(results, "sift1m")
best_metrics = evaluator.get_best_metrics(results)
```

## 🚀 Complete Experiment Execution

### Automated Experiment

```bash
# Run complete automated experiment (84 models)
python experiment_runner.py
```

### Manual Step-by-Step

```bash
# Step 1: Check available datasets
python main.py --list-datasets

# Step 2: Download dataset (recommend starting with GloVe)
python main.py --download-dataset glove

# Step 3: Build models
python main.py --mode build --dataset glove --algorithm pq --k_values 16,64 --m_values 5,10

# Step 4: Performance evaluation
python main.py --mode search --dataset glove --topk 100 --recall_k 1,10,100
```

### Algorithm-Specific Testing

```bash
# Test 4BitFastScan
python main.py --mode build --dataset sift1m --algorithm 4bitfastscan --k_values 16 --m_values 8,16

# Test RQ algorithm
python main.py --mode build --dataset gist1m --algorithm rq --k_values 16,32,64 --m_values 4,8

# Large-scale IVF-PQ
python main.py --mode build --dataset deep10m --algorithm ivf-pq --k_values 64 --m_values 8 --n_clusters 1024
```

## 📊 Performance Optimization

### Algorithm Selection Guide

| Scenario | Recommended Algorithm | Parameter Suggestions |
|----------|----------------------|----------------------|
| **Small-scale High Accuracy** | `pq` | K=256, M=8-16 |
| **Large-scale Data** | `ivf-pq` | K=64, M=8, clusters=1024+ |
| **Extreme Speed** | `4bitfastscan` | K=16, M=8-32 |
| **High Accuracy Requirements** | `rq` | K=64, M=4-8 |
| **Speed-Accuracy Balance** | `rq_fastscan` | K=16-64, M=4-8 |

### Memory Optimization

```python
# Batch processing
data_manager = DataManager()
data_manager.batch_size = 10000  # Adjust batch size

# Model cache management
searcher = PQSearcher()
searcher.unload_model("unused_model")  # Release unused models
searcher.clear_cache()  # Clear all cache
```

## ⏰ Experiment Time Estimation

### Training Time
- **Small datasets** (GloVe): 5-15 minutes/model
- **Medium datasets** (SIFT1M): 10-30 minutes/model  
- **Large datasets** (GIST1M, Deep10M): 30-60 minutes/model

### Total Time Estimation
- **Single dataset complete experiment**: 2-8 hours
- **All 4 datasets**: 12-30 hours
- **Recommended execution**: Step-by-step by dataset

## 🎯 Research Value

### Academic Value
1. **Algorithm Comparison**: First comprehensive PQ vs RQ algorithm comparison
2. **Parameter Optimization**: Optimal parameter configurations for different scenarios
3. **Performance Benchmarks**: Standardized vector retrieval performance benchmarks

### Practical Applications
1. **Recommendation Systems**: Efficient retrieval of user/item vectors
2. **Image Search**: Similarity search for image features
3. **Natural Language Processing**: Semantic retrieval for word/sentence vectors
4. **Anomaly Detection**: Anomaly pattern recognition in high-dimensional feature spaces

## 🔧 Troubleshooting

### Common Issues
1. **Out of Memory**: Adjust `max_train_samples` parameter or use smaller datasets
2. **Download Failures**: Check network connection, can manually download dataset files
3. **Build Failures**: Check if parameters meet dimension constraints (PQ algorithms)

### Technical Support
- Detailed error logs recorded in `experiment.log`
- Each step has detailed progress output
- Support for resuming experiments from interruption points

## 🚀 Getting Started

Ready to start this exciting vector quantization algorithm comparison experiment?

```bash
# Step 1: Verify environment
python main.py --list-datasets

# Step 2: Start first experiment
python main.py --mode build --dataset glove --algorithm pq --k_values 16,64 --m_values 5,10

# Step 3: View results
python main.py --mode search --dataset glove --topk 100 --recall_k 1,10,100
```

## 📋 Dependencies

### Python Requirements
```
faiss-cpu>=1.7.0
numpy>=1.21.0
pandas>=1.3.0
PyYAML>=6.0
scikit-learn>=1.0.0
tqdm>=4.62.0
requests>=2.26.0
h5py>=3.6.0
matplotlib>=3.5.0
seaborn>=0.11.0
psutil>=5.8.0
```

### Conda Environment
```bash
conda env create -f environment.yml
conda activate faiss-pq
```

---

🎉 **Happy experimenting! Looking forward to discovering interesting experimental results!**
