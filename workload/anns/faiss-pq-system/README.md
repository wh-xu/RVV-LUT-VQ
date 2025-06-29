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

## 📊 Supported Datasets

| Dataset | Dimension | Vectors | Download Size | Use Case |
|---------|-----------|---------|---------------|----------|
| **GloVe** | 100 | 1.18M | ~463MB | Word embeddings |
| **SIFT1M** | 128 | 1M | ~500MB | Image features |
| **GIST1M** | 960 | 1M | ~3.6GB | Image descriptors |
| **Deep10M** | 96 | 10M | ~3.6GB | Deep learning features |

## 🎯 Core Algorithms

- **PQ**: Vector partitioning quantization, memory efficient
- **IVF-PQ**: Hierarchical quantization for large-scale data

```bash
# PQ algorithm
python main.py --mode build --dataset sift1m --algorithm pq --k_values 16,64 --m_values 8,16

# IVF-PQ algorithm
python main.py --mode build --dataset deep10m --algorithm ivf-pq --k_values 64 --m_values 8 --n_clusters 1024
```

## 📈 Typical Performance

**SIFT1M Dataset Results:**
- PQ K=64, M=16: Recall@10=0.85, Search time=2.9s
- PQ K=256, M=32: Recall@10=0.98, Search time=6.0s

## 📁 Auto-created Structure

```
workload/anns/
├── Dataset/     # 🔄 Auto-download data
├── models/      # 🔄 Auto-save models
├── results/     # 🔄 Auto-save results
└── main.py      # Main program
```

## 🛠️ System Requirements

- **Python**: 3.8+
- **Memory**: 8GB+
- **Storage**: 10GB+
- **Key Dependencies**: `faiss-cpu`, `numpy`, `pandas`, `h5py`

## 🐛 Common Issues

```bash
# FAISS installation issue
pip install faiss-cpu>=1.7.0

# Get help
python main.py --help
python main.py --list-datasets
python main.py --list-models
```

## 📜 License

This project is licensed under [Apache 2.0](LICENSE).

---

**🎯 Perfect for RVV (RISC-V Vector) optimization research and ANN benchmarking!** 