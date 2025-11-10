#!/bin/bash

# Single-core execution script for Bit-serial LUT-based GEMM benchmark
# This script ensures the program runs on a single CPU core

echo "=== Single-Core LUT-based GEMM Benchmark ==="
# Get CPU information
echo "CPU Information:"
echo "Number of cores: $(nproc)"
echo "CPU model: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
echo "AVX256 support: $(grep -o 'avx2' /proc/cpuinfo | head -1 || echo 'Not available')"
echo "AVX512 support: $(grep -o 'avx512' /proc/cpuinfo | head -1 || echo 'Not available')"
# echo "NEON SIMD support: $(grep -o 'neon' /proc/cpuinfo | head -1 || echo 'Not available')"

# Get CPU maximum frequency (in MHz or GHz)
max_freq=$(lscpu | grep "CPU max MHz" | awk -F: '{print $2}' | xargs)
echo "Max CPU frequency: $max_freq"

# Compile the program
mkdir -p build

clang++ -O3 \
    -mavx2 -mfma \
    -D_SCALAR_ \
    -o build/bench_lut_gemm_single_core \
    test_tbl.cc


echo ""
echo "=== Running benchmark on single core (CPU 0) ==="
echo ""

# Using taskset (most reliable)
# Run the benchmark and capture output
output_lut_gemm=$(taskset -c 0 ./build/bench_lut_gemm_single_core 2>&1)
exit_code=$?

# echo "Scalar Performance @ $max_freq:"
# echo "$output_scalar"
# echo ""


# # Check if execution was successful
# if [ $exit_code -eq 0 ]; then
#     echo "✅ Benchmark completed successfully on single core"
# else
#     echo "❌ Benchmark failed with exit code: $exit_code"
# fi
