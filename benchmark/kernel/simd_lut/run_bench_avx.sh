#!/bin/bash

# Single-core execution script for AVX512 LUT benchmark
# This script ensures the program runs on a single CPU core

echo "=== Single-Core AVX512 LUT Benchmark ==="
# Get CPU information
echo "CPU Information:"
echo "Number of cores: $(nproc)"
echo "CPU model: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
echo "AVX256 support: $(grep -o 'avx2' /proc/cpuinfo | head -1 || echo 'Not available')"
echo "AVX512 support: $(grep -o 'avx512' /proc/cpuinfo | head -1 || echo 'Not available')"
# Get CPU maximum frequency (in MHz or GHz)
if [ -f /proc/cpuinfo ]; then
    max_freq=$(awk -F': ' '/cpu MHz/ {if($2>max) max=$2} END{if(max) print max " MHz"; else print "N/A"}' /proc/cpuinfo)
    echo "Max CPU frequency: $max_freq"
else
    echo "Max CPU frequency: N/A"
fi
echo ""

# Compile the program
mkdir -p build

# clang++ -O3 -funroll-loops -march=native -ffast-math \
clang++ -O3 \
    -D_SCALAR_ \
    -o build/bench_scalar_lut_single_core \
    bench_avx_lut.cpp

# clang++ -O3 -funroll-loops -march=native -ffast-math \
clang++ -O3 \
    -mavx2 \
    -D_AVX256_EPI8_ \
    -o build/bench_avx256_epi8_lut_single_core \
    bench_avx_lut.cpp

# clang++ -O3 -funroll-loops -march=native -ffast-math \
clang++ -O3 \
    -mavx2 -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vbmi \
    -D_AVX512_EPI8_ \
    -o build/bench_avx512_epi8_lut_single_core \
    bench_avx_lut.cpp

# clang++ -O3 -funroll-loops -march=native -ffast-math \
clang++ -O3 \
    -mavx2 -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vbmi \
    -D_AVX512_EPI16_ \
    -o build/bench_avx512_epi16_lut_single_core \
    bench_avx_lut.cpp

echo ""
echo "=== Running benchmark on single core (CPU 0) ==="
echo ""

# Using taskset (most reliable)
# Run the benchmark and capture output
output_scalar=$(taskset -c 0 ./build/bench_scalar_lut_single_core 2>&1)
exit_code=$?

echo "Scalar Performance @ $max_freq:"
echo "$output_scalar"
echo ""

output_256=$(taskset -c 0 ./build/bench_avx256_epi8_lut_single_core 2>&1)
exit_code=$?

echo "AVX256 Performance @ $max_freq:"
echo "$output_256"
echo ""

output_512=$(taskset -c 0 ./build/bench_avx512_epi8_lut_single_core 2>&1)
exit_code=$?

echo "AVX512 Performance @ $max_freq:"
echo "$output_512"
echo ""

output_512_epi16=$(taskset -c 0 ./build/bench_avx512_epi16_lut_single_core 2>&1)
exit_code=$?

echo "AVX512 EPI16 Performance @ $max_freq:"
echo "$output_512_epi16"
echo ""

# Check if execution was successful
if [ $exit_code -eq 0 ]; then
    echo "✅ Benchmark completed successfully on single core"
else
    echo "❌ Benchmark failed with exit code: $exit_code"
fi

# # Set CPU governor to performance for consistent results
# echo "Setting CPU governor to performance..."
# if command -v cpupower &> /dev/null; then
#     sudo cpupower frequency-set -g performance > /dev/null 2>&1
#     echo "CPU governor set to performance mode"
# else
#     echo "Warning: cpupower not available, using default governor"
# fi

# # Disable CPU scaling for more consistent results
# echo "Disabling CPU scaling..."
# for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
#     if [ -f "$cpu" ]; then
#         echo performance | sudo tee "$cpu" > /dev/null 2>&1
#     fi
# done
