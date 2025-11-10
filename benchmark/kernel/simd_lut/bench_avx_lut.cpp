#include <immintrin.h>
#include <iostream>
#include <random>
#include <iomanip>
#include <chrono>
#include <thread>
#include <sched.h>

// #define USE_AVX512
// #define USE_AVX256

#ifdef _AVX512_EPI8_
#define CB_SIZE 64
#define BIT_PACK_SIZE 2
#elif defined(_AVX512_EPI16_)
#define CB_SIZE 32
#define BIT_PACK_SIZE 3
#elif defined(_AVX256_EPI8_)
#define CB_SIZE 32
#define BIT_PACK_SIZE 2
#elif defined(_SCALAR_)
#define CB_SIZE 16
#define BIT_PACK_SIZE 1
#endif

inline void lookup_and_add_simd(
    #ifdef _AVX512_EPI16_
    uint16_t * idx,
    int16_t * lut,
    #else
    uint8_t * idx,
    int8_t * lut,
    #endif
    volatile int16_t * output,
    uint64_t n_runs,
    uint64_t n_lut_per_run
) {
    // Load lut into SIMD registers
    #ifdef _AVX256_EPI8_
    __m256i idx_vec;
    __m256i lut_vec = _mm256_loadu_si256((__m256i*)lut);
    __m256i psum = _mm256_setzero_si256();
    const __m256i low_mask_avx = _mm256_set1_epi8(0x0f);
    #elif defined(_AVX512_EPI8_) || defined(_AVX512_EPI16_)
    __m512i idx_vec;
    __m512i lut_vec = _mm512_loadu_si512(lut);
    __m512i psum = _mm512_setzero_si512();
    const __m512i low_mask_avx = _mm512_set1_epi8(0x0f);
    #endif

    for (int i = 0; i < n_runs; i++) {
        for (int j = 0; j < n_lut_per_run; j++) {
            // Perform SIMD LUT and addition
            // From Quicker ADC
            #ifdef _AVX512_EPI8_
            
            // Lookup add (low)
            __m512i comps = _mm512_loadu_si512((__m512i*)(idx+j*CB_SIZE));
            __m512i masked = _mm512_and_si512(comps, low_mask_avx);
            __m512i partiala = _mm512_shuffle_epi8(lut_vec, masked);

            // Lookup add (high)
            __m512i compsb = _mm512_srli_epi64(comps, 4);
            __m512i maskedb = _mm512_and_si512(compsb, low_mask_avx);
            __m512i partialb = _mm512_shuffle_epi8(lut_vec, maskedb);
            __m512i partial_sum = _mm512_adds_epu8(partiala, partialb);
            psum = _mm512_adds_epu8(psum, partial_sum);

            #elif defined(_AVX512_EPI16_)

            __m512i partiala, partialb, partialc;
            __m512i comps_012 = _mm512_loadu_si512((__m512i*)(idx+j*CB_SIZE));
            partiala = _mm512_permutex2var_epi16(lut_vec, comps_012, lut_vec);
            psum = _mm512_adds_epi16(psum, partiala);

            __m512i comps_12 = _mm512_srli_epi16(comps_012, 5);
            partialb = _mm512_permutex2var_epi16(lut_vec, comps_12, lut_vec);
            psum = _mm512_adds_epi16(psum, partialb);
            
            __m512i comps_2 = _mm512_srli_epi16(comps_012, 10);
            partialc = _mm512_permutex2var_epi16(lut_vec, comps_2, lut_vec);
            psum = _mm512_adds_epi16(psum, partialc);

            #elif defined(_AVX256_EPI8_)

            // Lookup add (low)
            __m256i comps = _mm256_loadu_si256((__m256i*)(idx+j*CB_SIZE));
            __m256i masked = _mm256_and_si256(comps, low_mask_avx);
            __m256i partiala = _mm256_shuffle_epi8(lut_vec, masked);

            // Lookup add (high)
            __m256i compsb = _mm256_srli_epi64(comps, 4);
            __m256i maskedb = _mm256_and_si256(compsb, low_mask_avx);
            __m256i partialb = _mm256_shuffle_epi8(lut_vec, maskedb);
            __m256i partial_sum = _mm256_adds_epi8(partiala, partialb);
            psum = _mm256_adds_epi8(psum, partial_sum);

            #elif defined(_SCALAR_)

            // Perform lookup operations using L1 cache
            // Prefetch next cache line to L1 cache
            _mm_prefetch(&idx[(j)*CB_SIZE], _MM_HINT_T0);
            for (int k = 0; k < CB_SIZE; k++) {
                // Process 8 elements at a time for better cache utilization
                output[k] += lut[idx[j*CB_SIZE+k]];
            }
            #endif
        }
    }

    // Reduce
    #if defined(_AVX512_EPI8_) || defined(_AVX512_EPI16_)
    __m128i sum_a = _mm512_extracti32x4_epi32(psum,0);
    __m128i sum_b = _mm_subs_epu8(_mm512_extracti32x4_epi32(psum,1), _mm_set1_epi8(1));
    __m128i sum_c = _mm_subs_epu8(_mm512_extracti32x4_epi32(psum,2), _mm_set1_epi8(1));
    __m128i sum_d = _mm_subs_epu8(_mm512_extracti32x4_epi32(psum,3), _mm_set1_epi8(1));
    __m128i candidates =  _mm_adds_epu8(_mm_adds_epu8(sum_a,sum_b), _mm_adds_epu8(sum_c,sum_d));

    // Extract values from 'candidates' and assign to 'output'
    alignas(16) uint8_t tmp[16];
    _mm_store_si128((__m128i*)tmp, candidates);
    for (int i = 0; i < 16; ++i) {
        output[i] = tmp[i];
    }
    #elif defined(_AVX256_EPI8_)
    __m128i sum_a = _mm256_extracti128_si256(psum,0);
    __m128i sum_b = _mm256_extracti128_si256(psum,1);
    __m128i sum_b_minus1 = _mm_subs_epu8(sum_b,_mm_set1_epi8(1));
    __m128i candidates = _mm_adds_epu8(sum_a, sum_b_minus1);

     // Extract values from 'candidates' and assign to 'output'
    alignas(16) uint8_t tmp[16];
    _mm_store_si128((__m128i*)tmp, candidates);
    for (int i = 0; i < 16; ++i) {
        output[i] = tmp[i];
    }
    #endif
}


int main() {
    // Force single-core execution
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);  // Pin to CPU 0
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
    
    // Create random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    // std::uniform_int_distribution<uint16_t> dist(0, (1<<15)-1); // 4-bit indices (0-15)
    std::uniform_int_distribution<uint16_t> dist(0, CB_SIZE-1); // 4-bit indices (0-15)
    
    // Create random indices
    const uint64_t n_runs = 100000;
    const uint64_t n_lut_per_run = 10000;
    #if defined(_SCALAR_) || defined(_AVX512_EPI8_)
    alignas(64) uint8_t idx[CB_SIZE*n_lut_per_run]; 
    alignas(64) int8_t lut[CB_SIZE];
    #elif defined(_AVX256_EPI8_)
    alignas(32) uint8_t idx[CB_SIZE*n_lut_per_run]; 
    alignas(32) int8_t lut[CB_SIZE];
    #elif defined(_AVX512_EPI16_)
    alignas(32) uint16_t idx[CB_SIZE*n_lut_per_run]; 
    alignas(32) int16_t lut[CB_SIZE];
    #endif
    for (int i = 0; i < CB_SIZE*n_lut_per_run; i++) {
        idx[i] = dist(gen);
    }

    // Create lookup table (LUT) - 16 entries for 4-bit indices
    volatile int16_t output[CB_SIZE];
    for (int i = 0; i < CB_SIZE; i++) {
        lut[i] = i * 10 + 100; // Example: values 100, 110, 120, ..., 250
        output[i] = 0;
    }

    // Perform lookup benchmark
    auto start = std::chrono::high_resolution_clock::now();
    lookup_and_add_simd(idx, lut, output, n_runs, n_lut_per_run);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    double ops = CB_SIZE * BIT_PACK_SIZE * n_runs * n_lut_per_run / 1e9; // 2 for LUT and Add, 4 for bit packing
    double throughput_overall = ops * 2.0 / (duration.count() * 1e-9); // elements per second
    double throughput_lut = ops / (duration.count() * 1e-9); // elements per second
    
    std::cout << "Runtime measurement (Single-Core Execution):" << std::endl;
    std::cout << "Duration: " << duration.count()*1e-6 << " ms" << std::endl;
    std::cout << "LUT Throughput: " << throughput_lut << " GLUTs/sec" << std::endl;
    
    // Verify we're running on the correct core
    int current_cpu = sched_getcpu();
    std::cout << "Execution CPU core: " << current_cpu << std::endl;
    
    return 0;
}
