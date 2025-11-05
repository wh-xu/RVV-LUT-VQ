/**
 * Copyright (c) 2018-present, Thomson Licensing, SAS.
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * Modifications related the introduction of Quicker ADC (Vectorized Product Quantization)
 * are licensed under the Clear BSD license found in the LICENSE file in the root directory
 * of this source tree.
 *
 * The rest of the source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree
 */

// -*- c++ -*-


#ifndef FAISS_VEC_PRODUCT_QUANTIZER_H
#define FAISS_VEC_PRODUCT_QUANTIZER_H

#include <stdint.h>

#include <vector>

#include "Clustering.h"
#include "Heap.h"

#include <cstddef>
#include <cstring>
#include <cstdio>
#include <algorithm>

#include "FaissAssert.h"
#include "VectorTransform.h"
#include "IndexFlat.h"
#include "utils.h"

#include <immintrin.h>
#include <x86intrin.h>
#include <iostream>
#include <boost/align/aligned_allocator.hpp>
#include <mkl.h>
#include <iomanip>

#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)

#define FORCE_INLINE __attribute__((always_inline))

extern "C" {

// this is to keep the clang syntax checker happy
#ifndef FINTEGER
#define FINTEGER int
#endif

#ifdef ECLIPSE
#define __AVX512F__
#endif


/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

int sgemm_ (
        const char *transa, const char *transb, FINTEGER *m, FINTEGER *
        n, FINTEGER *k, const float *alpha, const float *a,
        FINTEGER *lda, const float *b,
        FINTEGER *ldb, float *beta,
        float *c, FINTEGER *ldc);


}


namespace faiss {





static void init_hypercube (int d, int nbits,
                            int n, const float * x,
                            float *centroids)
{

    std::vector<float> mean (d);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d; j++)
            mean [j] += x[i * d + j];

    float maxm = 0;
    for (int j = 0; j < d; j++) {
        mean [j] /= n;
        if (fabs(mean[j]) > maxm) maxm = fabs(mean[j]);
    }

    for (int i = 0; i < (1 << nbits); i++) {
        float * cent = centroids + i * d;
        for (int j = 0; j < nbits; j++)
            cent[j] = mean [j] + (((i >> j) & 1) ? 1 : -1) * maxm;
        for (int j = nbits; j < d; j++)
            cent[j] = mean [j];
    }


}

static void init_hypercube_pca (int d, int nbits,
                                int n, const float * x,
                                float *centroids)
{
    PCAMatrix pca (d, nbits);
    pca.train (n, x);


    for (int i = 0; i < (1 << nbits); i++) {
        float * cent = centroids + i * d;
        for (int j = 0; j < d; j++) {
            cent[j] = pca.mean[j];
            float f = 1.0;
            for (int k = 0; k < nbits; k++)
                cent[j] += f *
                    sqrt (pca.eigenvalues [k]) *
                    (((i >> k) & 1) ? 1 : -1) *
                    pca.PCAMat [j + k * d];
        }
    }

}

/*
 *  Epu16 "extended" intrisics
 *  Public domain from http://www.alfredklomp.com/programming/sse-intrinsics/
 */
static inline __m128i _mm_cmple_epu8 (__m128i x, __m128i y) {
	// Returns 0xFF where x <= y:
	return _mm_cmpeq_epi8(_mm_min_epu8(x, y), x);
}

static inline __m128i _mm_cmpge_epu8 (__m128i x, __m128i y) {
	// Returns 0xFF where x >= y:
	return _mm_cmple_epu8(y, x);
}

static inline __m128i _mm_cmpgt_epu8 (__m128i x, __m128i y) {
	// Returns 0xFF where x > y:
	return _mm_andnot_si128(
		_mm_cmpeq_epi8(x, y),
		_mm_cmpeq_epi8(_mm_max_epu8(x, y), x)
	);
}

static inline __m128i _mm_cmplt_epu8 (__m128i x, __m128i y) {
	// Returns 0xFF where x < y:
	return _mm_cmpgt_epu8(y, x);
}

static inline __m128i _mm_cmple_epu16 (__m128i x, __m128i y) {
	// Returns 0xFFFF where x <= y:
	return _mm_cmpeq_epi16(_mm_subs_epu16(x, y), _mm_setzero_si128());
}

static inline __m128i _mm_cmpge_epu16 (__m128i x, __m128i y) {
	// Returns 0xFFFF where x >= y:
	return _mm_cmple_epu16(y, x);
}

static inline __m128i _mm_cmpgt_epu16 (__m128i x, __m128i y) {
	// Returns 0xFFFF where x > y:
	return _mm_andnot_si128(_mm_cmpeq_epi16(x, y), _mm_cmple_epu16(y, x));
}

static inline __m128i _mm_cmplt_epu16 (__m128i x, __m128i y) {
	// Returns 0xFFFF where x < y:
	return _mm_cmpgt_epu16(y, x);
}
/*
 * End of EPU16 extended intrisics
 */



template<class T_TSC>
struct QuantizerMAX {
    std::vector<float> min;
    float min_sum;
    float max;
    int M;
    float delta;
    float inv_delta;
    //float gmin;
    //float gmin_sum;
    T_TSC QMAX;


    QuantizerMAX(float *min_,float min_sum_, float max_, int M_): min(), min_sum(min_sum_),  max(max_), M(M_)  {
        QMAX = std::numeric_limits<T_TSC>::max();
        //gmin = std::numeric_limits<float>::max();
        min.resize(M);
        for(int m=0;m<M;m++){
        	min[m]= min_[m];
        	FAISS_THROW_IF_NOT_MSG(max > min[m], "Max value to quantize must be larger than min value to quantize");
        	//gmin = std::min(gmin, min[m]);
        }
        //gmin_sum=gmin*M;
    	FAISS_THROW_IF_NOT_MSG(max > min_sum, "Max value to quantize must be larger than min value to quantize");
        delta = (max - min_sum) / QMAX;
        inv_delta = 1.0f/delta;
        //printf("[%f -- %f] (delta: %g, inv_delta: %g\n)",gmin,max,delta,inv_delta);
    }

    void quantize_val(float val, T_TSC* qval, int m) const {
        if(val >= max) {
            *qval = QMAX;
            return;
        }
        *qval = static_cast<T_TSC>(((val - min[m])*inv_delta));
    }

    void quantize_sum(float val, T_TSC* qval)  const {
    	if(val >= max) {
    		*qval = QMAX;
    		return;
    	}
    	*qval = static_cast<T_TSC>(((val - min_sum)*inv_delta));
    }

    float unquantize_sum(T_TSC qval) const {
    	float fval=qval+0.5;
    	return (fval*delta)+min_sum;
    }

    inline void quantize_val_simd(const float* val, int8_t* qval, const int table_size, const int ksub, const int m) const {
    	FAISS_THROW_IF_NOT_MSG(ksub%16 == 0 && table_size%16 == 0 , "Requires table size and ksub to be multiples of of 16");
    	// Set values 0 to ksub to their quantized values and values table_size to ksub-1, if any,  to zero
    	const __m256 min_r = _mm256_set1_ps(min[m]);
    	const __m256 inv_delta_r = _mm256_set1_ps(inv_delta);
    	const __m128i shuf_r = _mm_set_epi8(15,14,13,12, 7,6,5,4  ,11,10,9,8, 3,2,1,0);
    	for(int i=0;i<ksub/16;i++){
    		__m128i * t = (__m128i*)&qval[i*16];
    		float * f1 = (float*)&val[i*16];
    		float * f2 = (float*)&val[i*16+8];
    		__m256 low = _mm256_loadu_ps(f1); // 8x32
    		__m256 high = _mm256_loadu_ps(f2); // 8x32

    		low = _mm256_sub_ps(low, min_r);
    		high = _mm256_sub_ps(high, min_r);
    		low = _mm256_mul_ps(low, inv_delta_r);
    		high = _mm256_mul_ps(high, inv_delta_r);

    		__m256i lowi = _mm256_cvtps_epi32(low);
    		__m256i highi = _mm256_cvtps_epi32(high);
    		__m256i packed16_interleaved4 = _mm256_packs_epi32(lowi, highi); // A B A B
    		__m128i p16i_l = _mm256_extracti128_si256(packed16_interleaved4,0); // A B
    		__m128i p16i_h = _mm256_extracti128_si256(packed16_interleaved4,1); // A B
    		__m128i packed8_interleaved4 = _mm_packs_epi16(p16i_l, p16i_h);  // A B A B
    		// Reorganize...
    		__m128i packed8 = _mm_shuffle_epi8(packed8_interleaved4, shuf_r); // A A A A B B B B
    		_mm_store_si128(t,packed8);
    	}
    	for(int i=ksub/16;i<table_size/16;i++){
    		// Set to zero
    		__m128i * t = (__m128i*)&qval[i*16];
    		_mm_store_si128(t,  _mm_set1_epi8(0));
    	}
    }

    inline void quantize_val_simd(const float* val, int16_t* qval, const int table_size, const int ksub, const int m) const {
        // Set values 0 to ksub to their quantized values and values table_size to ksub-1, if any,  to zero
    	FAISS_THROW_IF_NOT_MSG(ksub%16 == 0 && table_size%16 == 0, "Requires table size and ksub to be multiples of of 16");
		//FIXME Update to use AVX512
    	// Set values 0 to ksub to their quantized values and values table_size to ksub-1, if any,  to zero
		const __m256 min_r = _mm256_set1_ps(min[m]);
		const __m256 inv_delta_r = _mm256_set1_ps(inv_delta);
		for(int i=0;i<ksub/16;i++){
			__m256i * t = (__m256i*)&qval[i*16];
			float * f1 = (float*)&val[i*16];
			float * f2 = (float*)&val[i*16+8];
			__m256 low = _mm256_loadu_ps(f1); // 8x32
			__m256 high = _mm256_loadu_ps(f2); // 8x32

			low = _mm256_sub_ps(low, min_r);
			high = _mm256_sub_ps(high, min_r);
			low = _mm256_mul_ps(low, inv_delta_r);
			high = _mm256_mul_ps(high, inv_delta_r);

			__m256i lowi = _mm256_cvtps_epi32(low);
			__m256i highi = _mm256_cvtps_epi32(high);
			__m256i packed16_interleaved4 = _mm256_packs_epi32(lowi, highi); // A B A B
			__m256i packed16 = _mm256_permute4x64_epi64(packed16_interleaved4,0b11011000); // A A B B
			_mm256_store_si256(t,packed16);
		}
		for(int i=ksub/16;i<table_size/16;i++){
			// Set to zero
			__m256i * t = (__m256i*)&qval[i*16];
			_mm256_store_si256(t,  _mm256_set1_epi16(0));
		}
    }
};


template<>
struct QuantizerMAX<uint8_t> {
    std::vector<float> min;
    float min_sum;
    float max;
    int M;
    float delta;
    float inv_delta;
    //float gmin;
    //float gmin_sum;
    uint8_t QMAX;


    QuantizerMAX(float *min_,float min_sum_, float max_, int M_): min(), min_sum(min_sum_),  max(max_*1.001), M(M_)  {
        QMAX = std::numeric_limits<uint8_t>::max() - M - 1;
        //gmin = std::numeric_limits<float>::max();
        min.resize(M);
        for(int m=0;m<M;m++){
        	min[m]= min_[m];
        	FAISS_THROW_IF_NOT_MSG(max > min[m], "Max value to quantize must be larger than min value to quantize");
        	//gmin = std::min(gmin, min[m]);
        }
        //gmin_sum=gmin*M;
    	FAISS_THROW_IF_NOT_MSG(max > min_sum, "Max value to quantize must be larger than min value to quantize");
        delta = (max - min_sum) / QMAX;
        inv_delta = 1.0f/delta;
        //printf("[%f -- %f] (delta: %g, inv_delta: %g\n)",gmin,max,delta,inv_delta);
    }

    void quantize_val(float val, uint8_t* qval, int m) const {
        if(val >= max) {
            *qval = QMAX + M + 1;  // Sum will be shifted by M (to the limit of uint8_t)
            return;
        }
        *qval = static_cast<uint8_t>(((val - min[m])*inv_delta)) + 1; // Shift by 1 as zero is a sticky in saturated arithmetic
    }

    void quantize_sum(float val, uint8_t* qval)  const {
    	if(val >= max) {
    		*qval = QMAX + M + 1;
    		return;
    	}
    	*qval = static_cast<uint8_t>(((val - min_sum)*inv_delta)) + M;
    }

    float unquantize_sum(uint8_t qval) const {
    	float fval=qval+0.5-M-1;
    	return (fval*delta)+min_sum;
    }

    inline void quantize_val_simd(const float* val, uint8_t* qval, const int table_size, const int ksub, const int m) const {
    	FAISS_THROW_IF_NOT_MSG(ksub%16 == 0 && table_size%16 == 0 , "Requires table size and ksub to be multiples of of 16");
    	// Set values 0 to ksub to their quantized values and values table_size to ksub-1, if any,  to zero
    	const __m256 min_r = _mm256_set1_ps(min[m]);
    	const __m256 inv_delta_r = _mm256_set1_ps(inv_delta);
    	const __m128i shuf_r = _mm_set_epi8(15,14,13,12, 7,6,5,4  ,11,10,9,8, 3,2,1,0);
    	for(int i=0;i<ksub/16;i++){
    		__m128i * t = (__m128i*)&qval[i*16];
    		float * f1 = (float*)&val[i*16];
    		float * f2 = (float*)&val[i*16+8];
    		__m256 low = _mm256_loadu_ps(f1); // 8x32
    		__m256 high = _mm256_loadu_ps(f2); // 8x32

    		low = _mm256_sub_ps(low, min_r);
    		high = _mm256_sub_ps(high, min_r);
    		low = _mm256_mul_ps(low, inv_delta_r);
    		high = _mm256_mul_ps(high, inv_delta_r);

    		__m256i lowi = _mm256_cvtps_epi32(low);
    		__m256i highi = _mm256_cvtps_epi32(high);
    		__m256i packed16_interleaved4 = _mm256_packs_epi32(lowi, highi); // A B A B
    		__m128i p16i_l = _mm256_extracti128_si256(packed16_interleaved4,0); // A B
    		__m128i p16i_h = _mm256_extracti128_si256(packed16_interleaved4,1); // A B
    		__m128i packed8_interleaved4 = _mm_packus_epi16(p16i_l, p16i_h);  // A B A B
    		// Reorganize...
    		__m128i packed8 = _mm_shuffle_epi8(packed8_interleaved4, shuf_r); // A A A A B B B B
    		__m128i offset8 = _mm_add_epi8(packed8, _mm_set1_epi8(1));
    		_mm_store_si128(t,offset8);
    	}
    	for(int i=ksub/16;i<table_size/16;i++){
    		// Set to zero
    		__m128i * t = (__m128i*)&qval[i*16];
    		_mm_store_si128(t,  _mm_set1_epi8(1));
    	}
    }


};


template<>
struct QuantizerMAX<uint16_t> {
    std::vector<float> min;
    float min_sum;
    float max;
    int M;
    float delta;
    float inv_delta;
    //float gmin;
    //float gmin_sum;
    uint16_t QMAX;


    QuantizerMAX(float *min_,float min_sum_, float max_, int M_): min(), min_sum(min_sum_),  max(max_*1.001), M(M_)  {
        QMAX = std::numeric_limits<uint16_t>::max() - M - 1;
        //gmin = std::numeric_limits<float>::max();
        min.resize(M);
        for(int m=0;m<M;m++){
        	min[m]= min_[m];
        	FAISS_THROW_IF_NOT_MSG(max > min[m], "Max value to quantize must be larger than min value to quantize");
        	//gmin = std::min(gmin, min[m]);
        }
        //gmin_sum=gmin*M;
    	FAISS_THROW_IF_NOT_MSG(max > min_sum, "Max value to quantize must be larger than min value to quantize");
        delta = (max - min_sum) / QMAX;
        inv_delta = 1.0f/delta;
        //printf("[%f -- %f] (delta: %g, inv_delta: %g\n)",gmin,max,delta,inv_delta);
    }

    void quantize_val(float val, uint16_t* qval, int m) const {
        if(val >= max) {
            *qval = QMAX + M + 1;  // Sum will be shifted by M (to the limit of uint8_t)
            return;
        }
        *qval = static_cast<uint16_t>(((val - min[m])*inv_delta)) + 1; // Shift by 1 as zero is a sticky in saturated arithmetic
    }

    void quantize_sum(float val, uint16_t* qval)  const {
    	if(val >= max) {
    		*qval = QMAX + M + 1;
    		return;
    	}
    	*qval = static_cast<uint16_t>(((val - min_sum)*inv_delta)) + M;
    }

    float unquantize_sum(uint16_t qval) const {
    	float fval=qval+0.5-M-1;
    	return (fval*delta)+min_sum;
    }

#if 0
    inline void quantize_val_simd(const float* val, uint16_t* qval, const int table_size, const int ksub, const int m) const {
        // Set values 0 to ksub to their quantized values and values table_size to ksub-1, if any,  to zero
    	FAISS_THROW_IF_NOT_MSG(ksub%16 == 0 && table_size%16 == 0, "Requires table size and ksub to be multiples of of 16");
    	// Set values 0 to ksub to their quantized values and values table_size to ksub-1, if any,  to zero
		const __m512 min_r = _mm512_set1_ps(min[m]);
		const __m512 inv_delta_r = _mm512_set1_ps(inv_delta);
		for(int i=0;i<ksub/16;i++){
			__m512i * t = (__m512i*)&qval[i*16];
			float * f1 = (float*)&val[i*16];
			__m512 all = _mm512_loadu_ps(f1); // 8x32

			all = _mm512_sub_ps(all, min_r);
			all = _mm512_mul_ps(all, inv_delta_r);

			__m512i alli = _mm512_cvt_roundps_epi32(all,_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC);
			__m256i lowi = _mm512_extract;
			__m256i highi = _mm512_extract;
			__m256i packed16_interleaved4 = _mm256_packs_epi32(lowi, highi); // A B A B
			__m256i packed16 = _mm256_permute4x64_epi64(packed16_interleaved4,0b11011000); // A A B B
    		__m256i offset16 = _mm256_add_epi16(packed16, _mm256_set1_epi16(1));
			_mm256_store_si256(t,offset16);
		}
		for(int i=ksub/16;i<table_size/16;i++){
			// Set to zero
			__m256i * t = (__m256i*)&qval[i*16];
			_mm256_store_si256(t,  _mm256_set1_epi16(1));
		}
    }
#else
    inline void quantize_val_simd(const float* val, uint16_t* qval, const int table_size, const int ksub, const int m) const {
        // Set values 0 to ksub to their quantized values and values table_size to ksub-1, if any,  to zero
    	FAISS_THROW_IF_NOT_MSG(ksub%16 == 0 && table_size%16 == 0, "Requires table size and ksub to be multiples of of 16");
    	// Set values 0 to ksub to their quantized values and values table_size to ksub-1, if any,  to zero
		const __m256 min_r = _mm256_set1_ps(min[m]);
		const __m256 inv_delta_r = _mm256_set1_ps(inv_delta);
		for(int i=0;i<ksub/16;i++){
			__m256i * t = (__m256i*)&qval[i*16];
			float * f1 = (float*)&val[i*16];
			float * f2 = (float*)&val[i*16+8];
			__m256 low = _mm256_loadu_ps(f1); // 8x32
			__m256 high = _mm256_loadu_ps(f2); // 8x32

			low = _mm256_sub_ps(low, min_r);
			high = _mm256_sub_ps(high, min_r);
			low = _mm256_mul_ps(low, inv_delta_r);
			high = _mm256_mul_ps(high, inv_delta_r);

			__m256i lowi = _mm256_cvtps_epi32(low);
			__m256i highi = _mm256_cvtps_epi32(high);
			__m256i packed16_interleaved4 = _mm256_packus_epi32(lowi, highi); // A B A B
			__m256i packed16 = _mm256_permute4x64_epi64(packed16_interleaved4,0b11011000); // A A B B
    		__m256i offset16 = _mm256_add_epi16(packed16, _mm256_set1_epi16(1));
			_mm256_store_si256(t,offset16);
		}
		for(int i=ksub/16;i<table_size/16;i++){
			// Set to zero
			__m256i * t = (__m256i*)&qval[i*16];
			_mm256_store_si256(t,  _mm256_set1_epi16(1));
		}
    }
#endif

};


/**
 * Vectorized Product Quantizer. Implemented only for METRIC_L2
 * This is a templated class supporting different SIMD width, and supporting different type of quantizers.
 * A few example instantiations are given bellow.
 *
 *  VecProductQuantizer<12,3,6,6,4,0,uint16_t,__m512i,__m512i> // AVX512 - 16 bits distances
 *  VecProductQuantizer<12,3,5,5,5,0,uint16_t,__m512i,__m512i> // AVX512 - 16 bits distances
 *  VecProductQuantizer<12,3,6,5,5,0,uint16_t,__m512i,__m512i> // AVX512 - 16 bits distances
 *  VecProductQuantizer<16,2,4,4,0,0,uint8_t,__m128i,__m128i> // SSE4.2 - 8 bits distances
 *  VecProductQuantizer<16,2,4,4,0,0,int8_t,_mm128i,__m256i> // AVX256 - 8 bits distances (AVX 256 <- use two 128-bit lanes)
 *  VecProductQuantizer<16,2,4,4,0,0,int8_t,_mm128i,__m512i> // AVX512 - 8 bits distances (AVX 512 <- use four 128-bit lanes)
 *  VecProductQuantizer<8,1,8,0,0,0,int8_t,_mm512i,__m512i> // AVX512 - 8 bits distances (256 element table)
 *
 * Note, it is prefered to put {6,6,4} for quantizer specification, rather than {4,6,6} due to the way remaining dsubs are allocated.
 *
 *
 * @tparam T_M     Number of sub_quantizers
 * @tparam T_P     Number of sub quantizers code that can be stored in a single SIMD element (epi16 or epi8)
 * @tparam T_PG_0    Layout in term of bits between subquantizers within a SIMD element (array whose value sum up to the width a of single element)
 * @tparam T_PG_1    Layout in term of bits between subquantizers within a SIMD element (array whose value sum up to the width a of single element)
 * @tparam T_PG_2    Layout in term of bits between subquantizers within a SIMD element (array whose value sum up to the width a of single element)
 * @tparam T_PG_3    Layout in term of bits between subquantizers within a SIMD element (array whose value sum up to the width a of single element)
 * @tparam T_TSC   Basic type correspond to elements (uint16_t for epi16)
 * @tparam T_TSCMM SIMD type desired (__m512i for AVX-512)
 * @tparam T_TSCMMXL    Specification of cross lanes type (e.g., AVX-256 works as two independent 128-bit lanes, other examples use a single 128 or 512-bit lane).
 */
template<int T_M,int T_P,int T_PG_0, int T_PG_1, int T_PG_2, int T_PG_3, class T_TSC, class T_TSCMM, class T_TSCMMXL> struct AbstractVecProductQuantizer {
	typedef CMax<float, long> CHeap;

public:
    static constexpr size_t M = T_M;              ///< number of subquantizers

    // Variants:
    // A: 6,6,4 = 16 bits
    // B: 5,5,5 = 15 bits + 1 bit padding
    // C: 5,5,6 = 16 bits
    // D: 4,4  (AVX 256)
    // => Repeated M times.

    // Interleaving
    static constexpr size_t simd_lanes = sizeof(T_TSCMM)/sizeof(T_TSC);
    static constexpr size_t simd_lane_width_bits = 8*sizeof(T_TSC);
    static constexpr size_t simd_width_bits = simd_lanes * simd_lane_width_bits; // 512
    static constexpr size_t simd_width_bytes = simd_width_bits / 8;
    static constexpr size_t simd_lane_width_bytes = simd_lane_width_bits / 8;

    // Derived values
    static constexpr size_t subcodes_per_lane = T_P;
    static constexpr size_t lanes_per_code = T_M/subcodes_per_lane;
    static constexpr size_t codes_per_group = simd_lanes;
    static constexpr size_t group_size_bytes = lanes_per_code * codes_per_group * simd_lane_width_bits;
    static constexpr size_t cross_lane=sizeof(T_TSCMMXL)/sizeof(T_TSCMM);

    struct group {
        union {
        	T_TSC c[lanes_per_code*codes_per_group] ;
        	T_TSCMM mm[lanes_per_code];
        	T_TSCMMXL mmxl[lanes_per_code/cross_lane];
        };
    };

    static_assert(sizeof(T_TSC)*lanes_per_code*codes_per_group == sizeof(T_TSCMM)*lanes_per_code, "Incorrect layout specification");
    static_assert(sizeof(T_TSC)*lanes_per_code*codes_per_group == sizeof(T_TSCMMXL)*lanes_per_code/cross_lane, "Incorrect layout specification");

    size_t d;              ///< size of the input vectors

   // Layout (exemple for AVX512 - 16 bit distances)
   // a1a2a3 b1b2b3 c1c2c3 ... z1z2z3  = 16 bits x 32 = 512 bits
   // a4a5a6 b4b5b6 c4c5c6 ... z4z5z6  = 16 bits x 32 = 512 bits
   // a7a8a9 b7b8b9 c7c8c9 ... z7z8z9  = 16 bits x 32 = 512 bits
   // axaxax bxbxbx cxcxcx ... zxzxzx  = 16 bits x 32 = 512 bits
   // Group : 4 (4x3 = 12 subquantizers)

    // Layout (example for SSE 4.2 - 8 bit distances)
    // a1a2 b1b2 c1c2 ... p1p2  = 8 bits x 16 = 128 bits
    // a3a4 b3b4 c3c4 ... p3p4  = 8 bits x 16 = 128 bits
    // ...
    // axax bxbx cxcx ... pxpx  = 8 bits x 16 = 128 bits
    // Group : 8 (8x2=16 subquantizers)



    // Offset for accessing/reconstructing real vectors.
    size_t dsub[T_M]; ///< dimensionality of each subvector
    size_t dsub_offset[T_M];
    size_t dsub_total;
    // Total uncompressed = 128
    //  * compressed: ceil( 128 * x*6/(x*6 + y*4))=a => encode to y vectors of a/y (rest on last)
    //  * compressed: 128-a => encode to z vectors of b/z (rest on last)

    // Variables for accessing the
    size_t csub_nbits[T_M];
    size_t csub_register[T_M];
    size_t csub_mask_inlane[T_M]; // (mask with offset already applied)
    size_t csub_offset_inlane[T_M];

    size_t ksub[T_M]; // Number of centroids for each subquantizer
    size_t ksub_offset[T_M];
    size_t ksub_total;

    size_t ksub_sq[T_M]; // Number of centroids for each subquantizer
    size_t ksub_sq_offset[T_M];
    size_t ksub_sq_total;

    size_t dt_lanes[T_M/cross_lane]; // Number of lanes needed for holding distance table (1 or 2)
    size_t dt_lanes_offset[T_M/cross_lane]; // Offset of given lanes
    size_t dt_lanes_total; // Total number of lanes

    // values derived from the above
    //size_t byte_per_idx;   ///< nb bytes per code component (1 or 2)
    //size_t code_size;      ///< byte per indexed vector
    bool verbose;          ///< verbose during training?
    bool ultra_verbose;    ///< verbose during setup ?

    /// initialization
    enum train_type_t {
        Train_default,
        Train_hot_start,   ///< the centroids are already initialized
        Train_shared,      ///< share dictionary accross PQ segments
        Train_hypercube,   ///< intialize centroids with nbits-D hypercube
        Train_hypercube_pca,   ///< intialize centroids with nbits-D hypercube
    };
    train_type_t train_type;

    ClusteringParameters cp; ///< parameters used during clustering

    /// Centroid table, size Sum(ksub[i]*dsub[i])
    std::vector<float> centroids;
    size_t centroids_offset[T_M];
    size_t centroids_size[T_M];
    size_t centroids_total_size; // Sum_i ( ksub[i] * dsub[i]) -- use as a matrix size

    /// return the centroids associated with subvector m
    float * get_centroids (size_t m, size_t i) {
        return &centroids[centroids_offset[m] + dsub[m]*i];
    }
    const float * get_centroids (size_t m, size_t i) const {
        return &centroids[centroids_offset[m] + dsub[m]*i];
    }

    // Train the product quantizer on a set of points. A clustering
    // can be set on input to define non-default clustering parameters
    void train (int n, const float *x) {
		if (train_type != Train_shared) {
			train_type_t final_train_type;
			final_train_type = train_type;
			if (train_type == Train_hypercube ||
				train_type == Train_hypercube_pca) {
				if (d < T_M/T_P*8*sizeof(T_TSC)) {
					final_train_type = Train_default;
					printf ("cannot train hypercube: nbits=%ld > log2(d=%ld)\n",
							T_M/T_P*8*sizeof(T_TSC), d);
				}
			}


			for (int m = 0; m < M; m++) {
				float * xslice = new float[n * dsub[m]];
				ScopeDeleter<float> del (xslice);
				for (int j = 0; j < n; j++)
					memcpy (xslice + j * dsub[m],
							x + j * d + dsub_offset[m],
							dsub[m] * sizeof(float));

				Clustering clus (dsub[m], ksub[m], cp);

				// we have some initialization for the centroids
				if (final_train_type != Train_default) {
					clus.centroids.resize (dsub[m] * ksub[m]);
				}

				switch (final_train_type) {
				case Train_hypercube:
					init_hypercube (dsub[m],csub_nbits[m] , n, xslice,
									clus.centroids.data ());
					break;
				case  Train_hypercube_pca:
					init_hypercube_pca (dsub[m], csub_nbits[m], n, xslice,
										clus.centroids.data ());
					break;
				case  Train_hot_start:
					memcpy (clus.centroids.data(),
							get_centroids (m, 0),
							dsub[m] * ksub[m] * sizeof (float));
					break;
				default: ;
				}

				if(verbose) {
					clus.verbose = true;
					printf ("Training PQ slice %d/%zd\n", m, M);
				}
				IndexFlatL2 index (dsub[m]);
				clus.train (n, xslice, index);
				set_params (clus.centroids.data(), m);
			}


		} else {
	    	FAISS_THROW_IF_NOT (false);

#if 0
	    	// This code path has not been updated for PQ with sub quantizers of unequal width.
			Clustering clus (dsub, ksub, cp);

			if(verbose) {
				clus.verbose = true;
				printf ("Training all PQ slices at once\n");
			}

			IndexFlatL2 index (dsub);
			clus.train (n * M, x, index);
			for (int m = 0; m < M; m++) {
				set_params (clus.centroids.data(), m);
			}
#endif

		}
	}


    AbstractVecProductQuantizer(size_t d) : /* dimensionality of the input vectors */
            	 d(d) {
    	set_derived_values();
    }

    AbstractVecProductQuantizer ()  :
    	d(0)
    {
    	set_derived_values();
    }

    virtual ~AbstractVecProductQuantizer ()
    { }



    inline int t_pg(int i){
    	switch(i){
    	case 0: return T_PG_0;
    	case 1: return T_PG_1;
    	case 2: return T_PG_2;
    	case 3: return T_PG_3;
    	default: FAISS_THROW_MSG("UNSUPPORT T_P SPECIFICATION");
    	}
    }

    inline int t_pgp(int i){
    	//return std::pow((double)t_pg(i),(double)1.25);
    	return t_pg(i);
    }

    /* Number of groups needed for storing n vectors */
    static size_t nb_groups(size_t n){
    	return (n+codes_per_group-1)/codes_per_group;
    }

    /// compute derived values when d, M and nbits have been set
    void set_derived_values () {
        verbose = false;
        ultra_verbose = false;

        // quite a few derived values
        // FAISS_THROW_IF_NOT (d % M == 0); // To support 12 sub quantizers, we have to deal with such case.
    	FAISS_THROW_IF_NOT (d % lanes_per_code == 0);
    	size_t dsub_per_lane = d / lanes_per_code;
    	if(ultra_verbose) std::cout << "d " << d << std::endl;
    	size_t bits_per_lane =  t_pgp(0)+t_pgp(1)+t_pgp(2)+t_pgp(3);
    	//size_t bits_per_lane =  T_PG_0 + T_PG_1 + T_PG_2 + T_PG_3;
    	size_t dsub_per_bit = dsub_per_lane / bits_per_lane;
    	size_t dsub_lane[T_P];
    	size_t consumed_dsubs=0;
    	for(size_t i=0;i<T_P;i++){
    		dsub_lane[i]=dsub_per_bit * t_pgp(i);
 //      		dsub_lane[i]=dsub_per_bit * t_pg(i);
    		consumed_dsubs += dsub_lane[i];
    	}

    	// Assign remaining dsubs
    	for(size_t i=0; consumed_dsubs < dsub_per_lane ; i=(i+1)%T_P){
    		dsub_lane[i] += 1;
    		consumed_dsubs += 1;
    	}

     	if(ultra_verbose){
     		for(size_t i=0;i<T_P;i++){
     			std::cout << "dsub_lane[" << i<<"] " << dsub_lane[i] << std::endl;
     		}
     	}

    	// Compute the dsub_offsets;
    	size_t current_centroid_offset_inlane=0;
    	dsub_total=0;
    	ksub_total=0;
    	dt_lanes_total=0;
    	ksub_sq_total=0;
    	centroids_total_size=0;
    	for(size_t i=0,j=0;i<M;i++,j=i%T_P){
    		dsub[i]=dsub_lane[j];
    		dsub_offset[i]=dsub_total;
    		dsub_total+=dsub[i];


			ksub[i]=1 << t_pg(j);
			ksub_offset[i]=ksub_total;
			ksub_total+=ksub[i];

			ksub_sq[i]=ksub[i]*ksub[i];
			ksub_sq_offset[i]=ksub_sq_total;
			ksub_sq_total+=ksub_sq[i];

			centroids_offset[i]=centroids_total_size;
			centroids_size[i]=dsub[i]*ksub[i];
			centroids_total_size+=dsub[i]*ksub[i];

			csub_nbits[i] = t_pg(j);
			csub_register[i] = i/T_P;
			csub_offset_inlane[i] = current_centroid_offset_inlane;
			csub_mask_inlane[i] = ((1 << t_pg(j)) - 1) << current_centroid_offset_inlane;
			current_centroid_offset_inlane= j == T_P-1 ? 0 : current_centroid_offset_inlane + t_pg(j);

			if(ultra_verbose){
				std::cout << "dsub[" << i << "] " << dsub[i] <<  std::endl;
				std::cout << "dsub_offset[" << i << "]" << dsub_offset[i] <<  std::endl;

				std::cout << "ksub[" << i << "] " << ksub[i] <<  std::endl;
				std::cout << "ksub_offset[" << i << "] " << ksub_offset[i] <<  std::endl;

				std::cout << "ksub_sq[" << i << "] " << ksub_sq[i] <<  std::endl;
				std::cout << "ksub_sq_offset[" << i << "] " << ksub_sq_offset[i] <<  std::endl;

				std::cout << "centroids_offset[" << i << "] " << centroids_offset[i] <<  std::endl;
				std::cout << "centroids_size[" << i << "] " << centroids_size[i] <<  std::endl;

				std::cout << "csub_nbits[" << i << "] " << csub_nbits[i] <<  std::endl;
				std::cout << "csub_register[" << i << "] " << csub_register[i] <<  std::endl;
				std::cout << "csub_offset_inlane[" << i << "] " << csub_offset_inlane[i] <<  std::endl;
				std::cout << "csub_mask_inlane[" << i << "] " << std::hex << csub_mask_inlane[i]  << std::dec <<  std::endl;
			}
    	}


    	for(size_t i=0;i<M/T_P/cross_lane;i++){
    		for(size_t j=0;j<T_P;j++){
    			dt_lanes[i*T_P+j]=(ksub[i*cross_lane*T_P+j]+simd_lanes-1)/simd_lanes*cross_lane;
    			dt_lanes_offset[i*T_P+j]=dt_lanes_total;
    			dt_lanes_total+=dt_lanes[i*T_P+j];
    		}
			if(ultra_verbose){
				for(size_t j=0;j<T_P;j++){
					std::cout << "dt_lanes[" << i*T_P+j << "] " << dt_lanes[i*T_P+j] << std::endl;
					std::cout << "dt_lanes_offset[" << i*T_P+j << "] " << dt_lanes_offset[i*T_P+j] << std::endl;
				}

			}
    	}

        //byte_per_idx = (nbits + 7) / 8;
        //code_size = byte_per_idx * M;
    	if(ultra_verbose){
    		std::cout << "dsub_total: " << dsub_total <<  std::endl;
    		std::cout << "ksub_total " << ksub_total <<  std::endl;
    		std::cout << "ksub_sq_total " << ksub_sq_total <<  std::endl;
    		std::cout << "centroids_total_size " << centroids_total_size <<  std::endl;
    		std::cout <<  "dt_lanes_total " << dt_lanes_total << std::endl;
    	}
    	centroids.resize(centroids_total_size);
        train_type = Train_default;
    }

    /// Define the centroids for subquantizer m
    void set_params (const float * centroids_, int m){
    	  memcpy (get_centroids(m, 0), centroids_,
    	            ksub[m] * dsub[m] * sizeof (centroids_[0]));
    }


    inline void set_code_component(group * codes, size_t idx, size_t m, unsigned val) const {
        /* Merge with other subquantizer indexes */
    	int group_index = idx / codes_per_group;
    	int index_in_group = idx % codes_per_group;
        T_TSC  c = codes[group_index].c[codes_per_group*csub_register[m]+index_in_group];
        c = c & ~csub_mask_inlane[m];
        c = c | (val << csub_offset_inlane[m]);
        codes[group_index].c[codes_per_group*csub_register[m]+index_in_group] = c;
    }

    inline unsigned get_code_component(const group * codes, size_t idx, size_t m) const {
        /* Merge with other subquantizer indexes */
    	int group_index = idx / codes_per_group;
    	int index_in_group = idx % codes_per_group;
        T_TSC  c = codes[group_index].c[codes_per_group*csub_register[m]+index_in_group];
        return (c & csub_mask_inlane[m]) >> csub_offset_inlane[m];
    }





    /// Quantize one vector with the product quantizer
    void encode (const float * x, group * codes, size_t idx) const {
        for (size_t m = 0; m < M; m++) {
            float distances [ksub[m]];
            float mindis = 1e20;
            unsigned idxm = 0;
            const float * xsub = x + dsub_offset[m];

            fvec_L2sqr_ny (distances, xsub, get_centroids(m, 0), dsub[m], ksub[m]);

            /* Find best centroid */
            size_t i;
            for (i = 0; i < ksub[m]; i++) {
                float dis = distances [i];
                if (dis < mindis) {
                    mindis = dis;
                    idxm = i;
                }
            }
            set_code_component(codes, idx, m, idxm);

        }

    }






    /// same as compute_code for several vectors
    void encode_multiple (const float * x,
                        group * codes, size_t offset_idx,
                        size_t n) const {
        if (d/M < 16) { // simple direct computation

			#pragma omp parallel for
            for (size_t i = 0; i < n; i++)
                encode(x + i * d, codes, i+offset_idx);

        } else { // worthwile to use BLAS
            float *dis_tables = new float [n * ksub_total];
            ScopeDeleter<float> del (dis_tables);
            compute_distance_tables (n, x, dis_tables);

    		#pragma omp parallel for
            for (size_t i = 0; i < n; i++) {
                const float * tab = dis_tables + i * ksub_total;
                encode_from_distance_table (tab, codes, i+offset_idx);
            }
        }

    }

    void copy_code(group * codes_dst, size_t offset_dst, const group* codes_src, size_t offset_src, size_t n) {
    	for(size_t i=0;i<n;i++){
    		for(size_t m=0;m < M; m++){
    			set_code_component(codes_dst, i+offset_dst, m, get_code_component(codes_src, i+offset_src, m));
    		}
    	}
    }

    void append_codes(std::vector<group, boost::alignment::aligned_allocator<group, 64>>&  codes_dst,size_t * count_dst, const group* codes_src, size_t offset_src, size_t n) {
    	if(codes_dst.size() < nb_groups(n+*count_dst)) codes_dst.resize(nb_groups(n+*count_dst));
    	copy_code(codes_dst.data(),*count_dst,codes_src,offset_src,n);
        (*count_dst)+=n;
    }

    /// decode a vector from a given code (or n vectors if third argument)
    void decode(const group *codes, float *x, size_t idx) const{
        for (size_t m = 0; m < M; m++) {
        	memcpy(x + dsub_offset[m], get_centroids(m, get_code_component(codes, idx, m)),
        			sizeof(float) * dsub[m]);
        }
    }

    void decode_multiple(const group *codes, float *x, size_t offset_idx, size_t n) const{
        for (size_t i = 0; i < n; i++) {
            this->decode(codes, x + d * i, i+offset_idx);
        }
    }

    /// If we happen to have the distance tables precomputed, this is
    /// more efficient to compute the codes.
    inline void encode_from_distance_table (const float *tab,
                                           group *codes, size_t idx) const {

        for (size_t m = 0; m < M; m++) {
            float mindis = 1e20;
            unsigned idxm = 0;

            /* Find best centroid */
            for (size_t j = 0; j < ksub[m]; j++) {
                float dis = *tab++;
                if (dis < mindis) {
                    mindis = dis;
                    idxm = j;
                }
            }
            set_code_component(codes, idx, m, idxm);
        }
    }


    /** Compute distance table for one vector.
     *
     * The distance table for x = [x_0 x_1 .. x_(M-1)] is a M * ksub
     * matrix that contains
     *
     *   dis_table (m, j) = || x_m - c_(m, j)||^2
     *   for m = 0..M-1 and j = 0 .. ksub - 1
     *
     * where c_(m, j) is the centroid no j of sub-quantizer m.
     *
     * @param x         input vector size d
     * @param dis_table output table, size M * ksub
     */
    void compute_distance_table (const float * x,
                                 float * dis_table) const
    {
        size_t m;

        for (m = 0; m < M; m++) {
            fvec_L2sqr_ny (dis_table + ksub_offset[m],
                           x + dsub_offset[m],
                           get_centroids(m, 0),
                           dsub[m],
                           ksub[m]);
        }
    }

    void compute_inner_prod_table (const float * x,
                                   float * dis_table) const {
        size_t m;

        for (m = 0; m < M; m++) {
            fvec_inner_products_ny (dis_table + ksub_offset[m],
                                    x + dsub_offset[m],
                                    get_centroids(m, 0),
                                    dsub[m],
                                    ksub[m]);
        }
    }


    /** compute distance table for several vectors
     * @param nx        nb of input vectors
     * @param x         input vector size nx * d
     * @param dis_table output table, size nx * M * ksub
     */
    void compute_distance_tables (size_t nx,
                                  const float * x,
                                  float * dis_tables) const {
        if ( d/M < 16) {
        	#pragma omp parallel for
            for (size_t i = 0; i < nx; i++) {
                compute_distance_table (x + i * d, dis_tables + i * ksub_total);
            }

        } else { // use BLAS

            for (int m = 0; m < M; m++) {
                pairwise_L2sqr (dsub[m],
                                nx, x + dsub_offset[m],
                                ksub[m], centroids.data() + centroids_offset[m],
                                dis_tables + ksub_offset[m],
                                d, dsub[m], ksub_total);
            }
        }
    }



    void compute_inner_prod_tables (size_t nx,
                                    const float * x,
                                    float * dis_tables) const{

        if ( d/M < 16) {
    		#pragma omp parallel for
            for (size_t i = 0; i < nx; i++) {
                compute_inner_prod_table (x + i * d, dis_tables + i * ksub_total);
            }

        } else { // use BLAS

            // compute distance tables
            for (int m = 0; m < M; m++) {
                FINTEGER ldc = ksub_total, nxi = nx, ksubi = ksub[m],
                    dsubi = dsub[m], di = d;
                float one = 1.0, zero = 0;

                sgemm_ ("Transposed", "Not transposed",
                        &ksubi, &nxi, &dsubi,
                        &one, &centroids [centroids_offset[m]], &dsubi,
                        x + dsub_offset[m], &di,
                        &zero, dis_tables + ksub_offset[m], &ldc);
            }

        }
    }


    inline void lookup_and_update_heap(size_t ncodes, size_t offset, const group * codes,
    		const float * dis_table,
			int k, float * __restrict heap_dis, long* __restrict heap_ids, float dis0,
			long key, const long* list_ids, bool store_pairs) const {

		for (size_t j = 0; j < ncodes; j++) {
			float dis = 0;

			for (size_t m = 0; m < M; m+=1) {
				int c = get_code_component(codes, offset+j, m);
				dis += dis_table[ksub_offset[m] + c];
			}

			long id;
			if(store_pairs){
				id = (key << 32 | (j+offset));
			}else{
				id = list_ids[j+offset];
			}
			heap_pushpop<CHeap>(k, heap_dis, heap_ids, dis+dis0, id);
		}

    }

    typedef QuantizerMAX<T_TSC> VPQQuant;
    typedef T_TSCMM QuantTableLane;


    virtual VPQQuant * quantize_tables(const float*  dis_table, QuantTableLane* mm_dis_tables, float max) const{
    	/* Find the global per-table minimum */
    	float gmin_global=std::numeric_limits<float>::infinity();
    	float gmin_sum=0.0;
    	float gmin_current[T_M];
    	for(size_t m=0;m<T_M;m++){
    		gmin_current[m]=std::numeric_limits<float>::infinity();
    		for(size_t i=0;i<ksub[m];i++){
    			gmin_current[m]=std::min(dis_table[ksub_offset[m]+i],gmin_current[m]);
    			gmin_global=std::min(dis_table[ksub_offset[m]+i],gmin_global);
    		}
    		gmin_sum+=gmin_current[m];
    	}

    	if(gmin_sum >= max){
    		/* We cannot build a quantizer as all distances quantize to infinite, the whole cell/inverted list can be skipped */
    		return nullptr;
    	}else{
    		/* We can build a quantizer, we will need to scan the inverted list/cell */
			VPQQuant * qmax = new VPQQuant(gmin_current,gmin_sum,max,T_M);
			// Get pointer to primitive type array
			T_TSC* q_dis_tables = (T_TSC*)mm_dis_tables;

			for(int il=0;il<lanes_per_code/cross_lane;il++){ // For each group of lanes (T_M/T_P/T_XL)
				for(int j=0;j<subcodes_per_lane;j++){ // For each sub-code within lane (T_P)
					//T_TSC* qt = q_dis_tables[dt_lanes_offset[il/T_XL*subcodes_per_lane+j]*T_LANES];
						// Lane(s) at position (il*T_XL*T_P + j*T_XL + i )  will hold table for quantizer (il*T_XL*T_P + i*T_P + j)
						// A distance table may require multiple SIMD lanes... cross lane has the priority in that case... (poorly tested)
						int lane_group_pos = il*T_P+j;
						int lane_offset = dt_lanes_offset[lane_group_pos]*simd_lanes;
						int lane_per_table= dt_lanes[lane_group_pos]/cross_lane;
						for(int l=0;l<lane_per_table;l++){
							for(int n=0;n<cross_lane;n++){
	#define SIMD_QUANT
	#ifdef SIMD_QUANT
								int lane_offset_f = lane_offset + l*cross_lane*simd_lanes + n*simd_lanes;
								int m = (il*cross_lane+n)*subcodes_per_lane+j;
								int ksub_offset_r = ksub_offset[m]+l*simd_lanes;
								int ksub_remaining = ksub[m]-l*simd_lanes;
								qmax->quantize_val_simd(&dis_table[ksub_offset_r], (T_TSC*)&q_dis_tables[lane_offset_f], simd_lanes, ksub_remaining, m);
	#else
								for(int o=0;o<simd_lanes;o++){
									int ix = l*simd_lanes+o;
									int lane_offset_f = lane_offset + l*cross_lane*simd_lanes + n*simd_lanes+o;
									int m = (il*cross_lane+n)*subcodes_per_lane+j;
									if(ix < ksub[m]){
										qmax->quantize_val(dis_table[ksub_offset[m]+ix], &q_dis_tables[lane_offset_f],m);
									}else{
										q_dis_tables[lane_offset_f]=0;
									}
								}
	#endif
							}
						}
				}
			}
			return qmax;
    	}
    }



    virtual inline void lookup_and_update_heap_simd(size_t ncodes, size_t offset, const group * codes,
    		                                                 const float * dis_table, T_TSCMM* mm_dis_tables, VPQQuant* qmax,
															 int k, float * __restrict heap_dis, long* __restrict heap_ids, float dis0,
															 long key, const long* list_ids, bool store_pairs) const {
    	FAISS_THROW_MSG("Incomplete implementation.");
    }

    /** perform a search (L2 distance)
     * @param x        query vectors, size nx * d
     * @param nx       nb of queries
     * @param codes    database codes, size ncodes * byte_per_idx
     * @param ncodes   nb of nb vectors
     * @param res      heap array to store results (nh == nx)
     * @param init_finalize_heap  initialize heap (input) and sort (output)?
     */
    void search(const float * x,
                 size_t nx,
                 const group * codes,
                 const size_t ncodes,
                 float_maxheap_array_t *res,
                 bool init_finalize_heap = true,
				 float initial_scan_estim_param = 4) const {
		size_t k = res->k;

		FAISS_THROW_IF_NOT (nx == res->nh);
		float * dis_tables = new float [nx * ksub_total];

		ScopeDeleter<float> del(dis_tables);
		compute_distance_tables (nx, x, dis_tables);


		#pragma omp parallel for
		for (size_t i = 0; i < nx; i++) {
			/* query preparation for asymmetric search: compute look-up tables */
			const float* dis_table = dis_tables + i * ksub_total;
			float dis0=0;

			/* Compute distances and keep smallest values */
			long * __restrict heap_ids = res->ids + i * k;
			float * __restrict heap_dis = res->val + i * k;

			if (init_finalize_heap) {
				heap_heapify<CHeap> (k, heap_dis, heap_ids);
			}

			const size_t codes_scanned_to_build_quantizer = std::min(ncodes,static_cast<unsigned long>(k*initial_scan_estim_param));

			lookup_and_update_heap(codes_scanned_to_build_quantizer, 0, codes, dis_table,k, heap_dis, heap_ids, dis0,0 , NULL, 1);

			std::vector<QuantTableLane,boost::alignment::aligned_allocator<QuantTableLane, 64>> mm_dis_tables;
			mm_dis_tables.resize(dt_lanes_total);

			VPQQuant * qmax = quantize_tables(dis_table, mm_dis_tables.data(), heap_dis[0]);

			if(qmax != nullptr){
				lookup_and_update_heap_simd(ncodes-codes_scanned_to_build_quantizer, codes_scanned_to_build_quantizer, codes, dis_table, mm_dis_tables.data(), qmax, k, heap_dis, heap_ids, dis0,0 , NULL, 1);
				delete qmax;
			}


			if (init_finalize_heap) {
				heap_reorder<CHeap> (k, heap_dis, heap_ids);
			}
		}
	}



    static inline float sqr (float x) {
        return x * x;
    }

    /** same as search, but with inner product similarity */
    void search_ip (const float * x,
                 size_t nx,
                 const group * codes,
                 const size_t ncodes,
                 float_minheap_array_t *res,
                 bool init_finalize_heap = true) const{
    	size_t k = res->k;

    	FAISS_THROW_MSG("Not implemented -- need to make it more generic to use a lookup_and_update which would use CMin.");
		FAISS_THROW_IF_NOT (nx == res->nh);
		float * dis_tables = new float [nx * ksub_total];

		ScopeDeleter<float> del(dis_tables);
		compute_inner_prod_tables (nx, x, dis_tables);


		#pragma omp parallel for
		for (size_t i = 0; i < nx; i++) {
			/* query preparation for asymmetric search: compute look-up tables */
			const float* dis_table = dis_tables + i * ksub_total;

			/* Compute distances and keep smallest values */
			long * __restrict heap_ids = res->ids + i * k;
			float * __restrict heap_dis = res->val + i * k;

			if (init_finalize_heap) {
				heap_heapify<CMin<float,long>> (k, heap_dis, heap_ids);
			}
			float dis0=0;
			lookup_and_update_heap(ncodes, 0, codes, dis_table,k, heap_dis, heap_ids,dis0,0 , NULL, 1);

			if (init_finalize_heap) {
				heap_reorder<CMin<float,long>> (k, heap_dis, heap_ids);
			}
		}
	}


    /// Symmetric Distance Table
    std::vector<float> sdc_table;

    int cmp(group a, size_t offset_a, group b, size_t offset_b) const {
    	for(size_t m=0;m<M;m++){
    		unsigned ca = get_code_component(a, offset_a, m);
    		unsigned cb = get_code_component(b, offset_b, m);
    		if( ca < cb){
    			return -1;
    		}else if(ca > cb){
    			return 1;
    		}
    	}
    	return 0;

    }


    // intitialize the SDC table from the centroids
    void compute_sdc_table () {
        sdc_table.resize (M * ksub_sq_total);

        for (int m = 0; m < M; m++) {

            const float *cents = centroids.data() + centroids_offset[m];
            float * dis_tab = sdc_table.data() + ksub_sq_offset[m];

            // TODO optimize with BLAS
            for (int i = 0; i < ksub[m]; i++) {
                const float *centi = cents + i*dsub[m];
                for (int j = 0; j < ksub[m]; j++) {
                    float accu = 0;
                    const float *centj = cents + j*dsub[m];
                    for (int k = 0; k < dsub[m]; k++)
                        accu += sqr (centi[k] - centj[k]);
                    dis_tab [i + ksub[m]*j] = accu;
                }
            }
        }
    }

    void search_sdc (const group * qcodes,
                     size_t nq,
                     const group * bcodes,
                     const size_t nb,
                     float_maxheap_array_t * res,
                     bool init_finalize_heap = true) const {
        FAISS_THROW_IF_NOT (sdc_table.size() == M * ksub_sq_total);
        size_t k = res->k;


    #pragma omp parallel for
        for (size_t i = 0; i < nq; i++) {

            /* Compute distances and keep smallest values */
            long * heap_ids = res->ids + i * k;
            float *  heap_dis = res->val + i * k;

            if (init_finalize_heap)
                maxheap_heapify (k, heap_dis, heap_ids);

            for (size_t j = 0; j < nb; j++) {
                float dis = 0;
                for (int m = 0; m < M; m++) {
                	const float * tab = sdc_table.data() + ksub_sq_offset[m];
					int cq = get_code_component(qcodes, i, m);
					int cb = get_code_component(bcodes, j, m);
                    dis += tab[cb + cq * ksub[m]];
                }
                if (dis < heap_dis[0]) {
                    maxheap_pop (k, heap_dis, heap_ids);
                    maxheap_push (k, heap_dis, heap_ids, dis, j);
                }
            }

            if (init_finalize_heap)
                maxheap_reorder (k, heap_dis, heap_ids);
        }
    }


	inline void extract_val128(int32_t cmp, const __m128i& candidates, size_t j, bool store_pairs, long key, const long * list_ids, float dis0,
			                int k, float* __restrict__  heap_dis, long * __restrict__  heap_ids,
							const VPQQuant* qmax, __m128i& bh_bound_sse) const {

		const std::uint64_t masktable[] = { 0x0, 0x0, 0x1, 0x100, 0x2, 0x200, 0x201,
				0x20100, 0x3, 0x300, 0x301, 0x30100, 0x302, 0x30200, 0x30201, 0x3020100,
				0x4, 0x400, 0x401, 0x40100, 0x402, 0x40200, 0x40201, 0x4020100, 0x403,
				0x40300, 0x40301, 0x4030100, 0x40302, 0x4030200, 0x4030201, 0x403020100,
				0x5, 0x500, 0x501, 0x50100, 0x502, 0x50200, 0x50201, 0x5020100, 0x503,
				0x50300, 0x50301, 0x5030100, 0x50302, 0x5030200, 0x5030201, 0x503020100,
				0x504, 0x50400, 0x50401, 0x5040100, 0x50402, 0x5040200, 0x5040201,
				0x504020100, 0x50403, 0x5040300, 0x5040301, 0x504030100, 0x5040302,
				0x504030200, 0x504030201, 0x50403020100, 0x6, 0x600, 0x601, 0x60100,
				0x602, 0x60200, 0x60201, 0x6020100, 0x603, 0x60300, 0x60301, 0x6030100,
				0x60302, 0x6030200, 0x6030201, 0x603020100, 0x604, 0x60400, 0x60401,
				0x6040100, 0x60402, 0x6040200, 0x6040201, 0x604020100, 0x60403,
				0x6040300, 0x6040301, 0x604030100, 0x6040302, 0x604030200, 0x604030201,
				0x60403020100, 0x605, 0x60500, 0x60501, 0x6050100, 0x60502, 0x6050200,
				0x6050201, 0x605020100, 0x60503, 0x6050300, 0x6050301, 0x605030100,
				0x6050302, 0x605030200, 0x605030201, 0x60503020100, 0x60504, 0x6050400,
				0x6050401, 0x605040100, 0x6050402, 0x605040200, 0x605040201,
				0x60504020100, 0x6050403, 0x605040300, 0x605040301, 0x60504030100,
				0x605040302, 0x60504030200, 0x60504030201, 0x6050403020100, 0x7, 0x700,
				0x701, 0x70100, 0x702, 0x70200, 0x70201, 0x7020100, 0x703, 0x70300,
				0x70301, 0x7030100, 0x70302, 0x7030200, 0x7030201, 0x703020100, 0x704,
				0x70400, 0x70401, 0x7040100, 0x70402, 0x7040200, 0x7040201, 0x704020100,
				0x70403, 0x7040300, 0x7040301, 0x704030100, 0x7040302, 0x704030200,
				0x704030201, 0x70403020100, 0x705, 0x70500, 0x70501, 0x7050100, 0x70502,
				0x7050200, 0x7050201, 0x705020100, 0x70503, 0x7050300, 0x7050301,
				0x705030100, 0x7050302, 0x705030200, 0x705030201, 0x70503020100,
				0x70504, 0x7050400, 0x7050401, 0x705040100, 0x7050402, 0x705040200,
				0x705040201, 0x70504020100, 0x7050403, 0x705040300, 0x705040301,
				0x70504030100, 0x705040302, 0x70504030200, 0x70504030201,
				0x7050403020100, 0x706, 0x70600, 0x70601, 0x7060100, 0x70602, 0x7060200,
				0x7060201, 0x706020100, 0x70603, 0x7060300, 0x7060301, 0x706030100,
				0x7060302, 0x706030200, 0x706030201, 0x70603020100, 0x70604, 0x7060400,
				0x7060401, 0x706040100, 0x7060402, 0x706040200, 0x706040201,
				0x70604020100, 0x7060403, 0x706040300, 0x706040301, 0x70604030100,
				0x706040302, 0x70604030200, 0x70604030201, 0x7060403020100, 0x70605,
				0x7060500, 0x7060501, 0x706050100, 0x7060502, 0x706050200, 0x706050201,
				0x70605020100, 0x7060503, 0x706050300, 0x706050301, 0x70605030100,
				0x706050302, 0x70605030200, 0x70605030201, 0x7060503020100, 0x7060504,
				0x706050400, 0x706050401, 0x70605040100, 0x706050402, 0x70605040200,
				0x70605040201, 0x7060504020100, 0x706050403, 0x70605040300,
				0x70605040301, 0x7060504030100, 0x70605040302, 0x7060504030200,
				0x7060504030201, 0x706050403020100 };

		if(cmp) {
			T_TSC bh_bound_quant;
			T_TSC candidates_mem[codes_per_group];
			const unsigned first_code_i = 0 + j*codes_per_group;
			_mm_storeu_si128(reinterpret_cast<__m128i*>(candidates_mem), candidates);
			//_mm_prefetch(&list_ids[first_code_i],_MM_HINT_T0);

			// Check low quadword
			const std::uint8_t cmp_low = cmp & 0xff;
			if (cmp_low) {
				const int match_count = _popcnt32(cmp_low);
				std::uint64_t match_pos = masktable[cmp_low];

				for (int i = 0; i < match_count; ++i) {
					const std::uint8_t pos = match_pos & 0xff;
					match_pos >>= 8;

					long id;
					if(store_pairs){
						id = (key << 32 | (first_code_i+pos));
					}else{
						id = list_ids[first_code_i+pos];
					}

					heap_pushpop<CMax<float, long>>(k, heap_dis, heap_ids, qmax->unquantize_sum(candidates_mem[pos])+dis0, id);
				   // bh_push(bh, labels, first_code_i + pos, max_scan,
				   //         candidates_mem[pos]);
				}
			}

			// Check high quadword
			const std::uint8_t cmp_high = (cmp >> 8);
			if (cmp_high) {
				const int match_count = _popcnt32(cmp_high);
				std::uint64_t match_pos = masktable[cmp_high] + 0x0808080808080808;

				for (int i = 0; i < match_count; ++i) {
					const std::uint8_t pos = match_pos & 0xff;
					match_pos >>= 8;

					long id;
					if(store_pairs){
						id = (key << 32 | (first_code_i+pos));
					}else{
						id = list_ids[first_code_i+pos];
					}
					heap_pushpop<CMax<float, long>>(k, heap_dis, heap_ids, qmax->unquantize_sum(candidates_mem[pos])+dis0, id);
				}
			}
			qmax->quantize_sum(heap_dis[0]-dis0, &bh_bound_quant);
			bh_bound_sse = _mm_set1_epi8(bh_bound_quant);
		}
	}


	inline void extract_val(__mmask32 cmp, const __m512i& candidates, size_t j, bool store_pairs, long key, const long * list_ids, float dis0,
			                int k, float* __restrict__  heap_dis, long * __restrict__  heap_ids,
							const VPQQuant* qmax, __m512i& bh_bound_av512) const {

		const std::uint64_t masktable[] = { 0x0, 0x0, 0x1, 0x100, 0x2, 0x200, 0x201,
				0x20100, 0x3, 0x300, 0x301, 0x30100, 0x302, 0x30200, 0x30201, 0x3020100,
				0x4, 0x400, 0x401, 0x40100, 0x402, 0x40200, 0x40201, 0x4020100, 0x403,
				0x40300, 0x40301, 0x4030100, 0x40302, 0x4030200, 0x4030201, 0x403020100,
				0x5, 0x500, 0x501, 0x50100, 0x502, 0x50200, 0x50201, 0x5020100, 0x503,
				0x50300, 0x50301, 0x5030100, 0x50302, 0x5030200, 0x5030201, 0x503020100,
				0x504, 0x50400, 0x50401, 0x5040100, 0x50402, 0x5040200, 0x5040201,
				0x504020100, 0x50403, 0x5040300, 0x5040301, 0x504030100, 0x5040302,
				0x504030200, 0x504030201, 0x50403020100, 0x6, 0x600, 0x601, 0x60100,
				0x602, 0x60200, 0x60201, 0x6020100, 0x603, 0x60300, 0x60301, 0x6030100,
				0x60302, 0x6030200, 0x6030201, 0x603020100, 0x604, 0x60400, 0x60401,
				0x6040100, 0x60402, 0x6040200, 0x6040201, 0x604020100, 0x60403,
				0x6040300, 0x6040301, 0x604030100, 0x6040302, 0x604030200, 0x604030201,
				0x60403020100, 0x605, 0x60500, 0x60501, 0x6050100, 0x60502, 0x6050200,
				0x6050201, 0x605020100, 0x60503, 0x6050300, 0x6050301, 0x605030100,
				0x6050302, 0x605030200, 0x605030201, 0x60503020100, 0x60504, 0x6050400,
				0x6050401, 0x605040100, 0x6050402, 0x605040200, 0x605040201,
				0x60504020100, 0x6050403, 0x605040300, 0x605040301, 0x60504030100,
				0x605040302, 0x60504030200, 0x60504030201, 0x6050403020100, 0x7, 0x700,
				0x701, 0x70100, 0x702, 0x70200, 0x70201, 0x7020100, 0x703, 0x70300,
				0x70301, 0x7030100, 0x70302, 0x7030200, 0x7030201, 0x703020100, 0x704,
				0x70400, 0x70401, 0x7040100, 0x70402, 0x7040200, 0x7040201, 0x704020100,
				0x70403, 0x7040300, 0x7040301, 0x704030100, 0x7040302, 0x704030200,
				0x704030201, 0x70403020100, 0x705, 0x70500, 0x70501, 0x7050100, 0x70502,
				0x7050200, 0x7050201, 0x705020100, 0x70503, 0x7050300, 0x7050301,
				0x705030100, 0x7050302, 0x705030200, 0x705030201, 0x70503020100,
				0x70504, 0x7050400, 0x7050401, 0x705040100, 0x7050402, 0x705040200,
				0x705040201, 0x70504020100, 0x7050403, 0x705040300, 0x705040301,
				0x70504030100, 0x705040302, 0x70504030200, 0x70504030201,
				0x7050403020100, 0x706, 0x70600, 0x70601, 0x7060100, 0x70602, 0x7060200,
				0x7060201, 0x706020100, 0x70603, 0x7060300, 0x7060301, 0x706030100,
				0x7060302, 0x706030200, 0x706030201, 0x70603020100, 0x70604, 0x7060400,
				0x7060401, 0x706040100, 0x7060402, 0x706040200, 0x706040201,
				0x70604020100, 0x7060403, 0x706040300, 0x706040301, 0x70604030100,
				0x706040302, 0x70604030200, 0x70604030201, 0x7060403020100, 0x70605,
				0x7060500, 0x7060501, 0x706050100, 0x7060502, 0x706050200, 0x706050201,
				0x70605020100, 0x7060503, 0x706050300, 0x706050301, 0x70605030100,
				0x706050302, 0x70605030200, 0x70605030201, 0x7060503020100, 0x7060504,
				0x706050400, 0x706050401, 0x70605040100, 0x706050402, 0x70605040200,
				0x70605040201, 0x7060504020100, 0x706050403, 0x70605040300,
				0x70605040301, 0x7060504030100, 0x70605040302, 0x7060504030200,
				0x7060504030201, 0x706050403020100 };


		if (cmp) {
			T_TSC bh_bound_quant;
			T_TSC candidates_mem[codes_per_group];

			const unsigned first_code_i = 0 + j * codes_per_group;
			//_mm_prefetch(&list_ids[first_code_i],_MM_HINT_T0);
			_mm512_storeu_si512(reinterpret_cast<__m512i*>(candidates_mem), candidates);
			// Check low quadword
			const std::uint8_t cmp_low = cmp & 0xff;
			if (cmp_low) {
				const int match_count = _popcnt32(cmp_low);
				std::uint64_t match_pos = masktable[cmp_low];
				for (int i = 0;i < match_count;++i) {
					const std::uint8_t pos = match_pos & 0xff;
					match_pos >>= 8;
					long id;
					if (store_pairs) {
						id = (key << 32 | (first_code_i + pos));
					} else {
						id = list_ids[first_code_i + pos];
					}
					heap_pushpop<CMax<float,long > >(k, heap_dis, heap_ids, qmax->unquantize_sum(candidates_mem[pos]) + dis0, id);
					// bh_push(bh, labels, first_code_i + pos, max_scan,
					//         candidates_mem[pos]);
				}
			}
			// Check high quadword
			const std::uint8_t cmp_high = (cmp >> 8) & 0xff;
			if (cmp_high) {
				const int match_count = _popcnt32(cmp_high);
				std::uint64_t match_pos = masktable[cmp_high] + 0x0808080808080808;
				for (int i = 0;i < match_count;++i) {
					const std::uint8_t pos = match_pos & 0xff;
					match_pos >>= 8;
					long id;
					if (store_pairs) {
						id = (key << 32 | (first_code_i + pos));
					} else {
						id = list_ids[first_code_i + pos];
					}
					heap_pushpop<CMax<float,long > >(k, heap_dis, heap_ids, qmax->unquantize_sum(candidates_mem[pos]) + dis0, id);
				}
			}
			// Check high quadword
			const std::uint8_t cmp_highb = (cmp >> 16) & 0xff;
			if (cmp_highb) {
				const int match_count = _popcnt32(cmp_highb);
				std::uint64_t match_pos = masktable[cmp_highb] + 0x1010101010101010;
				for (int i = 0;i < match_count;++i) {
					const std::uint8_t pos = match_pos & 0xff;
					match_pos >>= 8;
					long id;
					if (store_pairs) {
						id = (key << 32 | (first_code_i + pos));
					} else {
						id = list_ids[first_code_i + pos];
					}
					heap_pushpop<CMax<float,long > >(k, heap_dis, heap_ids, qmax->unquantize_sum(candidates_mem[pos]) + dis0, id);
				}
			}
			// Check high quadword
			const std::uint8_t cmp_highc = (cmp >> 24) & 0xff;
			if (cmp_highc) {
				const int match_count = _popcnt32(cmp_highc);
				std::uint64_t match_pos = masktable[cmp_highc] + 0x1818181818181818;
				for (int i = 0;i < match_count;++i) {
					const std::uint8_t pos = match_pos & 0xff;
					match_pos >>= 8;
					long id;
					if (store_pairs) {
						id = (key << 32 | (first_code_i + pos));
					} else {
						id = list_ids[first_code_i + pos];
					}
					heap_pushpop<CMax<float,long > >(k, heap_dis, heap_ids, qmax->unquantize_sum(candidates_mem[pos]) + dis0, id);
				}
			}
			qmax->quantize_sum(heap_dis[0] - dis0, &bh_bound_quant);
			bh_bound_av512 = _mm512_set1_epi16(bh_bound_quant);
		}
	}


/* New control structure for iterating over masks */
#define FOREACH_IN_MASK32(i,m,mn) for(mn=m,i=__builtin_ctz(mn); mn != 0; mn &= ~(1lu << i),i=__builtin_ctz(mn))
#define FOREACH_IN_MASK64(i,m,mn) for(mn=m,i=__builtin_ctzl(mn); mn != 0; mn &= ~(1llu << i),i=__builtin_ctzl(mn))

/* Example usage
  FOREACH_IN_MASK(idx, new_nat_mask, tmpmask1){
	//printf("nbtosend (new) %u\n",*nb_to_send);
	uint16_t customerid = extract_customer_id(pkt_burst[idx]);
  }
 */


	inline void extract_val_loop(__mmask32 cmp, __m512i& candidates, size_t j, bool store_pairs, long key, const long * list_ids, float dis0,
			                int& k, float* __restrict__  heap_dis, long * __restrict__  heap_ids,
							const VPQQuant* qmax, __m512i& bh_bound_av512) const {

		if (cmp) {
			T_TSC bh_bound_quant;
			T_TSC candidates_mem[codes_per_group];
			const unsigned first_code_i = 0 + j * codes_per_group;
			_mm512_storeu_si512(reinterpret_cast<__m512i*>(candidates_mem), candidates);
			//_mm_prefetch(&list_ids[first_code_i],_MM_HINT_T0);

			int pos;
			int tmp_mask;
			FOREACH_IN_MASK32(pos, cmp, tmp_mask ){
				long id;
				if (store_pairs) {
					id = (key << 32 | (first_code_i + pos));
				} else {
					id = list_ids[first_code_i + pos];
				}
				heap_pushpop<CMax<float,long > >(k, heap_dis, heap_ids, qmax->unquantize_sum(candidates_mem[pos]) + dis0, id);

			}

			qmax->quantize_sum(heap_dis[0] - dis0, &bh_bound_quant);
			bh_bound_av512 = _mm512_set1_epi16(bh_bound_quant);
		}
	}

	inline void extract_val_loop128(unsigned cmp, const __m128i& candidates, size_t j, bool store_pairs, long key, const long * list_ids, float dis0,
			                int& k, float* __restrict__  heap_dis, long * __restrict__  heap_ids,
							const VPQQuant* qmax, __m128i& bh_bound_sse) const {

		if (cmp) {
			T_TSC bh_bound_quant;
			T_TSC candidates_mem[codes_per_group];
			const unsigned first_code_i = 0 + j * codes_per_group;
			_mm_storeu_si128(reinterpret_cast<__m128i*>(candidates_mem), candidates);
			//_mm_prefetch(&list_ids[first_code_i],_MM_HINT_T0);

			int pos;
			int tmp_mask;
			FOREACH_IN_MASK32(pos, cmp, tmp_mask ){
				long id;
				if (store_pairs) {
					id = (key << 32 | (first_code_i + pos));
				} else {
					id = list_ids[first_code_i + pos];
				}
				heap_pushpop<CMax<float,long > >(k, heap_dis, heap_ids, qmax->unquantize_sum(candidates_mem[pos]) + dis0, id);
			}

			qmax->quantize_sum(heap_dis[0] - dis0, &bh_bound_quant);
			bh_bound_sse = _mm_set1_epi8(bh_bound_quant);
		}
	}

};


template<int TT_M, int TT_P, int TT_PG_0, int TT_PG_1, int TT_PG_2, int TT_PG_3, class TT_TSC, class TT_TSCMM, class TT_TSCMMXL>
struct VecProductQuantizer_NoVecTable : public AbstractVecProductQuantizer<TT_M,TT_P,TT_PG_0,TT_PG_1,TT_PG_2,TT_PG_3,TT_TSC,TT_TSCMM,TT_TSCMMXL> {

	typedef typename AbstractVecProductQuantizer<TT_M,TT_P,TT_PG_0,TT_PG_1,TT_PG_2,TT_PG_3,TT_TSC,TT_TSCMM,TT_TSCMMXL>::group group;
	typedef typename AbstractVecProductQuantizer<TT_M,TT_P,TT_PG_0,TT_PG_1,TT_PG_2,TT_PG_3,TT_TSC,TT_TSCMM,TT_TSCMMXL>::VPQQuant VPQQuant;


	VecProductQuantizer_NoVecTable(size_t d) : /* dimensionality of the input vectors */
		AbstractVecProductQuantizer<TT_M,TT_P,TT_PG_0,TT_PG_1,TT_PG_2,TT_PG_3,TT_TSC,TT_TSCMM,TT_TSCMMXL>(d) {}

	VecProductQuantizer_NoVecTable() : /* dimensionality of the input vectors */
			AbstractVecProductQuantizer<TT_M,TT_P,TT_PG_0,TT_PG_1,TT_PG_2,TT_PG_3,TT_TSC,TT_TSCMM,TT_TSCMMXL>() {}



	inline void lookup_and_update_heap_simd(size_t ncodes, size_t offset, const group * codes,
		                                                 const float * dis_table, TT_TSCMM * mm_dis_tables,  VPQQuant * qmax,
														 int k, float * __restrict heap_dis, long* __restrict heap_ids, float dis0,
														 long key, const long* list_ids, bool store_pairs) const override {


		FAISS_THROW_IF_NOT_MSG(this->cross_lane == 1 , "This requires quantize tables not layed out for cross-lanes operations");

		for (size_t j = 0; j < ncodes; j++) {
			int32_t dis = 0;

			for (size_t m = 0; m < TT_M; m+=1) {
				TT_TSC* quant_dis_table = (TT_TSC*)&mm_dis_tables[this->dt_lanes_offset[m]];
				int c = this->get_code_component(codes, offset+j, m);
				//float disf = dis_table[this->ksub_offset[m] + c];
				TT_TSC dis8 = quant_dis_table[c];
				//qmax.quantize_val(disf, &dis8);
				dis += static_cast<int32_t>(dis8);
			}

			// Apply saturation
			dis = std::min((int32_t)std::numeric_limits<TT_TSC>::max(), dis);

			long id;
			if(store_pairs){
				id = (key << 32 | (j+offset));
			}else{
				id = list_ids[j+offset];
			}
			heap_pushpop<typename AbstractVecProductQuantizer<TT_M,TT_P,TT_PG_0,TT_PG_1,TT_PG_2,TT_PG_3,TT_TSC,TT_TSCMM,TT_TSCMMXL>::CHeap>(k, heap_dis, heap_ids, qmax->unquantize_sum(dis)+dis0, id);
		}


	}
};


template<int TT_M, int TT_P, int TT_PG_0, int TT_PG_1, int TT_PG_2, int TT_PG_3, class TT_TSC, class TT_TSCMM, class TT_TSCMMXL>
struct VecProductQuantizer_NoVec : public AbstractVecProductQuantizer<TT_M,TT_P,TT_PG_0,TT_PG_1,TT_PG_2,TT_PG_3,TT_TSC,TT_TSCMM,TT_TSCMMXL> {

	typedef typename AbstractVecProductQuantizer<TT_M,TT_P,TT_PG_0,TT_PG_1,TT_PG_2,TT_PG_3,TT_TSC,TT_TSCMM,TT_TSCMMXL>::group group;
	typedef typename AbstractVecProductQuantizer<TT_M,TT_P,TT_PG_0,TT_PG_1,TT_PG_2,TT_PG_3,TT_TSC,TT_TSCMM,TT_TSCMMXL>::VPQQuant VPQQuant;


	VecProductQuantizer_NoVec(size_t d) : /* dimensionality of the input vectors */
		AbstractVecProductQuantizer<TT_M,TT_P,TT_PG_0,TT_PG_1,TT_PG_2,TT_PG_3,TT_TSC,TT_TSCMM,TT_TSCMMXL>(d) {}

	VecProductQuantizer_NoVec() : /* dimensionality of the input vectors */
			AbstractVecProductQuantizer<TT_M,TT_P,TT_PG_0,TT_PG_1,TT_PG_2,TT_PG_3,TT_TSC,TT_TSCMM,TT_TSCMMXL>() {}



	inline void lookup_and_update_heap_simd(size_t ncodes, size_t offset, const group * codes,
		                                                 const float * dis_table, TT_TSCMM * mm_dis_tables,  VPQQuant * qmax,
														 int k, float * __restrict heap_dis, long* __restrict heap_ids, float dis0,
														 long key, const long* list_ids, bool store_pairs) const override {



		for (size_t j = 0; j < ncodes; j++) {
			int32_t dis = 1; // 1 for unsigned distance
//FIXME dis=0 for signed distance

			for (size_t m = 0; m < TT_M; m+=1) {
				int c = this->get_code_component(codes, offset+j, m);
				float disf = dis_table[this->ksub_offset[m] + c];
				TT_TSC dis8;
				qmax->quantize_val(disf, &dis8,m);
				dis += static_cast<int32_t>(dis8);
			}

			// Apply saturation
			dis = std::min((int32_t)std::numeric_limits<TT_TSC>::max(), dis);

			long id;
			if(store_pairs){
				id = (key << 32 | (j+offset));
			}else{
				id = list_ids[j+offset];
			}
			heap_pushpop<typename AbstractVecProductQuantizer<TT_M,TT_P,TT_PG_0,TT_PG_1,TT_PG_2,TT_PG_3,TT_TSC,TT_TSCMM,TT_TSCMMXL>::CHeap>(k, heap_dis, heap_ids, qmax->unquantize_sum(dis)+dis0, id);
		}


	}


	/* We do not need (and must not since memory is not necessarily of the right size) build quantized lookup tables */
	VPQQuant * quantize_tables(const float*  dis_table, TT_TSCMM* mm_dis_tables, float max) const override{
		/* Find the global per-table minimum */
		float gmin_global=std::numeric_limits<float>::infinity();
		float gmin_sum=0.0;
		float gmin_current[TT_M];
		for(size_t m=0;m<TT_M;m++){
			gmin_current[m]=std::numeric_limits<float>::infinity();
			for(size_t i=0;i<this->ksub[m];i++){
				gmin_current[m]=std::min(dis_table[this->ksub_offset[m]+i],gmin_current[m]);
				gmin_global=std::min(dis_table[this->ksub_offset[m]+i],gmin_global);
			}
			gmin_sum+=gmin_current[m];
		}

    	if(gmin_sum >= max){
    		/* We cannot build a quantizer as all distances quantize to infinite, the whole cell/inverted list can be skipped */
    		return nullptr;
    	}else{
    		return new VPQQuant(gmin_current,gmin_sum,max,TT_M);
    	}
	}

};

template<int TT_M, int TT_P, int TT_PG_0, int TT_PG_1, int TT_PG_2, int TT_PG_3, class TT_TSC, class TT_TSCMM,class TT_TSCMMXL>
struct VecProductQuantizer_NoVecNoQuant : public AbstractVecProductQuantizer<TT_M,TT_P,TT_PG_0,TT_PG_1,TT_PG_2,TT_PG_3,TT_TSC,TT_TSCMM,TT_TSCMMXL> {

	typedef typename AbstractVecProductQuantizer<TT_M,TT_P,TT_PG_0,TT_PG_1,TT_PG_2,TT_PG_3,TT_TSC,TT_TSCMM,TT_TSCMMXL>::group group;
	typedef typename AbstractVecProductQuantizer<TT_M,TT_P,TT_PG_0,TT_PG_1,TT_PG_2,TT_PG_3,TT_TSC,TT_TSCMM,TT_TSCMMXL>::VPQQuant VPQQuant;

	VecProductQuantizer_NoVecNoQuant(size_t d) : /* dimensionality of the input vectors */
		AbstractVecProductQuantizer<TT_M,TT_P,TT_PG_0,TT_PG_1,TT_PG_2,TT_PG_3,TT_TSC,TT_TSCMM,TT_TSCMMXL>(d) {}

	VecProductQuantizer_NoVecNoQuant() : /* dimensionality of the input vectors */
			AbstractVecProductQuantizer<TT_M,TT_P,TT_PG_0,TT_PG_1,TT_PG_2,TT_PG_3,TT_TSC,TT_TSCMM,TT_TSCMMXL>() {}


	inline void lookup_and_update_heap_simd(size_t ncodes, size_t offset, const group * codes,
		                                                 const float * dis_table, TT_TSCMM * mm_dis_tables, VPQQuant* qmax,
														 int k, float * __restrict heap_dis, long* __restrict heap_ids, float dis0,
														 long key, const long* list_ids, bool store_pairs) const override {

		this->lookup_and_update_heap(ncodes, offset, codes, dis_table, k, heap_dis, heap_ids, dis0,key,list_ids,store_pairs);
	}

	/* We do not need (and must not since memory is not necessarily of the right size) build quantized lookup tables */
	VPQQuant * quantize_tables(const float*  dis_table, TT_TSCMM* mm_dis_tables, float max) const override{
		/* Find the global per-table minimum */
		float gmin_global=std::numeric_limits<float>::infinity();
		float gmin_sum=0.0;
		float gmin_current[TT_M];
		for(size_t m=0;m<TT_M;m++){
			gmin_current[m]=std::numeric_limits<float>::infinity();
			for(size_t i=0;i<this->ksub[m];i++){
				gmin_current[m]=std::min(dis_table[this->ksub_offset[m]+i],gmin_current[m]);
				gmin_global=std::min(dis_table[this->ksub_offset[m]+i],gmin_global);
			}
			gmin_sum+=gmin_current[m];
		}

    	if(gmin_sum >= max){
    		/* We cannot build a quantizer as all distances quantize to infinite, the whole cell/inverted list can be skipped */
    		return nullptr;
    	}else{
    		return new VPQQuant(gmin_current,gmin_sum,max,TT_M);
    	}
	}
};



template<int TT_M>
struct VecProductQuantizer_4_AVX256 : public AbstractVecProductQuantizer<TT_M,2,4,4,0,0,int8_t,__m128i,__m256i> {

	typedef typename AbstractVecProductQuantizer<TT_M,2,4,4,0,0,int8_t,__m128i,__m256i>::group group;
	typedef typename AbstractVecProductQuantizer<TT_M,2,4,4,0,0,int8_t,__m128i,__m256i>::VPQQuant VPQQuant;



	VecProductQuantizer_4_AVX256(size_t d) : /* dimensionality of the input vectors */
		AbstractVecProductQuantizer<TT_M,2,4,4,0,0,int8_t,__m128i,__m256i>(d) {
	}

	VecProductQuantizer_4_AVX256() : /* dimensionality of the input vectors */
		AbstractVecProductQuantizer<TT_M,2,4,4,0,0,int8_t,__m128i,__m256i>() {
	}

	//__attribute__((optimize("unroll-loops")))
	inline void lookup_and_update_heap_simd(size_t ncodes, size_t offset, const group * __restrict codes,
		                                                 const float * __restrict dis_table, __m128i *__restrict mm_dis_tables1, VPQQuant * qmax,
														 int k, float * __restrict heap_dis, long* __restrict heap_ids, float dis0,
														 long key, const long* list_ids, bool store_pairs) const override {

		const __m256i low_mask_avx = _mm256_set1_epi8(0x0f);

		// Access static member of super class (shortcut)
		constexpr int codes_per_group = AbstractVecProductQuantizer<TT_M,2,4,4,0,0,int8_t,__m128i,__m256i>::codes_per_group;
		constexpr int lanes_per_code = AbstractVecProductQuantizer<TT_M,2,4,4,0,0,int8_t,__m128i,__m256i>::lanes_per_code;


		__m256i * __restrict mm_dis_tables = (__m256i*)mm_dis_tables1;

		// Binheap extraction
		int8_t bh_bound_quant;
		qmax->quantize_sum(heap_dis[0]-dis0, &bh_bound_quant);
		__m128i bh_bound_sse = _mm_set1_epi8(bh_bound_quant);

		int start_group_index = offset / codes_per_group;
		int start_index_in_group = offset % codes_per_group;

		int last_group_index = (offset+ncodes-1)/codes_per_group;
		int last_index_in_group = (offset+ncodes-1) % codes_per_group;

		int start_mask=~((1 << start_index_in_group)-1);  //2 => 0b0000011 => 0b11111100;
		int end_mask=((1 <<(last_index_in_group+1)) -1); //2 => 3 => 0b0000111


		for (size_t j = start_group_index; j <= last_group_index; j++) {

			__m256i twolane_sum = _mm256_setzero_si256();
			// Rows 1..ROW_COUNT
			for(int row_i = 0; row_i < lanes_per_code/2; ++row_i) {
				// Lookup add (low)
				const __m256i comps = _mm256_load_si256(&codes[j].mmxl[row_i]);
				const __m256i masked = _mm256_and_si256(comps, low_mask_avx);
				const __m256i partiala = _mm256_shuffle_epi8(mm_dis_tables[2*row_i], masked);

				// Lookup add (high)
				const __m256i compsb = _mm256_srli_epi64(comps, 4);
				const __m256i maskedb = _mm256_and_si256(compsb, low_mask_avx);
				const __m256i partialb = _mm256_shuffle_epi8(mm_dis_tables[2*row_i+1], maskedb);
				const __m256i partial_sum = _mm256_adds_epi8(partiala, partialb);
				twolane_sum = _mm256_adds_epi8(twolane_sum,partial_sum);
			}

			// Reduce
			const __m128i sum_a = _mm256_extracti128_si256(twolane_sum,0);
			const __m128i sum_b = _mm256_extracti128_si256(twolane_sum,1);
			const __m128i candidates = _mm_adds_epu8(sum_a, sum_b);

			// Compare
			const __m128i compare = _mm_cmplt_epi8(candidates, bh_bound_sse);
			int cmp = _mm_movemask_epi8(compare);

			// Apply masks for potentially incomplete first and last groups
			if(unlikely(j == start_group_index)){
				cmp &= start_mask;
			}
			if(unlikely(j == last_group_index)){
				cmp &= end_mask;
			}

			if(cmp){
				this->extract_val128(cmp, candidates, j, store_pairs, key, list_ids, dis0, k, heap_dis, heap_ids, qmax, bh_bound_sse);
			}
			//compare_extract_matches_sse(candidates, bh_bound_sse, scanned, max_scan,
			//		candidates_mem, bh, labels, labels_offset);

		}

	};
};



template<int TT_M>
struct VecProductQuantizer_4_AVX256_unsigned : public AbstractVecProductQuantizer<TT_M,2,4,4,0,0,uint8_t,__m128i,__m256i> {

	typedef typename AbstractVecProductQuantizer<TT_M,2,4,4,0,0,uint8_t,__m128i,__m256i>::group group;
	typedef typename AbstractVecProductQuantizer<TT_M,2,4,4,0,0,uint8_t,__m128i,__m256i>::VPQQuant VPQQuant;



	VecProductQuantizer_4_AVX256_unsigned(size_t d) : /* dimensionality of the input vectors */
		AbstractVecProductQuantizer<TT_M,2,4,4,0,0,uint8_t,__m128i,__m256i>(d) {
	}

	VecProductQuantizer_4_AVX256_unsigned() : /* dimensionality of the input vectors */
		AbstractVecProductQuantizer<TT_M,2,4,4,0,0,uint8_t,__m128i,__m256i>() {
	}

	//__attribute__((optimize("unroll-loops")))
	inline void lookup_and_update_heap_simd(size_t ncodes, size_t offset, const group * __restrict codes,
		                                                 const float * __restrict dis_table, __m128i *__restrict mm_dis_tables1, VPQQuant * qmax,
														 int k, float * __restrict heap_dis, long* __restrict heap_ids, float dis0,
														 long key, const long* list_ids, bool store_pairs) const override {

		const __m256i low_mask_avx = _mm256_set1_epi8(0x0f);

		// Access static member of super class (shortcut)
		constexpr int codes_per_group = AbstractVecProductQuantizer<TT_M,2,4,4,0,0,uint8_t,__m128i,__m256i>::codes_per_group;
		constexpr int lanes_per_code = AbstractVecProductQuantizer<TT_M,2,4,4,0,0,uint8_t,__m128i,__m256i>::lanes_per_code;


		__m256i * __restrict mm_dis_tables = (__m256i*)mm_dis_tables1;

		// Binheap extraction
		uint8_t bh_bound_quant;
		qmax->quantize_sum(heap_dis[0]-dis0, &bh_bound_quant);
		__m128i bh_bound_sse = _mm_set1_epi8(bh_bound_quant);

		int start_group_index = offset / codes_per_group;
		int start_index_in_group = offset % codes_per_group;

		int last_group_index = (offset+ncodes-1)/codes_per_group;
		int last_index_in_group = (offset+ncodes-1) % codes_per_group;

		int start_mask=~((1 << start_index_in_group)-1);  //2 => 0b0000011 => 0b11111100;
		int end_mask=((1 <<(last_index_in_group+1)) -1); //2 => 3 => 0b0000111


		for (size_t j = start_group_index; j <= last_group_index; j++) {

			//__m256i twolane_sum = _mm256_setzero_si256();
			__m256i twolane_sum = _mm256_set1_epi8(1);
			// Rows 1..ROW_COUNT
			for(int row_i = 0; row_i < lanes_per_code/2; ++row_i) {
				// Lookup add (low)
				const __m256i comps = _mm256_load_si256(&codes[j].mmxl[row_i]);
				const __m256i masked = _mm256_and_si256(comps, low_mask_avx);
				const __m256i partiala = _mm256_shuffle_epi8(mm_dis_tables[2*row_i], masked);
				//twolane_sum = _mm256_adds_epu8(twolane_sum,partiala);


				// Lookup add (high)
				const __m256i compsb = _mm256_srli_epi64(comps, 4);
				const __m256i maskedb = _mm256_and_si256(compsb, low_mask_avx);
				const __m256i partialb = _mm256_shuffle_epi8(mm_dis_tables[2*row_i+1], maskedb);
				const __m256i partial_sum = _mm256_adds_epu8(partiala,partialb);

				twolane_sum = _mm256_adds_epu8(twolane_sum,partial_sum);

			}

			// Reduce
			const __m128i sum_a = _mm256_extracti128_si256(twolane_sum,0);
			const __m128i sum_b = _mm256_extracti128_si256(twolane_sum,1);
			const __m128i sum_b_minus1 = _mm_subs_epu8(sum_b,_mm_set1_epi8(1));
			const __m128i candidates = _mm_adds_epu8(sum_a, sum_b_minus1);

			// Compare
			const __m128i compare = _mm_cmplt_epu8(candidates, bh_bound_sse);
			int cmp = _mm_movemask_epi8(compare);

			// Apply masks for potentially incomplete first and last groups
			if(unlikely(j == start_group_index)){
				cmp &= start_mask;
			}
			if(unlikely(j == last_group_index)){
				cmp &= end_mask;
			}

			if(cmp){
				this->extract_val128(cmp, candidates, j, store_pairs, key, list_ids, dis0, k, heap_dis, heap_ids, qmax, bh_bound_sse);
			}
			//compare_extract_matches_sse(candidates, bh_bound_sse, scanned, max_scan,
			//		candidates_mem, bh, labels, labels_offset);

		}

	};
};

#ifdef __AVX512F__
template<int TT_M>
struct VecProductQuantizer_4_AVX512_unsigned : public AbstractVecProductQuantizer<TT_M,2,4,4,0,0,uint8_t,__m128i,__m512i> {

	typedef typename AbstractVecProductQuantizer<TT_M,2,4,4,0,0,uint8_t,__m128i,__m512i>::group group;
	typedef typename AbstractVecProductQuantizer<TT_M,2,4,4,0,0,uint8_t,__m128i,__m512i>::VPQQuant VPQQuant;



	VecProductQuantizer_4_AVX512_unsigned(size_t d) : /* dimensionality of the input vectors */
		AbstractVecProductQuantizer<TT_M,2,4,4,0,0,uint8_t,__m128i,__m512i>(d) {
	}

	VecProductQuantizer_4_AVX512_unsigned() : /* dimensionality of the input vectors */
		AbstractVecProductQuantizer<TT_M,2,4,4,0,0,uint8_t,__m128i,__m512i>() {
	}

	//__attribute__((optimize("unroll-loops")))
	inline void lookup_and_update_heap_simd(size_t ncodes, size_t offset, const group * __restrict codes,
		                                                 const float * __restrict dis_table, __m128i *__restrict mm_dis_tables1, VPQQuant * qmax,
														 int k, float * __restrict heap_dis, long* __restrict heap_ids, float dis0,
														 long key, const long* list_ids, bool store_pairs) const override {

		const __m512i low_mask_avx = _mm512_set1_epi8(0x0f);

		// Access static member of super class (shortcut)
		constexpr int codes_per_group = AbstractVecProductQuantizer<TT_M,2,4,4,0,0,uint8_t,__m128i,__m512i>::codes_per_group;
		constexpr int lanes_per_code = AbstractVecProductQuantizer<TT_M,2,4,4,0,0,uint8_t,__m128i,__m512i>::lanes_per_code;


		__m512i * __restrict mm_dis_tables = (__m512i*)mm_dis_tables1;

		// Binheap extraction
		uint8_t bh_bound_quant;
		qmax->quantize_sum(heap_dis[0]-dis0, &bh_bound_quant);
		__m128i bh_bound_sse = _mm_set1_epi8(bh_bound_quant);

		int start_group_index = offset / codes_per_group;
		int start_index_in_group = offset % codes_per_group;

		int last_group_index = (offset+ncodes-1)/codes_per_group;
		int last_index_in_group = (offset+ncodes-1) % codes_per_group;

		int start_mask=~((1 << start_index_in_group)-1);  //2 => 0b0000011 => 0b11111100;
		int end_mask=((1 <<(last_index_in_group+1)) -1); //2 => 3 => 0b0000111


		for (size_t j = start_group_index; j <= last_group_index; j++) {

			__m512i fourlane_sum = _mm512_set1_epi8(1);

			// Rows 0..ROW_COUNT
			for(int row_i = 0; row_i < lanes_per_code/4; ++row_i) {
				// Lookup add (low)
				const __m512i comps = _mm512_load_si512(&codes[j].mmxl[row_i]);
				const __m512i masked = _mm512_and_si512(comps, low_mask_avx);
				const __m512i partiala = _mm512_shuffle_epi8(mm_dis_tables[2*row_i], masked);
				//fourlane_sum = _mm512_adds_epu8(fourlane_sum,partiala);

				// Lookup add (high)
				const __m512i compsb = _mm512_srli_epi64(comps, 4);
				const __m512i maskedb = _mm512_and_si512(compsb, low_mask_avx);
				const __m512i partialb = _mm512_shuffle_epi8(mm_dis_tables[2*row_i+1], maskedb);
				const __m512i partial_sum = _mm512_adds_epu8(partiala,partialb);
				fourlane_sum = _mm512_adds_epu8(fourlane_sum,partial_sum);
			}

			// Reduce
			__m128i sum_a = _mm512_extracti32x4_epi32(fourlane_sum,0);
			__m128i sum_b = _mm_subs_epu8(_mm512_extracti32x4_epi32(fourlane_sum,1), _mm_set1_epi8(1));
			__m128i sum_c = _mm_subs_epu8(_mm512_extracti32x4_epi32(fourlane_sum,2), _mm_set1_epi8(1));
			__m128i sum_d = _mm_subs_epu8(_mm512_extracti32x4_epi32(fourlane_sum,3), _mm_set1_epi8(1));


			const __m128i candidates =  _mm_adds_epu8(
					_mm_adds_epu8(sum_a,sum_b),
					_mm_adds_epu8(sum_c,sum_d));

			// Compare
			const __m128i compare = _mm_cmplt_epu8(candidates, bh_bound_sse);
			int cmp = _mm_movemask_epi8(compare);

			// Apply masks for potentially incomplete first and last groups
			if(unlikely(j == start_group_index)){
				cmp &= start_mask;
			}
			if(unlikely(j == last_group_index)){
				cmp &= end_mask;
			}

			if(cmp){
				this->extract_val128(cmp, candidates, j, store_pairs, key, list_ids, dis0, k, heap_dis, heap_ids, qmax, bh_bound_sse);
			}
			//compare_extract_matches_sse(candidates, bh_bound_sse, scanned, max_scan,
			//		candidates_mem, bh, labels, labels_offset);

		}

	};
};
#else
template<int TT_M>
struct VecProductQuantizer_4_AVX512_unsigned : public VecProductQuantizer_NoVec<TT_M,2,4,4,0,0,uint8_t,__m128i,__m512i> {

	VecProductQuantizer_4_AVX512_unsigned(size_t d) : /* dimensionality of the input vectors */
		VecProductQuantizer_NoVec<TT_M,2,4,4,0,0,uint8_t,__m128i,__m512i>(d) {
	}

	VecProductQuantizer_4_AVX512_unsigned() : /* dimensionality of the input vectors */
		VecProductQuantizer_NoVec<TT_M,2,4,4,0,0,uint8_t,__m128i,__m512i>() {
	}

};

#endif



template<int TT_M>
struct VecProductQuantizer_4_SSE128 : public AbstractVecProductQuantizer<TT_M,2,4,4,0,0,int8_t,__m128i,__m128i> {

	typedef typename AbstractVecProductQuantizer<TT_M,2,4,4,0,0,int8_t,__m128i,__m128i>::group group;
	typedef typename AbstractVecProductQuantizer<TT_M,2,4,4,0,0,int8_t,__m128i,__m128i>::VPQQuant VPQQuant;



	VecProductQuantizer_4_SSE128(size_t d) : /* dimensionality of the input vectors */
		AbstractVecProductQuantizer<TT_M,2,4,4,0,0,int8_t,__m128i,__m128i>(d) {
	}

	VecProductQuantizer_4_SSE128() : /* dimensionality of the input vectors */
		AbstractVecProductQuantizer<TT_M,2,4,4,0,0,int8_t,__m128i,__m128i>() {
	}

	//__attribute__((optimize("unroll-loops")))
	inline void lookup_and_update_heap_simd(size_t ncodes, size_t offset, const group * __restrict codes,
		                                                 const float * __restrict dis_table, __m128i *__restrict mm_dis_tables, VPQQuant * qmax,
														 int k, float * __restrict heap_dis, long* __restrict heap_ids, float dis0,
														 long key, const long* list_ids, bool store_pairs) const override {

	    const std::uint64_t low_mask = 0x0f0f0f0f0f0f0f0f;
	    const __m128i low_mask_sse = _mm_set_epi64x(low_mask, low_mask);

		// Access static member of super class (shortcut)
		constexpr int codes_per_group = AbstractVecProductQuantizer<TT_M,2,4,4,0,0,int8_t,__m128i,__m256i>::codes_per_group;
		constexpr int lanes_per_code = AbstractVecProductQuantizer<TT_M,2,4,4,0,0,int8_t,__m128i,__m256i>::lanes_per_code;

		// Binheap extraction
		int8_t bh_bound_quant;
		qmax->quantize_sum(heap_dis[0]-dis0, &bh_bound_quant);
		__m128i bh_bound_sse = _mm_set1_epi8(bh_bound_quant);

		int start_group_index = offset / codes_per_group;
		int start_index_in_group = offset % codes_per_group;

		int last_group_index = (offset+ncodes-1)/codes_per_group;
		int last_index_in_group = (offset+ncodes-1) % codes_per_group;

		int start_mask=~((1 << start_index_in_group)-1);  //2 => 0b0000011 => 0b11111100;
		int end_mask=((1 <<(last_index_in_group+1)) -1); //2 => 3 => 0b0000111


		for (size_t j = start_group_index; j <= last_group_index; j++) {

			// Subquantizer 0
			const __m128i comps_01 = _mm_load_si128(&codes[j].mmxl[0]);
			const __m128i comps_0 = _mm_and_si128(comps_01, low_mask_sse);
			__m128i candidates = _mm_shuffle_epi8(mm_dis_tables[0], comps_0);

			// Subquantizer 1
			const __m128i comps_01_shift = _mm_srli_epi64(comps_01, 4);
			const __m128i comps_1 = _mm_and_si128(comps_01_shift, low_mask_sse);
			const __m128i partial = _mm_shuffle_epi8(mm_dis_tables[1], comps_1);
			candidates = _mm_adds_epi8(candidates, partial);

			// Subquantizers 2..SQ_COUNT
			for(int row_i = 1; row_i < lanes_per_code; ++row_i) {
				const int sq_i = row_i * 2;
				const __m128i comps = _mm_loadu_si128(&codes[j].mmxl[row_i]);
				// Low comps
				const __m128i comps_low = _mm_and_si128(comps, low_mask_sse);
				const __m128i partial_low = _mm_shuffle_epi8(mm_dis_tables[sq_i], comps_low);
				candidates = _mm_adds_epi8(candidates, partial_low);
				// High comps
				const __m128i comps_shift = _mm_srli_epi64(comps, 4);
				const __m128i comps_high = _mm_and_si128(comps_shift, low_mask_sse);
				const __m128i partial_high = _mm_shuffle_epi8(mm_dis_tables[sq_i + 1],comps_high);
				candidates = _mm_adds_epi8(candidates, partial_high);
			}

			// Compare
			const __m128i compare = _mm_cmplt_epi8(candidates, bh_bound_sse);
			int cmp = _mm_movemask_epi8(compare);

			// Apply masks for potentially incomplete first and last groups
			if(unlikely(j == start_group_index)){
				cmp &= start_mask;
			}
			if(unlikely(j == last_group_index)){
				cmp &= end_mask;
			}

			if(cmp){
				this->extract_val128(cmp, candidates, j, store_pairs, key, list_ids, dis0, k, heap_dis, heap_ids, qmax, bh_bound_sse);
			}
			//compare_extract_matches_sse(candidates, bh_bound_sse, scanned, max_scan,
			//		candidates_mem, bh, labels, labels_offset);

		}

	};
};





template<int TT_M>
struct VecProductQuantizer_4_SSE128_unsigned : public AbstractVecProductQuantizer<TT_M,2,4,4,0,0,uint8_t,__m128i,__m128i> {

	typedef typename AbstractVecProductQuantizer<TT_M,2,4,4,0,0,uint8_t,__m128i,__m128i>::group group;
	typedef typename AbstractVecProductQuantizer<TT_M,2,4,4,0,0,uint8_t,__m128i,__m128i>::VPQQuant VPQQuant;



	VecProductQuantizer_4_SSE128_unsigned(size_t d) : /* dimensionality of the input vectors */
		AbstractVecProductQuantizer<TT_M,2,4,4,0,0,uint8_t,__m128i,__m128i>(d) {
	}

	VecProductQuantizer_4_SSE128_unsigned() : /* dimensionality of the input vectors */
		AbstractVecProductQuantizer<TT_M,2,4,4,0,0,uint8_t,__m128i,__m128i>() {
	}

	//__attribute__((optimize("unroll-loops")))
	inline void lookup_and_update_heap_simd(size_t ncodes, size_t offset, const group * __restrict codes,
		                                                 const float * __restrict dis_table, __m128i *__restrict mm_dis_tables, VPQQuant * qmax,
														 int k, float * __restrict heap_dis, long* __restrict heap_ids, float dis0,
														 long key, const long* list_ids, bool store_pairs) const override {

	    const std::uint64_t low_mask = 0x0f0f0f0f0f0f0f0f;
	    const __m128i low_mask_sse = _mm_set_epi64x(low_mask, low_mask);

		// Access static member of super class (shortcut)
		constexpr int codes_per_group = AbstractVecProductQuantizer<TT_M,2,4,4,0,0,uint8_t,__m128i,__m256i>::codes_per_group;
		constexpr int lanes_per_code = AbstractVecProductQuantizer<TT_M,2,4,4,0,0,uint8_t,__m128i,__m256i>::lanes_per_code;

		// Binheap extraction
		uint8_t bh_bound_quant;
		qmax->quantize_sum(heap_dis[0]-dis0, &bh_bound_quant);
		__m128i bh_bound_sse = _mm_set1_epi8(bh_bound_quant);

		int start_group_index = offset / codes_per_group;
		int start_index_in_group = offset % codes_per_group;

		int last_group_index = (offset+ncodes-1)/codes_per_group;
		int last_index_in_group = (offset+ncodes-1) % codes_per_group;

		int start_mask=~((1 << start_index_in_group)-1);  //2 => 0b0000011 => 0b11111100;
		int end_mask=((1 <<(last_index_in_group+1)) -1); //2 => 3 => 0b0000111


		for (size_t j = start_group_index; j <= last_group_index; j++) {

			__m128i candidates = _mm_set1_epi8(1);

			// Subquantizers 0..SQ_COUNT
			for(int row_i = 0; row_i < lanes_per_code; ++row_i) {
				const int sq_i = row_i * 2;
				const __m128i comps = _mm_loadu_si128(&codes[j].mmxl[row_i]);
				// Low comps
				const __m128i comps_low = _mm_and_si128(comps, low_mask_sse);
				const __m128i partial_low = _mm_shuffle_epi8(mm_dis_tables[sq_i], comps_low);
				candidates = _mm_adds_epu8(candidates, partial_low);

				// High comps
				const __m128i comps_shift = _mm_srli_epi64(comps, 4);
				const __m128i comps_high = _mm_and_si128(comps_shift, low_mask_sse);
				const __m128i partial_high = _mm_shuffle_epi8(mm_dis_tables[sq_i + 1],comps_high);
				candidates = _mm_adds_epu8(candidates, partial_high);
			}

			// Compare
			const __m128i compare = _mm_cmplt_epu8(candidates, bh_bound_sse);
			int cmp = _mm_movemask_epi8(compare);

			// Apply masks for potentially incomplete first and last groups
			if(unlikely(j == start_group_index)){
				cmp &= start_mask;
			}
			if(unlikely(j == last_group_index)){
				cmp &= end_mask;
			}

			if(cmp){
				this->extract_val128(cmp, candidates, j, store_pairs, key, list_ids, dis0, k, heap_dis, heap_ids, qmax, bh_bound_sse);
			}
			//compare_extract_matches_sse(candidates, bh_bound_sse, scanned, max_scan,
			//		candidates_mem, bh, labels, labels_offset);

		}

	};
};




#ifdef __AVX512F__
template<int TT_M, int T_PG_1, int T_PG_2, int T_PG_3>
struct VecProductQuantizer_XYZ_AVX512 : public AbstractVecProductQuantizer<TT_M,3,T_PG_1,T_PG_2,T_PG_3,0,int16_t,__m512i,__m512i> {

	typedef typename AbstractVecProductQuantizer<TT_M,3,T_PG_1,T_PG_2,T_PG_3,0,int16_t,__m512i,__m512i>::group group;
	typedef typename AbstractVecProductQuantizer<TT_M,3,T_PG_1,T_PG_2,T_PG_3,0,int16_t,__m512i,__m512i>::VPQQuant VPQQuant;



	VecProductQuantizer_XYZ_AVX512(size_t d) : /* dimensionality of the input vectors */
		AbstractVecProductQuantizer<TT_M,3,T_PG_1,T_PG_2,T_PG_3,0,int16_t,__m512i,__m512i>(d) {
	}

	VecProductQuantizer_XYZ_AVX512() : /* dimensionality of the input vectors */
		AbstractVecProductQuantizer<TT_M,3,T_PG_1,T_PG_2,T_PG_3,0,int16_t,__m512i,__m512i>() {
	}

	//__attribute__((optimize("unroll-loops")))
	inline void lookup_and_update_heap_simd(size_t ncodes, size_t offset, const group * __restrict codes,
		                                                 const float * __restrict dis_table, __m512i *__restrict mm_dis_tables, VPQQuant * qmax,
														 int k, float * __restrict heap_dis, long* __restrict heap_ids, float dis0,
														 long key, const long* list_ids, bool store_pairs) const override {

		// Access static member of super class (shortcut)
		constexpr int codes_per_group = AbstractVecProductQuantizer<TT_M,3,T_PG_1,T_PG_2,T_PG_3,0,int16_t,__m512i,__m512i>::codes_per_group;
		constexpr int lanes_per_code = AbstractVecProductQuantizer<TT_M,3,T_PG_1,T_PG_2,T_PG_3,0,int16_t,__m512i,__m512i>::lanes_per_code;

		// Binheap extraction
		int16_t bh_bound_quant;
		qmax->quantize_sum(heap_dis[0]-dis0, &bh_bound_quant);
		__m512i bh_bound_av512 = _mm512_set1_epi16(bh_bound_quant);

		int start_group_index = offset / codes_per_group;
		int start_index_in_group = offset % codes_per_group;

		int last_group_index = (offset+ncodes-1)/codes_per_group;
		int last_index_in_group = (offset+ncodes-1) % codes_per_group;

		int start_mask=~((1 << start_index_in_group)-1);  //2 => 0b0000011 => 0b11111100;
		int end_mask=((1 <<(last_index_in_group+1)) -1); //2 => 3 => 0b0000111

		for (size_t j = start_group_index; j <= last_group_index; j++) {

			__m512i candidates = _mm512_setzero_si512();

			// Subquantizers 0..SQ_COUNT
			for(int row_i = 0; row_i < lanes_per_code; ++row_i) {
				const int offset_1 = 1 + (T_PG_1 == 6);
				const int offset_2 = 1 + (T_PG_2 == 6) + offset_1;
				const int offset_3 = 1 + (T_PG_3 == 6) + offset_2;
				const int sqdt_i = row_i * (offset_3);
				__m512i partiala, partialb, partialc;

				// Subquantizer 0
				const __m512i comps_012 = _mm512_loadu_si512(&codes[j].mmxl[row_i]);
				if(T_PG_1 == 6){
					partiala = _mm512_permutex2var_epi16(mm_dis_tables[sqdt_i+0], comps_012, mm_dis_tables[sqdt_i+1]);
				}else{
					partiala = _mm512_permutexvar_epi16(comps_012,mm_dis_tables[sqdt_i+0]);
				}
				candidates = _mm512_adds_epi16(candidates, partiala);

				// Subquantizer 1
				const __m512i comps_12 = _mm512_srli_epi16(comps_012, this->csub_offset_inlane[1]);
				if(T_PG_2 ==6){
					partialb = _mm512_permutex2var_epi16(mm_dis_tables[sqdt_i+offset_1], comps_12, mm_dis_tables[sqdt_i+offset_1+1]);
				}else{
					partialb = _mm512_permutexvar_epi16(comps_12,mm_dis_tables[sqdt_i+offset_1]);
				}
				candidates = _mm512_adds_epi16(candidates, partialb);

				// Subquantizer 2
				const __m512i comps_2 = _mm512_srli_epi16(comps_012, this->csub_offset_inlane[2]);
				if(T_PG_3 ==6){
					partialc = _mm512_permutex2var_epi16(mm_dis_tables[sqdt_i+offset_2], comps_2, mm_dis_tables[sqdt_i+offset_2+1]);
				}else{
					partialc = _mm512_permutexvar_epi16(comps_2,mm_dis_tables[sqdt_i+offset_2]);
				}
				candidates = _mm512_adds_epi16(candidates, partialc);
			}


			// Compare
			__mmask32 cmp = _mm512_cmplt_epi16_mask(candidates, bh_bound_av512);

			// Apply masks for potentially incomplete first and last groups
			if(unlikely(j == start_group_index)){
				cmp &= start_mask;
			}
			if(unlikely(j == last_group_index)){
				cmp &= end_mask;
			}

			if(cmp){
				this->extract_val_loop(cmp, candidates, j, store_pairs, key, list_ids, dis0, k, heap_dis, heap_ids, qmax, bh_bound_av512);
			}

		}

	};
};
#else
template<int TT_M, int T_PG_1, int T_PG_2, int T_PG_3>
struct VecProductQuantizer_XYZ_AVX512 : public VecProductQuantizer_NoVecTable<TT_M,3,T_PG_1,T_PG_2,T_PG_3,0,int16_t,__m512i,__m512i> {

	VecProductQuantizer_XYZ_AVX512(size_t d) : /* dimensionality of the input vectors */
		VecProductQuantizer_NoVecTable<TT_M,3,T_PG_1,T_PG_2,T_PG_3,0,int16_t,__m512i,__m512i>(d) {
	}

	VecProductQuantizer_XYZ_AVX512() : /* dimensionality of the input vectors */
		VecProductQuantizer_NoVecTable<TT_M,3,T_PG_1,T_PG_2,T_PG_3,0,int16_t,__m512i,__m512i>() {
	}

};

#endif



#ifdef __AVX512F__
template<int TT_M, int T_PG_1, int T_PG_2, int T_PG_3>
struct VecProductQuantizer_XYZ_AVX512_unsigned : public AbstractVecProductQuantizer<TT_M,3,T_PG_1,T_PG_2,T_PG_3,0,uint16_t,__m512i,__m512i> {

	typedef typename AbstractVecProductQuantizer<TT_M,3,T_PG_1,T_PG_2,T_PG_3,0,uint16_t,__m512i,__m512i>::group group;
	typedef typename AbstractVecProductQuantizer<TT_M,3,T_PG_1,T_PG_2,T_PG_3,0,uint16_t,__m512i,__m512i>::VPQQuant VPQQuant;



	VecProductQuantizer_XYZ_AVX512_unsigned(size_t d) : /* dimensionality of the input vectors */
		AbstractVecProductQuantizer<TT_M,3,T_PG_1,T_PG_2,T_PG_3,0,uint16_t,__m512i,__m512i>(d) {
	}

	VecProductQuantizer_XYZ_AVX512_unsigned() : /* dimensionality of the input vectors */
		AbstractVecProductQuantizer<TT_M,3,T_PG_1,T_PG_2,T_PG_3,0,uint16_t,__m512i,__m512i>() {
	}

	//__attribute__((optimize("unroll-loops")))
	inline void lookup_and_update_heap_simd(size_t ncodes, size_t offset, const group * __restrict codes,
		                                                 const float * __restrict dis_table, __m512i *__restrict mm_dis_tables, VPQQuant * qmax,
														 int k, float * __restrict heap_dis, long* __restrict heap_ids, float dis0,
														 long key, const long* list_ids, bool store_pairs) const override {

		// Access static member of super class (shortcut)
		constexpr int codes_per_group = AbstractVecProductQuantizer<TT_M,3,T_PG_1,T_PG_2,T_PG_3,0,uint16_t,__m512i,__m512i>::codes_per_group;
		constexpr int lanes_per_code = AbstractVecProductQuantizer<TT_M,3,T_PG_1,T_PG_2,T_PG_3,0,uint16_t,__m512i,__m512i>::lanes_per_code;

		// Binheap extraction
		uint16_t bh_bound_quant;
		qmax->quantize_sum(heap_dis[0]-dis0, &bh_bound_quant);
		__m512i bh_bound_av512 = _mm512_set1_epi16(bh_bound_quant);

		int start_group_index = offset / codes_per_group;
		int start_index_in_group = offset % codes_per_group;

		int last_group_index = (offset+ncodes-1)/codes_per_group;
		int last_index_in_group = (offset+ncodes-1) % codes_per_group;

		int start_mask=~((1 << start_index_in_group)-1);  //2 => 0b0000011 => 0b11111100;
		int end_mask=((1 <<(last_index_in_group+1)) -1); //2 => 3 => 0b0000111


		for (size_t j = start_group_index; j <= last_group_index; j++) {

			__m512i candidates = _mm512_set1_epi16(1);

			// Subquantizers 0..SQ_COUNT
			for(int row_i = 0; row_i < lanes_per_code; ++row_i) {
				const int offset_1 = 1 + (T_PG_1 == 6);
				const int offset_2 = 1 + (T_PG_2 == 6) + offset_1;
				const int offset_3 = 1 + (T_PG_3 == 6) + offset_2;
				const int sqdt_i = row_i * (offset_3);
				__m512i partiala, partialb, partialc;

				// Subquantizer 0
				const __m512i comps_012 = _mm512_loadu_si512(&codes[j].mmxl[row_i]);
				if(T_PG_1 == 6){
					partiala = _mm512_permutex2var_epi16(mm_dis_tables[sqdt_i+0], comps_012, mm_dis_tables[sqdt_i+1]);
				}else{
					partiala = _mm512_permutexvar_epi16(comps_012,mm_dis_tables[sqdt_i+0]);
				}
				candidates = _mm512_adds_epu16(candidates, partiala);

				// Subquantizer 1
				const __m512i comps_12 = _mm512_srli_epi16(comps_012, this->csub_offset_inlane[1]);
				if(T_PG_2 ==6){
					partialb = _mm512_permutex2var_epi16(mm_dis_tables[sqdt_i+offset_1], comps_12, mm_dis_tables[sqdt_i+offset_1+1]);
				}else{
					partialb = _mm512_permutexvar_epi16(comps_12,mm_dis_tables[sqdt_i+offset_1]);
				}
				candidates = _mm512_adds_epu16(candidates, partialb);

				// Subquantizer 2
				const __m512i comps_2 = _mm512_srli_epi16(comps_012, this->csub_offset_inlane[2]);
				if(T_PG_3 ==6){
					partialc = _mm512_permutex2var_epi16(mm_dis_tables[sqdt_i+offset_2], comps_2, mm_dis_tables[sqdt_i+offset_2+1]);
				}else{
					partialc = _mm512_permutexvar_epi16(comps_2,mm_dis_tables[sqdt_i+offset_2]);
				}
				candidates = _mm512_adds_epu16(candidates, partialc);
			}


			// Compare
			__mmask32 cmp = _mm512_cmplt_epu16_mask(candidates, bh_bound_av512);

			// Apply masks for potentially incomplete first and last groups
			if(unlikely(j == start_group_index)){
				cmp &= start_mask;
			}
			if(unlikely(j == last_group_index)){
				cmp &= end_mask;
			}

			if(cmp){
				this->extract_val_loop(cmp, candidates, j, store_pairs, key, list_ids, dis0, k, heap_dis, heap_ids, qmax, bh_bound_av512);
			}

		}

	};
};
#else
template<int TT_M, int T_PG_1, int T_PG_2, int T_PG_3>
struct VecProductQuantizer_XYZ_AVX512_unsigned : public VecProductQuantizer_NoVecTable<TT_M,3,T_PG_1,T_PG_2,T_PG_3,0,uint16_t,__m512i,__m512i> {

	VecProductQuantizer_XYZ_AVX512_unsigned(size_t d) : /* dimensionality of the input vectors */
		VecProductQuantizer_NoVecTable<TT_M,3,T_PG_1,T_PG_2,T_PG_3,0,uint16_t,__m512i,__m512i>(d) {
	}

	VecProductQuantizer_XYZ_AVX512_unsigned() : /* dimensionality of the input vectors */
		VecProductQuantizer_NoVecTable<TT_M,3,T_PG_1,T_PG_2,T_PG_3,0,uint16_t,__m512i,__m512i>() {
	}

};

#endif




#ifdef __AVX512F__
template<int TT_M>
struct VecProductQuantizer_88_AVX512_unsigned : public AbstractVecProductQuantizer<TT_M,2,8,8,0,0,uint16_t,__m512i,__m512i> {

	typedef typename AbstractVecProductQuantizer<TT_M,2,8,8,0,0,uint16_t,__m512i,__m512i>::group group;
	typedef typename AbstractVecProductQuantizer<TT_M,2,8,8,0,0,uint16_t,__m512i,__m512i>::VPQQuant VPQQuant;

	VecProductQuantizer_88_AVX512_unsigned(size_t d) : /* dimensionality of the input vectors */
		AbstractVecProductQuantizer<TT_M,2,8,8,0,0,uint16_t,__m512i,__m512i>(d) {
	}

	VecProductQuantizer_88_AVX512_unsigned() : /* dimensionality of the input vectors */
		AbstractVecProductQuantizer<TT_M,2,8,8,0,0,uint16_t,__m512i,__m512i>() {
	}

	//__attribute__((optimize("unroll-loops")))
	inline void lookup_and_update_heap_simd(size_t ncodes, size_t offset, const group * __restrict codes,
		                                                 const float * __restrict dis_table, __m512i *__restrict mm_dis_tables, VPQQuant * qmax,
														 int k, float * __restrict heap_dis, long* __restrict heap_ids, float dis0,
														 long key, const long* list_ids, bool store_pairs) const override {

		// Access static member of super class (shortcut)
		constexpr int codes_per_group = AbstractVecProductQuantizer<TT_M,2,8,8,0,0,uint16_t,__m512i,__m512i>::codes_per_group;
		constexpr int lanes_per_code = AbstractVecProductQuantizer<TT_M,2,8,8,0,0,uint16_t,__m512i,__m512i>::lanes_per_code;

		// Binheap extraction
		uint16_t bh_bound_quant;
		qmax->quantize_sum(heap_dis[0]-dis0, &bh_bound_quant);
		__m512i bh_bound_av512 = _mm512_set1_epi16(bh_bound_quant);

		int start_group_index = offset / codes_per_group;
		int start_index_in_group = offset % codes_per_group;

		int last_group_index = (offset+ncodes-1)/codes_per_group;
		int last_index_in_group = (offset+ncodes-1) % codes_per_group;

		int start_mask=~((1 << start_index_in_group)-1);  //2 => 0b0000011 => 0b11111100;
		int end_mask=((1 <<(last_index_in_group+1)) -1); //2 => 3 => 0b0000111


		for (size_t j = start_group_index; j <= last_group_index; j++) {

			__m512i candidates = _mm512_set1_epi16(1);

			// Subquantizers 0..SQ_COUNT
			for(int row_i = 0; row_i < lanes_per_code; ++row_i) {
				const int sqdt_i = row_i * 16;

				const __m512i comps_01 = _mm512_loadu_si512(&codes[j].mmxl[row_i]);

				/* First component */
				const __m512i bit7_0 = _mm512_slli_epi16(comps_01,9);
				const __mmask32 bit7_0_m = _mm512_movepi16_mask(bit7_0);
				const __m512i bit8_0 = _mm512_slli_epi16(comps_01,8);
				const __mmask32 bit8_0_m = _mm512_movepi16_mask(bit8_0);

				const __m512i partial_0_00 = _mm512_permutex2var_epi16(mm_dis_tables[sqdt_i+0], comps_01, mm_dis_tables[sqdt_i+1]);
				const __m512i partial_0_01 = _mm512_permutex2var_epi16(mm_dis_tables[sqdt_i+2], comps_01, mm_dis_tables[sqdt_i+3]);
				const __m512i partial_0_0 = _mm512_mask_blend_epi16(bit7_0_m,partial_0_00,partial_0_01);

				const __m512i partial_0_10 = _mm512_permutex2var_epi16(mm_dis_tables[sqdt_i+4], comps_01, mm_dis_tables[sqdt_i+5]);
				const __m512i partial_0_11 =_mm512_permutex2var_epi16(mm_dis_tables[sqdt_i+6], comps_01, mm_dis_tables[sqdt_i+7]);
				const __m512i partial_0_1 = _mm512_mask_blend_epi16(bit7_0_m,partial_0_10,partial_0_11);

				const __m512i partial_0 = _mm512_mask_blend_epi16(bit8_0_m,partial_0_0,partial_0_1);

				/* Second component */
				const __m512i comps_1 = _mm512_srli_epi16(comps_01, 8);
				const __m512i bit7_1 = _mm512_slli_epi16(comps_01,1);
				const __mmask32 bit7_1_m = _mm512_movepi16_mask(bit7_1);
				const __m512i bit8_1 = comps_01;
				const __mmask32 bit8_1_m = _mm512_movepi16_mask(bit8_1);

				const __m512i partial_1_00 = _mm512_permutex2var_epi16(mm_dis_tables[sqdt_i+8], comps_1, mm_dis_tables[sqdt_i+9]);
				const __m512i partial_1_01 = _mm512_permutex2var_epi16(mm_dis_tables[sqdt_i+10], comps_1, mm_dis_tables[sqdt_i+11]);
				const __m512i partial_1_0 = _mm512_mask_blend_epi16(bit7_1_m,partial_1_00,partial_1_01);

				const __m512i partial_1_10 = _mm512_permutex2var_epi16(mm_dis_tables[sqdt_i+12], comps_1, mm_dis_tables[sqdt_i+13]);
				const __m512i partial_1_11 =_mm512_permutex2var_epi16(mm_dis_tables[sqdt_i+14], comps_1, mm_dis_tables[sqdt_i+15]);
				const __m512i partial_1_1 = _mm512_mask_blend_epi16(bit7_1_m,partial_1_10,partial_1_11);

				const __m512i partial_1 = _mm512_mask_blend_epi16(bit8_1_m,partial_1_0,partial_1_1);

				const __m512i partial_sum = _mm512_adds_epu16(partial_0, partial_1);
				candidates = _mm512_adds_epu16(candidates, partial_sum);

			}


			// Compare
			__mmask32 cmp = _mm512_cmplt_epu16_mask(candidates, bh_bound_av512);

			// Apply masks for potentially incomplete first and last groups
			if(unlikely(j == start_group_index)){
				cmp &= start_mask;
			}
			if(unlikely(j == last_group_index)){
				cmp &= end_mask;
			}

			if(cmp){
				this->extract_val_loop(cmp, candidates, j, store_pairs, key, list_ids, dis0, k, heap_dis, heap_ids, qmax, bh_bound_av512);
			}

		}

	};
};
#else
template<int TT_M>
struct VecProductQuantizer_88_AVX512_unsigned : public VecProductQuantizer_NoVecTable<TT_M,2,8,8,0,0,uint16_t,__m512i,__m512i> {

	VecProductQuantizer_88_AVX512_unsigned(size_t d) : /* dimensionality of the input vectors */
		VecProductQuantizer_NoVecTable<TT_M,2,8,8,0,0,uint16_t,__m512i,__m512i>(d) {
	}

	VecProductQuantizer_88_AVX512_unsigned() : /* dimensionality of the input vectors */
		VecProductQuantizer_NoVecTable<TT_M,2,8,8,0,0,uint16_t,__m512i,__m512i>() {
	}

};

#endif





#ifdef __AVX512VBMI__
template<int TT_M>
struct VecProductQuantizer_8_AVX512_unsigned : public AbstractVecProductQuantizer<TT_M,1,8,0,0,0,uint8_t,__m512i,__m512i> {

	typedef typename AbstractVecProductQuantizer<TT_M,1,8,0,0,0,uint8_t,__m512i,__m512i>::group group;
	typedef typename AbstractVecProductQuantizer<TT_M,1,8,0,0,0,uint8_t,__m512i,__m512i>::VPQQuant VPQQuant;
	VecProductQuantizer_8_AVX512_unsigned(size_t d) : /* dimensionality of the input vectors */
		AbstractVecProductQuantizer<TT_M,1,8,0,0,0,uint8_t,__m512i,__m512i>(d) {
	}

	VecProductQuantizer_8_AVX512_unsigned() : /* dimensionality of the input vectors */
		AbstractVecProductQuantizer<TT_M,1,8,0,0,0,uint8_t,__m512i,__m512i>() {
	}

	//__attribute__((optimize("unroll-loops")))
	inline void lookup_and_update_heap_simd(size_t ncodes, size_t offset, const group * __restrict codes,
		                                                 const float * __restrict dis_table, __m512i *__restrict mm_dis_tables, VPQQuant * qmax,
														 int k, float * __restrict heap_dis, long* __restrict heap_ids, float dis0,
														 long key, const long* list_ids, bool store_pairs) const override {

		// Access static member of super class (shortcut)
		constexpr int codes_per_group = AbstractVecProductQuantizer<TT_M,1,8,0,0,0,uint8_t,__m512i,__m512i>::codes_per_group;
		constexpr int lanes_per_code = AbstractVecProductQuantizer<TT_M,1,8,0,0,0,uint8_t,__m512i,__m512i>::lanes_per_code;

		// Binheap extraction
		uint8_t bh_bound_quant;
		qmax->quantize_sum(heap_dis[0]-dis0, &bh_bound_quant);
		__m512i bh_bound_av512 = _mm512_set1_epi8(bh_bound_quant);

		int start_group_index = offset / codes_per_group;
		int start_index_in_group = offset % codes_per_group;

		int last_group_index = (offset+ncodes-1)/codes_per_group;
		int last_index_in_group = (offset+ncodes-1) % codes_per_group;

		int start_mask=~((1 << start_index_in_group)-1);  //2 => 0b0000011 => 0b11111100;
		int end_mask=((1 <<(last_index_in_group+1)) -1); //2 => 3 => 0b0000111


		for (size_t j = start_group_index; j <= last_group_index; j++) {

			__m512i candidates = _mm512_set1_epi8(1);

			// Subquantizers 0..SQ_COUNT
			for(int row_i = 0; row_i < lanes_per_code; ++row_i) {
				const int sqdt_i = row_i * 4;

				const __m512i comps = _mm512_loadu_si512(&codes[j].mmxl[row_i]);
				const __mmask64 bit8_m = _mm512_movepi8_mask(comps);

				const __m512i partial_0 = _mm512_permutex2var_epi8(mm_dis_tables[sqdt_i+0], comps, mm_dis_tables[sqdt_i+1]);
				const __m512i partial_1 = _mm512_permutex2var_epi8(mm_dis_tables[sqdt_i+2], comps, mm_dis_tables[sqdt_i+3]);
				const __m512i partial_sum = _mm512_mask_blend_epi8(bit8_m,partial_0,partial_1);
				candidates = _mm512_adds_epu8(candidates, partial_sum);

			}


			// Compare
			__mmask32 cmp = _mm512_cmplt_epu8_mask(candidates, bh_bound_av512);

			// Apply masks for potentially incomplete first and last groups
			if(unlikely(j == start_group_index)){
				cmp &= start_mask;
			}
			if(unlikely(j == last_group_index)){
				cmp &= end_mask;
			}

			if(cmp){
				this->extract_val_loop(cmp, candidates, j, store_pairs, key, list_ids, dis0, k, heap_dis, heap_ids, qmax, bh_bound_av512);
			}

		}

	};
};
#else
template<int TT_M>
struct VecProductQuantizer_8_AVX512_unsigned : public VecProductQuantizer_NoVecTable<TT_M,1,8,0,0,0,uint8_t,__m512i,__m512i> {

	VecProductQuantizer_8_AVX512_unsigned(size_t d) : /* dimensionality of the input vectors */
		VecProductQuantizer_NoVecTable<TT_M,1,8,0,0,0,uint8_t,__m512i,__m512i>(d) {
	}

	VecProductQuantizer_8_AVX512_unsigned() : /* dimensionality of the input vectors */
		VecProductQuantizer_NoVecTable<TT_M,1,8,0,0,0,uint8_t,__m512i,__m512i>() {
	}

};

#endif


/** Applies a rotation to align the dimensions with a PQ to minimize
 *  the reconstruction error. Can be used before an IndexPQ or an
 *  IndexIVFPQ. The method is the non-parametric version described in:
 *
 * "Optimized Product Quantization for Approximate Nearest Neighbor Search"
 * Tiezheng Ge, Kaiming He, Qifa Ke, Jian Sun, CVPR'13
 *
 */
template <class T_VPQ>
struct OVPQMatrix: LinearTransform {

#ifdef ECLIPSE
	typedef VecProductQuantizer_4_AVX256<16> T_VPQ;
#endif

    int niter;      ///< Number of outer training iterations
    int niter_pq;   ///< Number of training iterations for the PQ
    int niter_pq_0; ///< same, for the first outer iteration

    /// if there are too many training points, resample
    size_t max_train_points;
    bool verbose;

    /// if d2 != -1, output vectors of this dimension
    explicit OVPQMatrix (int d = 0, int d2 = -1) :
    		LinearTransform (d, d2 == -1 ? d : d2, false),
    	    niter (50),
    	    niter_pq (4), niter_pq_0 (40),
    	    verbose(false)
    {
    	    is_trained = false;
    	    // OPQ is quite expensive to train, so set this right.
    	    max_train_points = 256 * 256;
    }

    void train(Index::idx_t n, const float* x) override {
    	    const float * x_in = x;

    	    x = fvecs_maybe_subsample (d_in, (size_t*)&n,
    	                               max_train_points, x, verbose);

    	    ScopeDeleter<float> del_x (x != x_in ? x : nullptr);

    	    // To support d_out > d_in, we pad input vectors with 0s to d_out
    	    size_t d = d_out <= d_in ? d_in : d_out;
    	    size_t d2 = d_out;

    	#if 0
    	    // what this test shows: the only way of getting bit-exact
    	    // reproducible results with sgeqrf and sgesvd seems to be forcing
    	    // single-threading.
    	    { // test repro
    	        std::vector<float> r (d * d);
    	        float * rotation = r.data();
    	        float_randn (rotation, d * d, 1234);
    	        printf("CS0: %016lx\n",
    	               ivec_checksum (128*128, (int*)rotation));
    	        matrix_qr (d, d, rotation);
    	        printf("CS1: %016lx\n",
    	               ivec_checksum (128*128, (int*)rotation));
    	        return;
    	    }
    	#endif

    	    if (verbose) {
    	        printf ("OPQMatrix::train: training an OPQ rotation matrix "
    	                "from %ld vectors in %dD -> %dD\n",
    	                n, d_in, d_out);
    	    }

    	    std::vector<float> xtrain (n * d);
    	    // center x
    	    {
    	        std::vector<float> sum (d);
    	        const float *xi = x;
    	        for (size_t i = 0; i < n; i++) {
    	            for (int j = 0; j < d_in; j++)
    	                sum [j] += *xi++;
    	        }
    	        for (int i = 0; i < d; i++) sum[i] /= n;
    	        float *yi = xtrain.data();
    	        xi = x;
    	        for (size_t i = 0; i < n; i++) {
    	            for (int j = 0; j < d_in; j++)
    	                *yi++ = *xi++ - sum[j];
    	            yi += d - d_in;
    	        }
    	    }
    	    float *rotation;

    	    if (A.size () == 0) {
    	        A.resize (d * d);
    	        rotation = A.data();
    	        if (verbose)
    	            printf("  OPQMatrix::train: making random %ld*%ld rotation\n",
    	                   d, d);
    	        float_randn (rotation, d * d, 1234);
    	        matrix_qr (d, d, rotation);
    	        // we use only the d * d2 upper part of the matrix
    	        A.resize (d * d2);
    	    } else {
    	        FAISS_THROW_IF_NOT (A.size() == d * d2);
    	        rotation = A.data();
    	    }


    	    std::vector<float>
    	        xproj (d2 * n), pq_recons (d2 * n), xxr (d * n),
    	        tmp(d * d * 4);

    	    T_VPQ pq_regular (d2);

    	    std::vector<typename T_VPQ::group, boost::alignment::aligned_allocator<typename T_VPQ::group, 64>> codes (pq_regular.nb_groups(n));

    	    double t0 = getmillisecs();
    	    for (int iter = 0; iter < niter; iter++) {

    	        { // torch.mm(xtrain, rotation:t())
    	            FINTEGER di = d, d2i = d2, ni = n;
    	            float zero = 0, one = 1;
    	            sgemm_ ("Transposed", "Not transposed",
    	                    &d2i, &ni, &di,
    	                    &one, rotation, &di,
    	                    xtrain.data(), &di,
    	                    &zero, xproj.data(), &d2i);
    	        }

    	        pq_regular.cp.max_points_per_centroid = 1000;
    	        pq_regular.cp.niter = iter == 0 ? niter_pq_0 : niter_pq;
    	        pq_regular.cp.verbose = verbose;
    	        pq_regular.train (n, xproj.data());

    	        // Encoding/decoding is more expensive with VPQ due to the SIMD oriented layout.
    	        // FIXME : add a method to access directly the quantization error rather than doing it indirectly.
    	        pq_regular.encode_multiple (xproj.data(), codes.data(), 0, n);
    	        pq_regular.decode_multiple(codes.data(), pq_recons.data(), 0, n);

    	        float pq_err = fvec_L2sqr (pq_recons.data(), xproj.data(), n * d2) / n;

    	        if (verbose)
    	            printf ("    Iteration %d (%d PQ iterations):"
    	                    "%.3f s, obj=%g\n", iter, pq_regular.cp.niter,
    	                    (getmillisecs () - t0) / 1000.0, pq_err);

    	        {
    	            float *u = tmp.data(), *vt = &tmp [d * d];
    	            float *sing_val = &tmp [2 * d * d];
    	            FINTEGER di = d, d2i = d2, ni = n;
    	            float one = 1, zero = 0;

    	            // torch.mm(xtrain:t(), pq_recons)
    	             sgemm_ ("Not", "Transposed",
    	                    &d2i, &di, &ni,
    	                   &one, pq_recons.data(), &d2i,
    	                    xtrain.data(), &di,
    	                    &zero, xxr.data(), &d2i);


    	            FINTEGER lwork = -1, info = -1;
    	            float worksz;
    	            // workspace query
    	            sgesvd_ ("All", "All",
    	                     &d2i, &di, xxr.data(), &d2i,
    	                     sing_val,
    	                     vt, &d2i, u, &di,
    	                     &worksz, &lwork, &info);

    	            lwork = int(worksz);
    	            std::vector<float> work (lwork);
    	            // u and vt swapped
    	            sgesvd_ ("All", "All",
    	                     &d2i, &di, xxr.data(), &d2i,
    	                     sing_val,
    	                     vt, &d2i, u, &di,
    	                     work.data(), &lwork, &info);

    	            sgemm_ ("Transposed", "Transposed",
    	                    &di, &d2i, &d2i,
    	                    &one, u, &di, vt, &d2i,
    	                    &zero, rotation, &di);

    	        }
    	        pq_regular.train_type = T_VPQ::Train_hot_start;
    	    }

    	    // revert A matrix
    	    if (d > d_in) {
    	        for (long i = 0; i < d_out; i++)
    	            memmove (&A[i * d_in], &A[i * d], sizeof(A[0]) * d_in);
    	        A.resize (d_in * d_out);
    	    }

    	    is_trained = true;
    	    is_orthonormal = true;
    }
};

template <class T>
inline std::string tc_vpq(const T* n){return "_";}


} // namespace faiss



#endif
