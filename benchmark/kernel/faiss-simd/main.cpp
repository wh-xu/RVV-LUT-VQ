#include <simdlib_avx2.h>
#include <iostream>

using namespace faiss;

int main() {
    const int nq = 1;
    const int bbs = 32;

    const int NQ = nq;
    const int BB = bbs/32;
    // const int nsq = 1;
    // const uint8_t* codes = nullptr;
    // const uint8_t* LUT = nullptr;
    // const NormTableScaler* scaler = nullptr;
    // ResultHandler res2;
    kernel_accumulate_block<NQ, BB>(nsq, codes, LUT, res2, scaler);

    
    return 0;
}