#pragma once
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace htistft {
constexpr std::size_t kFft=4096,kRows=32,kColumns=48,kBlocks=86,kBatches=85;
constexpr std::size_t kAElements=kRows*kFft,kBElements=kFft*kColumns,kCElements=kRows*kColumns;
constexpr std::size_t kInput=16*2048*336,kOutput=8*344064,kGuard=16;
// A=[Re Z[0..2048], Im Z[1..2047]]. B reconstructs the Hermitian pair.
// The 1/sqrt(4096) factor is applied by HtIstftMerge, once only.
inline std::vector<float> MakePlan(std::size_t block) {
    if(block>=kBlocks)throw std::runtime_error("Invalid inverse DFT block");
    const double pi=std::acos(-1.0);std::vector<float> b(kBElements,0.0f);
    for(std::size_t j=0;j<kColumns;++j){
        const std::size_t n=block*kColumns+j;if(n>=kFft)continue;
        b[j]=1.0f;b[2048*kColumns+j]=n%2?-1.0f:1.0f;
        for(std::size_t k=1;k<2048;++k){
            const double phase=(2*pi/4096)*((k*n)%4096);
            b[k*kColumns+j]=static_cast<float>(2*std::cos(phase));
            b[(2048+k)*kColumns+j]=static_cast<float>(-2*std::sin(phase));
        }
    }
    return b;
}
}
