#include "kernel_operator.h"

class KernelHtIstftWindowOla {
public:
    __aicore__ inline KernelHtIstftWindowOla() {}
    __aicore__ inline void Init(GM_ADDR frames,GM_ADDR window,GM_ADDR numerator,GM_ADDR envelope,
                               uint32_t channels,uint32_t frameCount,uint32_t nfft,uint32_t hop,uint32_t length) {
        channels_=channels;frameCount_=frameCount;nfft_=nfft;hop_=hop;length_=length;
        framesGm_.SetGlobalBuffer((__gm__ float*)frames,channels_*frameCount_*nfft_);
        windowGm_.SetGlobalBuffer((__gm__ float*)window,nfft_);
        numeratorGm_.SetGlobalBuffer((__gm__ float*)numerator,channels_*length_);
        envelopeGm_.SetGlobalBuffer((__gm__ float*)envelope,length_);
        // 2 input tiles + 1 output tile + 1 product tile = 16 KiB UB.
        pipe_.InitBuffer(framesQ_,1,hop_*sizeof(float));
        pipe_.InitBuffer(windowQ_,1,hop_*sizeof(float));
        pipe_.InitBuffer(outQ_,1,hop_*sizeof(float));
        pipe_.InitBuffer(productBuf_,hop_*sizeof(float));
    }
    __aicore__ inline void Process() {
        // Output-owned tiles: every GM output position is written exactly once.
        // Contributor order matches a CPU loop over frames in ascending order.
        auto product=productBuf_.Get<float>();
        const uint32_t tiles=length_/hop_;
        for (uint32_t tile=0;tile<tiles;++tile) {
            const uint32_t first=tile>=3?tile-3:0;
            const uint32_t last=tile<frameCount_?tile:frameCount_-1;
            auto out=outQ_.AllocTensor<float>();
            AscendC::Duplicate(out,0.0f,hop_);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t frame=first;frame<=last;++frame) {
                const uint32_t local=(tile-frame)*hop_;
                auto window=windowQ_.AllocTensor<float>();
                AscendC::DataCopy(window,windowGm_[local],hop_);
                windowQ_.EnQue(window);window=windowQ_.DeQue<float>();
                AscendC::Mul(product,window,window,hop_);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(out,out,product,hop_);
                AscendC::PipeBarrier<PIPE_V>();
                windowQ_.FreeTensor(window);
            }
            outQ_.EnQue(out);out=outQ_.DeQue<float>();
            AscendC::DataCopy(envelopeGm_[tile*hop_],out,hop_);
            outQ_.FreeTensor(out);
        }
        for (uint32_t channel=0;channel<channels_;++channel) {
            for (uint32_t tile=0;tile<tiles;++tile) {
                const uint32_t first=tile>=3?tile-3:0;
                const uint32_t last=tile<frameCount_?tile:frameCount_-1;
                auto out=outQ_.AllocTensor<float>();
                AscendC::Duplicate(out,0.0f,hop_);
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t frame=first;frame<=last;++frame) {
                    const uint32_t local=(tile-frame)*hop_;
                    const uint32_t source=(channel*frameCount_+frame)*nfft_+local;
                    auto x=framesQ_.AllocTensor<float>();auto window=windowQ_.AllocTensor<float>();
                    AscendC::DataCopy(x,framesGm_[source],hop_);framesQ_.EnQue(x);
                    AscendC::DataCopy(window,windowGm_[local],hop_);windowQ_.EnQue(window);
                    x=framesQ_.DeQue<float>();window=windowQ_.DeQue<float>();
                    AscendC::Mul(product,x,window,hop_);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(out,out,product,hop_);
                    AscendC::PipeBarrier<PIPE_V>();
                    framesQ_.FreeTensor(x);windowQ_.FreeTensor(window);
                }
                outQ_.EnQue(out);out=outQ_.DeQue<float>();
                AscendC::DataCopy(numeratorGm_[channel*length_+tile*hop_],out,hop_);
                outQ_.FreeTensor(out);
            }
        }
    }
private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN,1> framesQ_,windowQ_;
    AscendC::TQue<AscendC::QuePosition::VECOUT,1> outQ_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> productBuf_;
    AscendC::GlobalTensor<float> framesGm_,windowGm_,numeratorGm_,envelopeGm_;
    uint32_t channels_,frameCount_,nfft_,hop_,length_;
};

extern "C" __global__ __aicore__ void ht_istft_window_ola(
    GM_ADDR frames,GM_ADDR window,GM_ADDR numerator,GM_ADDR envelope,GM_ADDR workspace,GM_ADDR tiling) {
    GET_TILING_DATA(tilingData,tiling);
    if (TILING_KEY_IS(1)) {
        KernelHtIstftWindowOla op;
        op.Init(frames,window,numerator,envelope,tilingData.channels,tilingData.frameCount,
                tilingData.nfft,tilingData.hop,tilingData.length);
        op.Process();
    }
}
