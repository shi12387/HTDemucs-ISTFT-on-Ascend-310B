#include "kernel_operator.h"

class KernelHtIstftCompensateCrop {
public:
    __aicore__ inline KernelHtIstftCompensateCrop() {}
    __aicore__ inline void Init(GM_ADDR numerator,GM_ADDR envelope,GM_ADDR y,uint32_t channels,
                               uint32_t inputLength,uint32_t outputLength,uint32_t cropStart,uint32_t tile) {
        channels_=channels;inputLength_=inputLength;outputLength_=outputLength;cropStart_=cropStart;tile_=tile;
        numeratorGm_.SetGlobalBuffer((__gm__ float*)numerator,channels_*inputLength_);
        envelopeGm_.SetGlobalBuffer((__gm__ float*)envelope,inputLength_);
        yGm_.SetGlobalBuffer((__gm__ float*)y,channels_*outputLength_);
        // Two inputs and one output, each 1024 FP32: 12 KiB UB.
        pipe_.InitBuffer(numeratorQ_,1,tile_*sizeof(float));
        pipe_.InitBuffer(envelopeQ_,1,tile_*sizeof(float));
        pipe_.InitBuffer(outQ_,1,tile_*sizeof(float));
    }
    __aicore__ inline void Process() {
        for (uint32_t channel=0;channel<channels_;++channel) {
            for (uint32_t offset=0;offset<outputLength_;offset+=tile_) {
                const uint32_t source=cropStart_+offset;
                auto numerator=numeratorQ_.AllocTensor<float>();
                auto envelope=envelopeQ_.AllocTensor<float>();auto out=outQ_.AllocTensor<float>();
                AscendC::DataCopy(numerator,numeratorGm_[channel*inputLength_+source],tile_);
                numeratorQ_.EnQue(numerator);
                AscendC::DataCopy(envelope,envelopeGm_[source],tile_);
                envelopeQ_.EnQue(envelope);
                numerator=numeratorQ_.DeQue<float>();envelope=envelopeQ_.DeQue<float>();
                // Contract: selected envelope values are finite and >1e-11.
                // No epsilon/clamp and no second inverse-DFT /64 normalization.
                AscendC::Div(out,numerator,envelope,tile_);
                AscendC::PipeBarrier<PIPE_V>();
                outQ_.EnQue(out);out=outQ_.DeQue<float>();
                AscendC::DataCopy(yGm_[channel*outputLength_+offset],out,tile_);
                numeratorQ_.FreeTensor(numerator);envelopeQ_.FreeTensor(envelope);outQ_.FreeTensor(out);
            }
        }
    }
private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN,1> numeratorQ_,envelopeQ_;
    AscendC::TQue<AscendC::QuePosition::VECOUT,1> outQ_;
    AscendC::GlobalTensor<float> numeratorGm_,envelopeGm_,yGm_;
    uint32_t channels_,inputLength_,outputLength_,cropStart_,tile_;
};
extern "C" __global__ __aicore__ void ht_istft_compensate_crop(
    GM_ADDR numerator,GM_ADDR envelope,GM_ADDR y,GM_ADDR workspace,GM_ADDR tiling) {
    GET_TILING_DATA(tilingData,tiling);
    if (TILING_KEY_IS(1)) {
        KernelHtIstftCompensateCrop op;
        op.Init(numerator,envelope,y,tilingData.channels,tilingData.inputLength,
                tilingData.outputLength,tilingData.cropStart,tilingData.tileLength);
        op.Process();
    }
}
