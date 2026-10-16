#include "kernel_operator.h"
// Fixed Decoder CaC -> eight-channel, 340-frame inverse operands.
// Same high/residual Cast sequence as passed Stage9H. Layout adapter is new.
class KernelHtDecoderPackSplit {
public:
    __aicore__ inline KernelHtDecoderPackSplit() {}
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR stats,GM_ADDR high,GM_ADDR low) {
        x_.SetGlobalBuffer((__gm__ float*)x,16*2048*336);
        stats_.SetGlobalBuffer((__gm__ float*)stats,8);
        high_.SetGlobalBuffer((__gm__ half*)high,8*340*4096);
        low_.SetGlobalBuffer((__gm__ half*)low,8*340*4096);
        pipe_.InitBuffer(packed_,4096*sizeof(float));pipe_.InitBuffer(expanded_,4096*sizeof(float));
        pipe_.InitBuffer(statsQ_,1,8*sizeof(float));
        pipe_.InitBuffer(highQ_,1,4096*sizeof(half));pipe_.InitBuffer(lowQ_,1,4096*sizeof(half));
        AscendC::DataCacheCleanAndInvalid<float,AscendC::CacheLine::ENTIRE_DATA_CACHE,
            AscendC::DcciDst::CACHELINE_OUT>(x_);
    }
    __aicore__ inline void Process() {
        auto s=statsQ_.AllocTensor<float>();AscendC::DataCopy(s,stats_,8);statsQ_.EnQue(s);s=statsQ_.DeQue<float>();
        Event<AscendC::HardEvent::MTE2_S>();const float mean=s.GetValue(0),std=s.GetValue(1);statsQ_.FreeTensor(s);
        auto p=packed_.Get<float>();auto expanded=expanded_.Get<float>();
        for(uint32_t channel=0;channel<8;++channel)for(uint32_t frame=0;frame<340;++frame) {
            if(frame>=2 && frame<338) {
                const uint32_t t=frame-2,realPlane=2*channel,imagPlane=realPlane+1;
                for(uint32_t k=0;k<2048;++k)p.SetValue(k,x_.GetValue((realPlane*2048+k)*336+t));
                p.SetValue(2048,0.0f);
                for(uint32_t k=1;k<2048;++k)p.SetValue(2048+k,x_.GetValue((imagPlane*2048+k)*336+t));
                Event<AscendC::HardEvent::S_V>();
                AscendC::Muls(p,p,std,4096);AscendC::PipeBarrier<PIPE_V>();
                AscendC::Adds(p,p,mean,4096);AscendC::PipeBarrier<PIPE_V>();
                Event<AscendC::HardEvent::V_S>();p.SetValue(2048,0.0f);Event<AscendC::HardEvent::S_V>();
            } else {AscendC::Duplicate(p,0.0f,4096);AscendC::PipeBarrier<PIPE_V>();}
            auto h=highQ_.AllocTensor<half>();auto l=lowQ_.AllocTensor<half>();
            AscendC::Cast(h,p,AscendC::RoundMode::CAST_NONE,4096);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(expanded,h,AscendC::RoundMode::CAST_NONE,4096);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(p,p,expanded,4096);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(l,p,AscendC::RoundMode::CAST_NONE,4096);AscendC::PipeBarrier<PIPE_V>();
            highQ_.EnQue(h);lowQ_.EnQue(l);h=highQ_.DeQue<half>();l=lowQ_.DeQue<half>();
            const uint32_t offset=(channel*340+frame)*4096;
            AscendC::DataCopy(high_[offset],h,4096);AscendC::DataCopy(low_[offset],l,4096);
            highQ_.FreeTensor(h);lowQ_.FreeTensor(l);
            // Scalar stores in the next iteration must follow all prior vector reads.
            Event<AscendC::HardEvent::V_S>();
        }
    }
private:
    template<AscendC::HardEvent E>__aicore__ inline void Event(){auto e=pipe_.FetchEventID(E);AscendC::SetFlag<E>(e);AscendC::WaitFlag<E>(e);}
    AscendC::TPipe pipe_;AscendC::TBuf<AscendC::QuePosition::VECCALC> packed_,expanded_;
    AscendC::TQue<AscendC::QuePosition::VECIN,1> statsQ_;
    AscendC::TQue<AscendC::QuePosition::VECOUT,1> highQ_,lowQ_;
    AscendC::GlobalTensor<float> x_,stats_;AscendC::GlobalTensor<half> high_,low_;
};
extern "C" __global__ __aicore__ void ht_decoder_pack_split(GM_ADDR x,GM_ADDR stats,GM_ADDR high,GM_ADDR low,GM_ADDR workspace,GM_ADDR tiling) {
    GET_TILING_DATA(td,tiling);if(TILING_KEY_IS(1)){KernelHtDecoderPackSplit op;op.Init(x,stats,high,low);op.Process();}
}
