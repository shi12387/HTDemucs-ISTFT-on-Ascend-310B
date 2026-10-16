#include "kernel_operator.h"

class KernelHtIstftMerge {
public:
    __aicore__ inline KernelHtIstftMerge() {}
    __aicore__ inline void Init(GM_ADDR hh,GM_ADDR hl,GM_ADDR lh,GM_ADDR ll,GM_ADDR y,
                               uint32_t blocks,uint32_t frames,uint32_t nfft) {
        blocks_=blocks;frames_=frames;nfft_=nfft;
        const uint32_t count=blocks_*frames_*48;
        hhGm_.SetGlobalBuffer((__gm__ float*)hh,count);
        hlGm_.SetGlobalBuffer((__gm__ float*)hl,count);
        lhGm_.SetGlobalBuffer((__gm__ float*)lh,count);
        llGm_.SetGlobalBuffer((__gm__ float*)ll,count);
        yGm_.SetGlobalBuffer((__gm__ float*)y,frames_*nfft_);
        // Four 32x48 input tiles plus one output tile: 30 KiB UB.
        pipe_.InitBuffer(hhQ_,1,1536*sizeof(float));
        pipe_.InitBuffer(hlQ_,1,1536*sizeof(float));
        pipe_.InitBuffer(lhQ_,1,1536*sizeof(float));
        pipe_.InitBuffer(llQ_,1,1536*sizeof(float));
        pipe_.InitBuffer(outQ_,1,1536*sizeof(float));
    }
    __aicore__ inline void Process() {
        for (uint32_t block=0;block<blocks_;++block) {
            const uint32_t offset=block*frames_*48;
            auto hh=hhQ_.AllocTensor<float>();auto hl=hlQ_.AllocTensor<float>();
            auto lh=lhQ_.AllocTensor<float>();auto ll=llQ_.AllocTensor<float>();
            AscendC::DataCopy(hh,hhGm_[offset],1536);hhQ_.EnQue(hh);
            AscendC::DataCopy(hl,hlGm_[offset],1536);hlQ_.EnQue(hl);
            AscendC::DataCopy(lh,lhGm_[offset],1536);lhQ_.EnQue(lh);
            AscendC::DataCopy(ll,llGm_[offset],1536);llQ_.EnQue(ll);
            hh=hhQ_.DeQue<float>();hl=hlQ_.DeQue<float>();
            lh=lhQ_.DeQue<float>();ll=llQ_.DeQue<float>();
            auto out=outQ_.AllocTensor<float>();
            // Match stage9A CPU parentheses and FP32 operation boundaries.
            AscendC::Add(out,hh,hl,1536);
            AscendC::Add(lh,lh,ll,1536);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(out,out,lh,1536);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(out,out,1.0f/64.0f,1536);
            AscendC::PipeBarrier<PIPE_V>();
            hhQ_.FreeTensor(hh);hlQ_.FreeTensor(hl);
            lhQ_.FreeTensor(lh);llQ_.FreeTensor(ll);
            outQ_.EnQue(out);out=outQ_.DeQue<float>();
            const uint32_t start=block*48;
            const uint32_t remaining=nfft_-start;
            const uint32_t valid=remaining<48?remaining:48;
            // 48 floats normally; last block writes 16. Both are 32-byte aligned.
            // Scatter tiles [timeBlock,frame,48] into contiguous [frame,4096].
            for (uint32_t frame=0;frame<frames_;++frame)
                AscendC::DataCopy(yGm_[frame*nfft_+start],out[frame*48],valid);
            outQ_.FreeTensor(out);
        }
    }
private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN,1> hhQ_,hlQ_,lhQ_,llQ_;
    AscendC::TQue<AscendC::QuePosition::VECOUT,1> outQ_;
    AscendC::GlobalTensor<float> hhGm_,hlGm_,lhGm_,llGm_,yGm_;
    uint32_t blocks_,frames_,nfft_;
};

extern "C" __global__ __aicore__ void ht_istft_merge(
    GM_ADDR hh,GM_ADDR hl,GM_ADDR lh,GM_ADDR ll,GM_ADDR y,GM_ADDR workspace,GM_ADDR tiling) {
    GET_TILING_DATA(tilingData,tiling);
    if (TILING_KEY_IS(1)) {
        KernelHtIstftMerge op;
        op.Init(hh,hl,lh,ll,y,tilingData.timeBlocks,tilingData.frames,tilingData.nfft);
        op.Process();
    }
}
