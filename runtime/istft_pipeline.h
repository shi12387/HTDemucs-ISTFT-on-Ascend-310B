#pragma once
#include "acl_resources.h"
#include "inverse_plan.h"

inline void Require(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
struct InverseTiming {
    double packMs=0,matmulMs=0,mergeMs=0,olaMs=0,cropMs=0,totalMs=0;
    std::size_t kernels=0,syncs=0;
};
class Pipeline {
public:
    Resources& r;Buffer spectrum,stats,a[2],b[2],raw[4],frames,window,numerator,envelope,cropped;
    aclTensor *packIn=nullptr,*statsT=nullptr,*packOut[2]={},*av[2][85]={},*bv[2][86]={},*cv[4][86]={},*rv[4]={},*merged[85]={};
    aclTensor *frameT=nullptr,*windowT=nullptr,*numT=nullptr,*envT=nullptr,*cropT=nullptr;
    explicit Pipeline(Resources& resources):r(resources){}
    void Prepare(const std::vector<float>& hann) {
        using namespace htistft;const auto start=Clock::now();Require(hann.size()==4096,"Wrong window size");
        spectrum=r.Allocate("spectrum",kInput*4);stats=r.Allocate("stats",8*4);
        packIn=r.View(spectrum,{1,16,2048,336},ACL_FLOAT);statsT=r.View(stats,{8},ACL_FLOAT);
        for(int p=0;p<2;++p){
            a[p]=r.Allocate("inverse_a",kBatches*kAElements*2);packOut[p]=r.View(a[p],{85,32,4096},ACL_FLOAT16);
            b[p]=r.Allocate("inverse_plan",kBlocks*kBElements*2);
            for(std::size_t batch=0;batch<kBatches;++batch)av[p][batch]=r.View(a[p],{32,4096},ACL_FLOAT16,batch*kAElements);
            for(std::size_t block=0;block<kBlocks;++block)bv[p][block]=r.View(b[p],{4096,48},ACL_FLOAT16,block*kBElements);
        }
        std::vector<aclFloat16> high(kBlocks*kBElements),low(high.size());
        for(std::size_t block=0;block<kBlocks;++block){
            const auto plan=MakePlan(block);
            for(std::size_t i=0;i<kBElements;++i){const auto j=block*kBElements+i;
                high[j]=aclFloatToFloat16(plan[i]);low[j]=aclFloatToFloat16(plan[i]-aclFloat16ToFloat(high[j]));}
        }
        r.Upload(b[0],high.data(),b[0].bytes);r.Upload(b[1],low.data(),b[1].bytes);
        for(int term=0;term<4;++term){
            raw[term]=r.Allocate("inverse_raw",kBlocks*kCElements*4);rv[term]=r.View(raw[term],{86,32,48},ACL_FLOAT);
            for(std::size_t block=0;block<kBlocks;++block)cv[term][block]=r.View(raw[term],{32,48},ACL_FLOAT,block*kCElements);
        }
        frames=r.Allocate("inverse_frames",8*340*4096*4);frameT=r.View(frames,{8,340,4096},ACL_FLOAT);
        for(std::size_t batch=0;batch<kBatches;++batch)merged[batch]=r.View(frames,{32,4096},ACL_FLOAT,batch*kAElements);
        window=r.Allocate("inverse_hann",4096*4);r.Upload(window,hann.data(),window.bytes);windowT=r.View(window,{4096},ACL_FLOAT);
        numerator=r.Allocate("ola_numerator",8*351232*4);envelope=r.Allocate("ola_envelope",351232*4);
        numT=r.View(numerator,{8,351232},ACL_FLOAT);envT=r.View(envelope,{351232},ACL_FLOAT);
        cropped=r.Allocate("istft_output",(kOutput+2*kGuard)*4);cropT=r.View(cropped,{8,344064},ACL_FLOAT,kGuard);
        r.Workspace(2097152);std::size_t bytes=0;for(const auto& v:r.buffers)bytes+=v.bytes;
        std::cout<<"[DEVICE CACHE] constant_uploads="<<r.copies.constants<<", constant_bytes="<<r.copies.constantBytes
                 <<", allocated_buffer_bytes="<<bytes<<", tensor_views="<<r.tensors.size()<<", preparation_wall_ms="<<Milliseconds(start)<<'\n';
        std::cout<<"[ISTFT] input=[1,16,2048,336], stats=[8], output=[8,344064], crop_start=3584\n";
    }
    InverseTiming Run(const std::string& tag) {
        Require(r.inChain&&!r.pending,"Invalid inverse chain state");InverseTiming t;
        const auto copies=r.copyOps,syncs=r.syncCount;const auto total=Clock::now();auto start=Clock::now();
        uint64_t size=0;aclOpExecutor* ex=nullptr;
#define HT_LAUNCH(OP,...) do{size=0;ex=nullptr;CHECK_ACL(OP##GetWorkspaceSize(__VA_ARGS__,&size,&ex));Require(ex!=nullptr,"Null executor: " #OP);void* ws=r.Workspace(size);r.pending=true;CHECK_ACL(OP(ws,size,ex,r.stream));++t.kernels;}while(0)
        HT_LAUNCH(aclnnHtDecoderPackSplit,packIn,statsT,packOut[0],packOut[1]);r.Sync();t.packMs=Milliseconds(start);
        for(std::size_t batch=0;batch<85;++batch){start=Clock::now();
            for(std::size_t block=0;block<86;++block)for(int term=0;term<4;++term){
                HT_LAUNCH(aclnnHtDftMatmul,av[term/2][batch],bv[term%2][block],cv[term][block]);
            }
            r.Sync();t.matmulMs+=Milliseconds(start);start=Clock::now();
            HT_LAUNCH(aclnnHtIstftMerge,rv[0],rv[1],rv[2],rv[3],merged[batch]);r.Sync();t.mergeMs+=Milliseconds(start);
        }
        start=Clock::now();HT_LAUNCH(aclnnHtIstftWindowOla,frameT,windowT,numT,envT);r.Sync();t.olaMs=Milliseconds(start);
        start=Clock::now();HT_LAUNCH(aclnnHtIstftCompensateCrop,numT,envT,cropT);r.Sync();t.cropMs=Milliseconds(start);
#undef HT_LAUNCH
        t.totalMs=Milliseconds(total);t.syncs=r.syncCount-syncs;
        Require(!r.pending&&t.kernels==29328&&t.syncs==173&&r.copyOps==copies,"Inverse call/copy accounting failed");
        std::cout<<"[ISTFT DEVICE] tag="<<tag<<", pipeline_wall_ms="<<t.totalMs<<", kernel_launches="<<t.kernels
                 <<", syncs="<<t.syncs<<", intermediate_host_copies=0, intermediate_readbacks=0\n";
        return t;
    }
};
