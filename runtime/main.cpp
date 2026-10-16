#include "istft_pipeline.h"
#include <sstream>
static std::string Json(const std::string& s){
    std::ostringstream f;f<<'"';for(unsigned char c:s){if(c=='"'||c=='\\')f<<'\\'<<c;else if(c=='\n')f<<"\\n";else if(c<32)f<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<unsigned(c)<<std::dec;else f<<c;}f<<'"';return f.str();
}
struct CaseTiming {std::string name;bool warmup=false;double reset=0,input=0,read=0,wall=0;InverseTiming compute;};
int main(int argc,char** argv){
    try{
        using namespace htistft;Require(argc==4,"Usage: istft_test DEVICE DATA_DIR RESULT_DIR");
        const int device=std::stoi(argv[1]);Require(device>=0,"Invalid device");const std::string data=argv[2],out=argv[3];
        std::cout<<std::setprecision(12);Resources r;r.device=device;r.Initialize();Pipeline pipeline(r);
        pipeline.Prepare(LoadFloat(data+"/window_hann_f32.bin",4096));
        std::vector<float> sentinel(kOutput+2*kGuard,std::numeric_limits<float>::quiet_NaN());
        std::fill(sentinel.begin(),sentinel.begin()+kGuard,456789.0f);std::fill(sentinel.end()-kGuard,sentinel.end(),-456789.0f);
        const std::vector<std::string> names={"tones_warmup","impulse","tones","noise","zero","tones_restore"};
        std::vector<CaseTiming> records;
        for(std::size_t i=0;i<names.size();++i){
            CaseTiming c;c.name=names[i];c.warmup=i==0;
            const auto x=LoadFloat(data+"/"+c.name+"_input_f32.bin",kInput);
            const auto stats=LoadFloat(data+"/"+c.name+"_stats_f32.bin",8);
            Require(stats[1]>0,"Nonpositive standard deviation");
            auto start=Clock::now();CHECK_ACL(aclrtMemcpy(pipeline.cropped.ptr,pipeline.cropped.bytes,sentinel.data(),pipeline.cropped.bytes,r.toDevice));++r.copyOps;c.reset=Milliseconds(start);
            const auto wall=Clock::now();start=Clock::now();r.Upload(pipeline.spectrum,x.data(),pipeline.spectrum.bytes);
            r.Upload(pipeline.stats,stats.data(),pipeline.stats.bytes);c.input=Milliseconds(start);
            const auto before=r.copyOps;r.inChain=true;c.compute=pipeline.Run(c.name);r.inChain=false;
            Require(!r.pending&&r.copyOps==before,"Application copy inside live iSTFT chain");++r.completedChains;
            start=Clock::now();const auto guarded=r.Download(pipeline.cropped);c.read=Milliseconds(start);c.wall=Milliseconds(wall);
            for(std::size_t j=0;j<kGuard;++j)Require(guarded[j]==456789.0f&&guarded[kGuard+kOutput+j]==-456789.0f,"Output guard changed");
            std::vector<float> y(guarded.begin()+kGuard,guarded.begin()+kGuard+kOutput);
            for(float v:y)Require(std::isfinite(v),"Unwritten sentinel/nonfinite output: "+c.name);
            SaveFloat(out+"/"+c.name+"_npu_f32.bin",y);records.push_back(c);
            std::cout<<"[CASE COMPLETE] "<<c.name<<", wall_ms="<<c.wall<<", pack_ms="<<c.compute.packMs
                     <<", matmul_ms="<<c.compute.matmulMs<<", merge_ms="<<c.compute.mergeMs
                     <<", ola_ms="<<c.compute.olaMs<<", crop_ms="<<c.compute.cropMs<<", guard_mismatch_count=0\n";
        }
        std::ofstream f((out+"/run_report.json").c_str());Require(bool(f),"Cannot write run report");
        f<<std::setprecision(12)<<"{\"status\":\"RUN_COMPLETE_REQUIRES_NUMERICAL_VALIDATION\",\"device\":"<<device
         <<",\"constant_uploads\":"<<r.copies.constants<<",\"constant_bytes\":"<<r.copies.constantBytes
         <<",\"workspace_bytes\":"<<r.workspaceCapacity<<",\"runs\":[";
        for(std::size_t i=0;i<records.size();++i){if(i)f<<',';const auto& c=records[i];const auto& t=c.compute;
            f<<"{\"name\":"<<Json(c.name)<<",\"warmup\":"<<(c.warmup?"true":"false")
             <<",\"sentinel_reset_ms\":"<<c.reset<<",\"input_transfer_ms\":"<<c.input
             <<",\"pack_ms\":"<<t.packMs<<",\"matmul_ms\":"<<t.matmulMs<<",\"merge_ms\":"<<t.mergeMs
             <<",\"ola_ms\":"<<t.olaMs<<",\"crop_ms\":"<<t.cropMs<<",\"device_chain_ms\":"<<t.totalMs
             <<",\"output_transfer_ms\":"<<c.read<<",\"segment_wall_ms\":"<<c.wall
             <<",\"kernel_launches\":"<<t.kernels<<",\"syncs\":"<<t.syncs
             <<",\"intermediate_application_copies\":0,\"guard_mismatch_count\":0,\"input_bytes\":"<<(pipeline.spectrum.bytes+pipeline.stats.bytes)
             <<",\"output_bytes\":"<<(kOutput*4)<<'}';
        }
        f<<"]}\n";f.close();Require(bool(f),"Cannot close run report");
        std::cout<<"COMPLETE: six cases, reused constants/device buffers; numerical validation required.\n";return 0;
    }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
