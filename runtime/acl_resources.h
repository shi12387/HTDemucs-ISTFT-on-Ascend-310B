#pragma once
#include "acl/acl.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include "aclnn_ht_decoder_pack_split.h"
extern "C" uint32_t HtIstftStandalonePreload_HtDecoderPackSplit();
#include "aclnn_ht_dft_matmul.h"
extern "C" uint32_t HtIstftStandalonePreload_HtDftMatmul();
#include "aclnn_ht_istft_merge.h"
extern "C" uint32_t HtIstftStandalonePreload_HtIstftMerge();
#include "aclnn_ht_istft_window_ola.h"
extern "C" uint32_t HtIstftStandalonePreload_HtIstftWindowOla();
#include "aclnn_ht_istft_compensate_crop.h"
extern "C" uint32_t HtIstftStandalonePreload_HtIstftCompensateCrop();
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <cerrno>

using Clock=std::chrono::steady_clock;
static double Milliseconds(Clock::time_point start) {return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
static void Check(std::int64_t code,const char* operation) {
    if (!code) return;
    std::string message=std::string(operation)+" failed, code="+std::to_string(code);
    if (const char* detail=aclGetRecentErrMsg()) message+=std::string("\nACL detail: ")+detail;
    throw std::runtime_error(message);
}
#define CHECK_ACL(call) Check((call),#call)

static std::vector<float> LoadFloat(const std::string& path,std::size_t count) {
    std::ifstream f(path.c_str(),std::ios::binary|std::ios::ate);
    if (!f || f.tellg()!=static_cast<std::streamoff>(count*sizeof(float))) throw std::runtime_error("Wrong FP32 input size: "+path);
    f.seekg(0);std::vector<float> x(count);
    if (!f.read(reinterpret_cast<char*>(x.data()),count*sizeof(float))) throw std::runtime_error("Cannot read "+path);
    for (float v:x) if (!std::isfinite(v)) throw std::runtime_error("Nonfinite data: "+path);
    return x;
}
static void SaveFloat(const std::string& path,const std::vector<float>& x) {
    std::ofstream f(path.c_str(),std::ios::binary|std::ios::trunc);
    if (!f || !f.write(reinterpret_cast<const char*>(x.data()),x.size()*sizeof(float))) throw std::runtime_error("Cannot save "+path);
}
struct Buffer {void* ptr=nullptr;std::size_t bytes=0;std::string role;};
struct CopyCounts {std::size_t constants=0,input=0,output=0,constantBytes=0,inputBytes=0,outputBytes=0;};

struct Resources {
    int device=0;bool initialized=false,deviceSet=false,pending=false,inChain=false,outputReady=false;
    std::size_t completedChains=0,syncCount=0,copyOps=0;CopyCounts copies;
    aclrtContext context=nullptr;aclrtStream stream=nullptr;
    aclrtMemcpyKind toDevice=ACL_MEMCPY_HOST_TO_DEVICE,fromDevice=ACL_MEMCPY_DEVICE_TO_HOST;
    std::vector<Buffer> buffers;std::vector<aclTensor*> tensors;void* workspace=nullptr;uint64_t workspaceCapacity=0;
    ~Resources(){if(pending&&stream)(void)aclrtSynchronizeStream(stream);
        for(auto* t:tensors)if(t)(void)aclDestroyTensor(t);
        if(workspace)(void)aclrtFree(workspace);
        for(auto b:buffers)if(b.ptr)(void)aclrtFree(b.ptr);
        if(stream)(void)aclrtDestroyStream(stream);
        if(context)(void)aclrtDestroyContext(context);
        if(deviceSet)(void)aclrtResetDevice(device);
        if(initialized)(void)aclFinalize();}
    void Initialize(){CHECK_ACL(aclInit(nullptr));initialized=true;CHECK_ACL(aclrtSetDevice(device));deviceSet=true;
        CHECK_ACL(aclrtCreateContext(&context,device));CHECK_ACL(aclrtSetCurrentContext(context));CHECK_ACL(aclrtCreateStream(&stream));
        aclrtRunMode mode;CHECK_ACL(aclrtGetRunMode(&mode));if(mode!=ACL_HOST&&mode!=ACL_DEVICE)throw std::runtime_error("Invalid ACL mode");
        toDevice=mode==ACL_HOST?ACL_MEMCPY_HOST_TO_DEVICE:ACL_MEMCPY_DEVICE_TO_DEVICE;
        fromDevice=mode==ACL_HOST?ACL_MEMCPY_DEVICE_TO_HOST:ACL_MEMCPY_DEVICE_TO_DEVICE;
        std::cout<<"run_mode="<<(mode==ACL_HOST?"ACL_HOST":"ACL_DEVICE")<<'\n';Preload();}
    void Preload();
    Buffer Allocate(const std::string& role,std::size_t bytes){Buffer b;b.bytes=bytes;b.role=role;
        CHECK_ACL(aclrtMalloc(&b.ptr,bytes,ACL_MEM_MALLOC_NORMAL_ONLY));buffers.push_back(b);return b;}
    aclTensor* View(const Buffer& b,const std::vector<int64_t>& dims,aclDataType dtype,std::size_t off=0){
        std::vector<int64_t> strides(dims.size(),1);std::size_t n=1;
        for(std::size_t i=dims.size();i>0;--i){if(dims[i-1]<=0)throw std::runtime_error("Invalid shape");strides[i-1]=n;n*=dims[i-1];}
        std::size_t item=dtype==ACL_FLOAT16?2:4;if(off>b.bytes/item||n>b.bytes/item-off||off*item%32)throw std::runtime_error("Invalid subview: "+b.role);
        auto* t=aclCreateTensor(dims.data(),dims.size(),dtype,strides.data(),0,ACL_FORMAT_ND,dims.data(),dims.size(),static_cast<uint8_t*>(b.ptr)+off*item);
        if(!t)throw std::runtime_error("Null tensor: "+b.role);
        tensors.push_back(t);return t;}
    void Sync(){CHECK_ACL(aclrtSynchronizeStream(stream));pending=false;++syncCount;}
    void* Workspace(uint64_t size){if(!size)return nullptr;if(size>workspaceCapacity){if(pending)Sync();if(workspace){CHECK_ACL(aclrtFree(workspace));workspace=nullptr;}
        CHECK_ACL(aclrtMalloc(&workspace,size,ACL_MEM_MALLOC_NORMAL_ONLY));workspaceCapacity=size;}return workspace;}
    void Upload(const Buffer& b,const void* p,std::size_t bytes){if(inChain||pending||bytes!=b.bytes)throw std::runtime_error("Upload inside chain/size mismatch");
        CHECK_ACL(aclrtMemcpy(b.ptr,b.bytes,p,bytes,toDevice));++copyOps;
        if(b.role=="spectrum" || b.role=="stats"){++copies.input;copies.inputBytes+=bytes;}
        else {++copies.constants;copies.constantBytes+=bytes;}}
    std::vector<float> Download(const Buffer& b){if(inChain||pending)throw std::runtime_error("Readback inside chain");
        std::vector<float> v(b.bytes/4);CHECK_ACL(aclrtMemcpy(v.data(),b.bytes,b.ptr,b.bytes,fromDevice));++copyOps;++copies.output;copies.outputBytes+=b.bytes;return v;}
};

inline void Resources::Preload(){
 std::cout<<"[PRELOAD] HtDecoderPackSplit="<<HtIstftStandalonePreload_HtDecoderPackSplit()<<'\n';
 std::cout<<"[PRELOAD] HtDftMatmul="<<HtIstftStandalonePreload_HtDftMatmul()<<'\n';
 std::cout<<"[PRELOAD] HtIstftMerge="<<HtIstftStandalonePreload_HtIstftMerge()<<'\n';
 std::cout<<"[PRELOAD] HtIstftWindowOla="<<HtIstftStandalonePreload_HtIstftWindowOla()<<'\n';
 std::cout<<"[PRELOAD] HtIstftCompensateCrop="<<HtIstftStandalonePreload_HtIstftCompensateCrop()<<'\n';
}
