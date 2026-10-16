#include "ht_istft_merge_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    for (uint32_t i=0;i<4;++i) {
        const auto* input=context->GetInputShape(i);
        if (input==nullptr) return ge::GRAPH_FAILED;
        const auto& x=input->GetOriginShape();
        if (x.GetDimNum()!=3 || x.GetDim(0)!=86 || x.GetDim(1)!=32 || x.GetDim(2)!=48)
            return ge::GRAPH_FAILED;
    }
    const auto* output=context->GetOutputShape(0);
    if (output==nullptr) return ge::GRAPH_FAILED;
    const auto& y=output->GetOriginShape();
    if (y.GetDimNum()!=2 || y.GetDim(0)!=32 || y.GetDim(1)!=4096) return ge::GRAPH_FAILED;
    HtIstftMergeTilingData tiling;
    tiling.set_timeBlocks(86);tiling.set_frames(32);tiling.set_nfft(4096);
    auto* raw=context->GetRawTilingData();
    if (raw==nullptr || raw->GetCapacity()<tiling.GetDataSize()) return ge::GRAPH_FAILED;
    context->SetBlockDim(1);context->SetTilingKey(1);
    tiling.SaveToBuffer(raw->GetData(),raw->GetCapacity());raw->SetDataSize(tiling.GetDataSize());
    context->GetWorkspaceSizes(1)[0]=0;
    return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context) {
    auto* y=context->GetOutputShape(0);if (y==nullptr) return GRAPH_FAILED;
    y->SetDimNum(2);y->SetDim(0,32);y->SetDim(1,4096);return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0,ge::DT_FLOAT);return GRAPH_SUCCESS;
}
}
namespace ops {
class HtIstftMerge:public OpDef {
public:
    explicit HtIstftMerge(const char* name):OpDef(name) {
        this->Input("hh").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("hl").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("lh").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("ll").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape);this->SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);this->AICore().AddConfig("ascend310b");
    }
};
OP_ADD(HtIstftMerge);
}
