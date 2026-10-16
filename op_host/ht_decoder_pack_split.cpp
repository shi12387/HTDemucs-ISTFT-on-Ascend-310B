#include "ht_decoder_pack_split_tiling.h"
#include "register/op_def_registry.h"
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    const auto* input0=context->GetInputShape(0);if(!input0)return ge::GRAPH_FAILED;
    const auto& input0Shape=input0->GetOriginShape();
    if(input0Shape.GetDimNum()!=4 || input0Shape.GetDim(0)!=1 || input0Shape.GetDim(1)!=16 || input0Shape.GetDim(2)!=2048 || input0Shape.GetDim(3)!=336)return ge::GRAPH_FAILED;
    const auto* input1=context->GetInputShape(1);if(!input1)return ge::GRAPH_FAILED;
    const auto& input1Shape=input1->GetOriginShape();
    if(input1Shape.GetDimNum()!=1 || input1Shape.GetDim(0)!=8)return ge::GRAPH_FAILED;
    const auto* output0=context->GetOutputShape(0);if(!output0)return ge::GRAPH_FAILED;
    const auto& output0Shape=output0->GetOriginShape();
    if(output0Shape.GetDimNum()!=3 || output0Shape.GetDim(0)!=85 || output0Shape.GetDim(1)!=32 || output0Shape.GetDim(2)!=4096)return ge::GRAPH_FAILED;
    const auto* output1=context->GetOutputShape(1);if(!output1)return ge::GRAPH_FAILED;
    const auto& output1Shape=output1->GetOriginShape();
    if(output1Shape.GetDimNum()!=3 || output1Shape.GetDim(0)!=85 || output1Shape.GetDim(1)!=32 || output1Shape.GetDim(2)!=4096)return ge::GRAPH_FAILED;
    HtDecoderPackSplitTilingData tiling;tiling.set_version(1);auto* raw=context->GetRawTilingData();
    if(!raw || raw->GetCapacity()<tiling.GetDataSize())return ge::GRAPH_FAILED;
    context->SetBlockDim(1);context->SetTilingKey(1);tiling.SaveToBuffer(raw->GetData(),raw->GetCapacity());
    raw->SetDataSize(tiling.GetDataSize());context->GetWorkspaceSizes(1)[0]=0;return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context) {
    auto* y0=context->GetOutputShape(0);if(!y0)return GRAPH_FAILED;y0->SetDimNum(3);y0->SetDim(0,85);y0->SetDim(1,32);y0->SetDim(2,4096);
    auto* y1=context->GetOutputShape(1);if(!y1)return GRAPH_FAILED;y1->SetDimNum(3);y1->SetDim(0,85);y1->SetDim(1,32);y1->SetDim(2,4096);
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0,ge::DT_FLOAT16);
    context->SetOutputDataType(1,ge::DT_FLOAT16);
    return GRAPH_SUCCESS;
}
}
namespace ops {
class HtDecoderPackSplit:public OpDef {
public:
    explicit HtDecoderPackSplit(const char* name):OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("stats").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("high").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("low").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape);this->SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);this->AICore().AddConfig("ascend310b");
    }
};
OP_ADD(HtDecoderPackSplit);
}
