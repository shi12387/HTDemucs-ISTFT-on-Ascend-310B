#include "ht_istft_compensate_crop_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    const auto* numerator=context->GetInputShape(0);const auto* envelope=context->GetInputShape(1);
    const auto* output=context->GetOutputShape(0);
    if (numerator==nullptr || envelope==nullptr || output==nullptr) return ge::GRAPH_FAILED;
    const auto& x=numerator->GetOriginShape();const auto& e=envelope->GetOriginShape();const auto& y=output->GetOriginShape();
    if (x.GetDimNum()!=2 || x.GetDim(0)!=8 || x.GetDim(1)!=351232 ||
        e.GetDimNum()!=1 || e.GetDim(0)!=351232 ||
        y.GetDimNum()!=2 || y.GetDim(0)!=8 || y.GetDim(1)!=344064) return ge::GRAPH_FAILED;
    HtIstftCompensateCropTilingData tiling;
    tiling.set_channels(8);tiling.set_inputLength(351232);tiling.set_outputLength(344064);
    tiling.set_cropStart(3584);tiling.set_tileLength(1024);
    auto* raw=context->GetRawTilingData();
    if (raw==nullptr || raw->GetCapacity()<tiling.GetDataSize()) return ge::GRAPH_FAILED;
    context->SetBlockDim(1);context->SetTilingKey(1);
    tiling.SaveToBuffer(raw->GetData(),raw->GetCapacity());raw->SetDataSize(tiling.GetDataSize());
    context->GetWorkspaceSizes(1)[0]=0;return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context) {
    auto* y=context->GetOutputShape(0);if (y==nullptr) return GRAPH_FAILED;
    y->SetDimNum(2);y->SetDim(0,8);y->SetDim(1,344064);return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0,ge::DT_FLOAT);return GRAPH_SUCCESS;
}
}
namespace ops {
class HtIstftCompensateCrop:public OpDef {
public:
    explicit HtIstftCompensateCrop(const char* name):OpDef(name) {
        this->Input("numerator").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("envelope").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape);this->SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);this->AICore().AddConfig("ascend310b");
    }
};
OP_ADD(HtIstftCompensateCrop);
}
