#include "ht_istft_window_ola_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    const auto* frames=context->GetInputShape(0);const auto* window=context->GetInputShape(1);
    const auto* numerator=context->GetOutputShape(0);const auto* envelope=context->GetOutputShape(1);
    if (frames==nullptr || window==nullptr || numerator==nullptr || envelope==nullptr) return ge::GRAPH_FAILED;
    const auto& x=frames->GetOriginShape();const auto& w=window->GetOriginShape();
    const auto& y=numerator->GetOriginShape();const auto& e=envelope->GetOriginShape();
    if (x.GetDimNum()!=3 || x.GetDim(0)!=8 || x.GetDim(1)!=340 || x.GetDim(2)!=4096 ||
        w.GetDimNum()!=1 || w.GetDim(0)!=4096 ||
        y.GetDimNum()!=2 || y.GetDim(0)!=8 || y.GetDim(1)!=351232 ||
        e.GetDimNum()!=1 || e.GetDim(0)!=351232) return ge::GRAPH_FAILED;
    HtIstftWindowOlaTilingData tiling;
    tiling.set_channels(8);tiling.set_frameCount(340);tiling.set_nfft(4096);tiling.set_hop(1024);tiling.set_length(351232);
    auto* raw=context->GetRawTilingData();
    if (raw==nullptr || raw->GetCapacity()<tiling.GetDataSize()) return ge::GRAPH_FAILED;
    context->SetBlockDim(1);context->SetTilingKey(1);
    tiling.SaveToBuffer(raw->GetData(),raw->GetCapacity());raw->SetDataSize(tiling.GetDataSize());
    context->GetWorkspaceSizes(1)[0]=0;return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context) {
    auto* y=context->GetOutputShape(0);auto* e=context->GetOutputShape(1);
    if (y==nullptr || e==nullptr) return GRAPH_FAILED;
    y->SetDimNum(2);y->SetDim(0,8);y->SetDim(1,351232);
    e->SetDimNum(1);e->SetDim(0,351232);return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0,ge::DT_FLOAT);context->SetOutputDataType(1,ge::DT_FLOAT);return GRAPH_SUCCESS;
}
}
namespace ops {
class HtIstftWindowOla:public OpDef {
public:
    explicit HtIstftWindowOla(const char* name):OpDef(name) {
        this->Input("frames").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("window").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("numerator").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("envelope").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape);this->SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);this->AICore().AddConfig("ascend310b");
    }
};
OP_ADD(HtIstftWindowOla);
}
