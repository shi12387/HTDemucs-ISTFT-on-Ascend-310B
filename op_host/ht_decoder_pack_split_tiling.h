#ifndef HT_DECODER_PACK_SPLIT_TILING_H
#define HT_DECODER_PACK_SPLIT_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(HtDecoderPackSplitTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, version);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(HtDecoderPackSplit,HtDecoderPackSplitTilingData)
}
#endif
