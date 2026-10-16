#ifndef HT_ISTFT_MERGE_TILING_H
#define HT_ISTFT_MERGE_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(HtIstftMergeTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, timeBlocks);
    TILING_DATA_FIELD_DEF(uint32_t, frames);
    TILING_DATA_FIELD_DEF(uint32_t, nfft);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(HtIstftMerge,HtIstftMergeTilingData)
}
#endif
