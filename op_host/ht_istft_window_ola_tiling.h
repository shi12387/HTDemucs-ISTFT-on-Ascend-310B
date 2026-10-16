#ifndef HT_ISTFT_WINDOW_OLA_TILING_H
#define HT_ISTFT_WINDOW_OLA_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(HtIstftWindowOlaTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, channels);
    TILING_DATA_FIELD_DEF(uint32_t, frameCount);
    TILING_DATA_FIELD_DEF(uint32_t, nfft);
    TILING_DATA_FIELD_DEF(uint32_t, hop);
    TILING_DATA_FIELD_DEF(uint32_t, length);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(HtIstftWindowOla,HtIstftWindowOlaTilingData)
}
#endif
