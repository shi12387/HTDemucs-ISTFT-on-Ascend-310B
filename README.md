# HTDemucs iSTFT on Ascend 310B

固定参数的 Ascend C iSTFT 工程，将四声部、双声道的 packed 频谱转换为时域波形。工程包含五个自定义算子、独立 ACL 调用程序、测试信号生成和 CPU 数值对照，不依赖模型权重或音频文件。

本工程已在 Orange Pi AI Pro 8T、Ascend 310B4、CANN Toolkit 8.0.0 环境中完成编译、五算子资源加载、ACL 调用与 CPU 参考数值对照。2026-10-16 板端运行的六组测试全部通过。`PASS_STANDALONE_ISTFT_ALIGNMENT` 由 `validate.py` 在全部输出数值、执行记录和输入恢复检查通过后生成；仅完成编译或执行不会产生该标志。测试方法见 [测试说明](tests/README.md)，实测数据见“验证结果与输出”。

## 1. 输入与输出

| 项目 | 约定 |
|---|---|
| 频谱输入 | `[1,16,2048,336]`，连续 FP32 |
| 输入排列 | 每个声部四平面：`left_real, left_imag, right_real, right_imag` |
| 统计量输入 | `[8]` FP32，`stats[0]=mean`、`stats[1]=std`，其余填零 |
| 反归一化 | `x*std+mean`，先 FP32 乘法，再 FP32 加法；所有声部共用一组 mean/std |
| 变换参数 | `n_fft=4096`、`hop_length=1024`、`normalized=True` |
| 合成窗 | 长度 4096 的周期 Hann 窗，使用未缩放的 Hann 值 |
| 恢复频谱 | 频点 2048 补零，前后各补两帧，得到 340 帧 |
| 实数逆变换约定 | DC 虚部忽略；Nyquist 为零；内部频点使用共轭对称权重 |
| 帧输出 | `[8,340,4096]` FP32 |
| OLA 长度 | 每通道 351232 点 |
| 裁剪 | 窗包络补偿后，从 OLA 结果的索引 3584 开始保留 344064 点 |
| 直接输出 | `[8,344064]`，连续 FP32，可视为 `[4,2,344064]` |

未做频谱 mean/std 归一化的输入使用 `mean=0, std=1`。标准差必须为正；有限输入反归一化后的数值应在 FP16 可表示范围内。

输出是频率分支重建波形。它不包含时域分支相加、全局音频尺度恢复，也不包含模型推理。

`344064` 是内核的每通道存储长度。面向长度为 `343980` 的音频片段时，每通道有效区为前 `343980` 点，尾部 84 点不属于该片段。直接输出不等于紧凑存储的 `[4,2,343980]`：读取时必须保留正确的通道步长。

## 2. 实现原理

逆 STFT 包括逆 DFT、乘合成窗、重叠相加和窗包络补偿。对一帧单边频谱，归一化逆变换的目标为：

```text
frame[n] = ( Re(Z[0]) + Re(Z[2048])*(-1)^n
           + 2*sum(k=1..2047) {
               Re(Z[k])*cos(2*pi*k*n/4096)
             - Im(Z[k])*sin(2*pi*k*n/4096)
           } ) / 64
```

本工程将余弦和负正弦系数组成矩阵，通过 Cube 矩阵乘法实现逆 DFT。频谱与系数均拆分成 FP16 high/low 两部分，计算 `HH`、`HL`、`LH`、`LL` 四项，按 `((HH+HL)+(LH+LL))/64` 合并为 FP32 帧数据。

随后计算：

```text
numerator[q] = sum(frame_index) frame[frame_index,local] * Hann[local]
envelope[q]  = sum(frame_index) Hann[local]^2
wave[q]      = numerator[q] / envelope[q]
local        = q - frame_index*1024
```

只在 `0 <= local < 4096` 时计入该帧。窗包络由全部 340 帧的窗计算，包括频谱补零的边界帧。裁剪起点 `3584=2048+1536`，分别对应中心裁剪与应用接口的外层裁剪。

| 算子 | 功能 |
|---|---|
| `HtDecoderPackSplit` | 频谱反归一化、恢复频点和帧、排列成逆变换输入并拆分 high/low |
| `HtDftMatmul` | 执行分块 Cube 矩阵乘法 |
| `HtIstftMerge` | 合并四项结果、完成 `1/64` 缩放并写成连续时域帧 |
| `HtIstftWindowOla` | 乘 Hann 窗、重叠相加，生成 numerator 和 envelope |
| `HtIstftCompensateCrop` | 除以窗包络，按固定起点和长度裁剪 |

以上五个阶段在 NPU 执行。CPU 负责系数初始化、ACL 调用和测试验证。窗与逆 DFT 系数上传一次；测试循环只更换频谱和统计量，计算链内部没有应用层中间张量上传或回读。

## 3. 目录与文件

| 路径 | 用途 |
|---|---|
| `op_host/` | 五个算子的注册、类型和形状推导、tiling 与参数结构 |
| `op_kernel/` | 五个 Ascend C 内核实现 |
| `runtime/main.cpp` | 六组 ACL 测试入口、输出覆盖检查和结果保存 |
| `runtime/acl_resources.h` | ACL 上下文、流、设备内存、张量视图与资源预加载 |
| `runtime/istft_pipeline.h` | 系数缓存和五算子连续调用 |
| `runtime/inverse_plan.h` | 生成实数逆 DFT 的余弦、负正弦及端点系数 |
| `tests/test_reference.py` | CPU 解析测试：符号、通道顺序、统计量、补零、OLA 与裁剪 |
| `tests/README.md` | CPU 与 NPU 测试流程、参考结果生成方法、通过判据及日志解读 |
| `01_prepare.sh` | 生成测试数据和参考值，创建并配置 CANN 工程 |
| `02_build_and_test.sh` | 编译自定义算子，随后运行测试及数值验证 |
| `03_test_only.sh` | 使用已构建的库重新运行独立验证 |
| `anchor_dft_matmul.json` | msopgen 使用的算子描述 |
| `configure_project.py` | 配置五算子编译条目及资源注册生成步骤 |
| `resource_preload_codegen.py` | 为生成的 ACLNN 包装代码添加共享资源预加载入口 |
| `cann_common.sh` | 配置 CANN 环境和头文件、动态库路径 |
| `prepare_tests.py` | 生成测试频谱、统计量、Hann 窗和 CPU iSTFT 参考 |
| `support.py` | 接口约定、源码与数据校验、公共函数 |
| `baseline_sources.json` | 15 个算子源码文件的 SHA256 |
| `check_project.py` | 检查生成工程、算子源码及构建产物 |
| `begin_run.py` | 校验测试输入，记录本次算子库并清除通过标志 |
| `validate.py` | 检查所有输出元素、全零输入和输入恢复，生成报告 |
| `THIRD_PARTY_NOTES.md` | 第三方依赖与参考资料 |
| `LOCAL_VALIDATION.md` | 本地检查方法和适用范围 |

以下目录由脚本生成，不需要提交源码仓库：

| 目录 | 内容 |
|---|---|
| `IstftStandalone/` | CANN 工程、构建中间文件、自定义库和生成的 ACLNN 头文件 |
| `bin/` | `istft_test` 可执行程序 |
| `test_data/` | 测试输入、窗、CPU 参考和文件清单 |
| `results/` | NPU 输出、验证报告和分项耗时 |

## 4. 快速开始

### 环境

- Ascend 310B4，CANN Toolkit 8.0.0，设备及驱动正常。
- Bash、CMake、Make、支持 C++11 的 g++。
- Python 3.8 或更高版本与 NumPy；默认 CPU 参考需要 PyTorch。
- 工程使用 ASCII、无空格路径。目录名避免包含芯片标识字符串 `ascend310b`，使用下面的工程目录即可。

解压到 `~/ascend_work`，首次执行：

```bash
cd "$HOME/ascend_work/htdemucs_istft_standalone"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
chmod go-w .
bash 01_prepare.sh
bash 02_build_and_test.sh
```

一行执行准备、编译和测试：

```bash
cd "$HOME/ascend_work/htdemucs_istft_standalone" && bash 01_prepare.sh && bash 02_build_and_test.sh
```

默认参考为 CPU `torch.istft`，使用恢复后的 `[8,2049,340]` 复数频谱：

```text
n_fft=4096, hop_length=1024, win_length=4096
center=True, normalized=True, onesided=True, length=347136
随后每通道取 [1536:1536+344064]
```

可显式选择 NumPy 参考，将准备命令替换为：

```bash
bash 01_prepare.sh --reference numpy
```

NumPy 路径执行 `irfft(norm="ortho")`，再按 FP32 运算顺序进行 OLA、窗包络补偿和裁剪。两种参考均使用传给 NPU 的实际 FP32 Hann 窗，后端及版本保存在 `test_data/manifest.json`。

生成工程已存在时，重新编译和测试使用：

```bash
bash 02_build_and_test.sh
```

已编译且源码、测试数据未改变时，仅重新测试：

```bash
bash 03_test_only.sh
```

修改运行脚本后需重新生成测试数据清单。以默认参考为例：

```bash
python3 prepare_tests.py
bash 02_build_and_test.sh
```

`baseline_sources.json` 用于锁定算子源码，修改算子源码时也需要同步维护其校验记录。更改参考后端时，对应使用 `python3 prepare_tests.py --reference numpy`。

设备默认编号为 0，可通过 `HTSTFT_DEVICE_ID` 设置。Toolkit 默认路径为 `/usr/local/Ascend/ascend-toolkit/latest`，可通过 `HTSTFT_CANN_ROOT` 设置。例如：

```bash
HTSTFT_DEVICE_ID=0 bash 03_test_only.sh
```

程序直接链接 `IstftStandalone/build_out/op_api/lib/libcust_opapi.so`，不需要安装到系统 OPP。工程生成、编译和测试均不下载模型权重或音频。CPU 解析测试可单独执行：

```bash
python3 tests/test_reference.py
```

## 5. 验证结果与输出

完整测试由三步构成：`prepare_tests.py` 生成输入和 CPU 参考，`runtime/main.cpp` 通过 ACL 运行 NPU 链并保存输出，最后由 `validate.py` 比较全部元素并判定是否通过。详细步骤、脚本职责和失败判据见 [tests/README.md](tests/README.md)。

测试用例依次为：

| 用例 | 检查内容 |
|---|---|
| `tones_warmup` | 首次运行，区分初始化后的首次调用开销 |
| `impulse` | 带不同位置和符号相位的频谱，检查逆变换与 OLA |
| `tones` | 不同通道的低频、高频与 DC 分量，检查平面排列和虚部符号 |
| `noise` | 随机频谱，使用非零 mean 和非单位 std 检查反归一化 |
| `zero` | 换零输入，检查设备缓冲区是否残留非零数据 |
| `tones_restore` | 复用相同设备地址，恢复 tones 并逐位比较 |

每个用例检查全部 2,752,512 个输出元素，要求最大逐元素判据 `abs(NPU-reference) <= 2e-5 + 2e-5*abs(reference)`，且相对 L2 误差不超过 `2e-5`。同时检查输出有限值、前后边界哨兵、全零输入和恢复结果一致性。

### 板端数值验证

2026-10-16 的板端记录中，六组测试均检查 2,752,512 个输出元素，`mismatch_count` 全部为 0。结果来自独立测试程序生成的频谱，误差以该次运行所使用的 CPU 参考为基准：

| 用例 | 最大绝对误差 | RMSE | 相对 L2 误差 | 超出容差元素数 |
|---|---:|---:|---:|---:|
| `tones_warmup` | 2.6077e-8 | 3.0261e-9 | 1.3134e-7 | 0 |
| `impulse` | 5.2452e-6 | 3.9523e-9 | 5.4458e-7 | 0 |
| `tones` | 2.6077e-8 | 3.0261e-9 | 1.3134e-7 | 0 |
| `noise` | 1.0617e-7 | 2.1019e-8 | 9.9817e-7 | 0 |
| `zero` | 0 | 0 | 0 | 0 |
| `tones_restore` | 2.6077e-8 | 3.0261e-9 | 1.3134e-7 | 0 |

运行还通过了以下检查：

- 五个算子的资源预加载均成功，运行模式为 `ACL_DEVICE`。
- 六组测试的输出边界 `guard_mismatch_count` 均为 0，输出均为有限值。
- 全零输入的输出数值为零，与参考结果逐位相同。
- `tones_restore` 与 `tones` 的 NPU 输出逐位相同，输入恢复检查通过。
- 设备计算链内的应用层中间主机拷贝与回读次数均为 0。

验证日志最终输出：

```text
PASS_STANDALONE_ISTFT_ALIGNMENT
```

`bitwise_difference_count` 统计与 CPU 参考的浮点位表示差异。除明确要求逐位一致的检查外，数值通过条件以逐元素容差和相对 L2 误差为准；该计数非零不等同于超出容差。

以上结果覆盖所列固定形状测试信号，不表示任意输入幅度、其他形状或其他硬件环境都已验证。实际参考后端及版本记录在 `test_data/manifest.json`，可用于复现。

### 板端耗时记录

以下为同一次板端运行的观测值，单位为毫秒。非预热范围取 `impulse`、`tones`、`noise`、`zero`、`tones_restore` 五个用例的最小值和最大值：

| 项目 | 耗时（ms） |
|---|---:|
| 常量与设备缓冲区准备 `preparation_wall_ms` | 2077.34 |
| 首次设备流水线 `pipeline_wall_ms` | 4398.93 |
| 非预热设备流水线 `pipeline_wall_ms` | 3822.38–3842.31 |
| 非预热频谱解包、反归一化及拆分 `pack_ms` | 1929.85–1937.00 |
| 非预热逆 DFT 矩阵乘法 `matmul_ms` | 1816.84–1832.20 |
| 非预热四项结果合并 `merge_ms` | 54.50–57.49 |
| 非预热加窗重叠相加 `ola_ms` | 15.00–15.02 |
| 非预热窗包络补偿及裁剪 `crop_ms` | 5.68–5.71 |
| 非预热输入上传、设备流水线及输出回读 `wall_ms` | 3876.39–3896.33 |

该记录对应四声部、双声道、每通道 340 帧的完整 iSTFT 链路。主要耗时集中在频谱解包与拆分、逆 DFT 矩阵乘法。频谱及统计量在每次计算前上传，窗和逆 DFT 系数在初始化时上传一次。

### 输出文件

| 文件 | 内容 |
|---|---|
| `results/*_npu_f32.bin` | 各用例 `[8,344064]` 的裸 FP32 输出，无文件头 |
| `results/validation.json` | 每个用例的最大绝对误差、RMSE、相对 L2 和验证状态 |
| `results/istft_pass.json` | 本次通过标志、输入与算子库校验信息 |
| `results/run_report.json` | 调用、同步、数据传输计数及原始计时 |
| `results/timing_runs.csv` | 各用例的分项耗时和预热标记 |
| `prepare.log`、`build.log` | 工程生成和算子编译日志 |
| `app_build.log`、`run.log`、`validate.log` | 程序编译、ACL 执行和数值验证日志 |

读取输出及选取有效音频的示例：

```python
import numpy as np

y = np.fromfile("results/tones_npu_f32.bin", dtype="<f4").reshape(4, 2, 344064)
valid_audio = y[..., :343980]
# 若需要紧凑输出，显式创建连续数组；此处是文件读取后的 CPU 操作。
valid_audio = np.ascontiguousarray(valid_audio)
```

每次设备计算链包含 29,240 次 MatMul、85 次 Merge 和其余三个算子各一次，共 29,328 次 kernel 启动及 173 次同步。计时采用主机 `steady_clock`，包含 ACL 查询、调用、调度和同步，不是 NPU Event 纯内核时间。

`segment_wall_ms` 包含每次频谱与统计量上传、设备计算链及最终输出回读，不包含常量初始化、哨兵重置、文件读写或 CPU 验证。首次调用单独标记为预热；六组不同信号的记录不等于多轮性能统计基准。

构建或 ACL 调用失败时脚本会停止，可能尚未生成 `validate.log`。数值结果与实际耗时以板端本次运行日志为准。

## 6. 第三方资料

依赖及参考资料见 [THIRD_PARTY_NOTES.md](THIRD_PARTY_NOTES.md)。本文件与第三方说明均不代替仓库的 `LICENSE`。
