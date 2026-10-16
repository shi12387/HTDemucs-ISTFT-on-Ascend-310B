# iSTFT 测试说明

本文件解释测试数据如何生成、CPU 参考如何计算、NPU 输出如何检查，以及 `PASS_STANDALONE_ISTFT_ALIGNMENT` 的产生条件。工程接口、构建环境和板端实测数据见 [工程 README](../README.md)。

## 1. 两类测试

| 测试 | 入口 | 是否需要 NPU | 验证内容 | 成功输出 |
|---|---|---|---|---|
| CPU 解析测试 | `tests/test_reference.py` | 否 | 参考计算的符号、索引、布局、统计量和裁剪约定 | Python unittest 的 `OK` |
| 板端数值测试 | `03_test_only.sh`，或构建后由 `02_build_and_test.sh` 调用 | 是 | 五个算子的完整执行、输出覆盖、CPU 对照、全零与恢复检查 | `PASS_STANDALONE_ISTFT_ALIGNMENT` |

CPU 解析测试不会调用 NPU，也不会生成板端通过标志。板端测试对每组输入检查完整的 `[8,344064]` 输出，共 2,752,512 个 FP32 元素。

## 2. 如何运行

以下命令均在工程根目录执行。

只运行 CPU 解析测试：

```bash
python3 tests/test_reference.py
```

首次生成工程、编译并完成板端测试：

```bash
bash 01_prepare.sh
bash 02_build_and_test.sh
```

已构建工程且源码、测试输入未改变时，重新运行板端测试：

```bash
bash 03_test_only.sh
```

单独重新执行数值判定：

```bash
python3 validate.py
```

最后一个命令只读取已生成的结果，不会重新运行 NPU；它要求本次输入清单、算子库、完整执行报告和六组输出均存在且满足校验条件。

## 3. 测试由哪些文件完成

| 文件 | 作用 |
|---|---|
| `prepare_tests.py` | 生成六组输入、统计量、Hann 窗与 CPU 参考；保存文件及源码校验清单 |
| `check_project.py` | 在测试前核对算子源码、生成工程源码、头文件和库是否符合要求 |
| `begin_run.py` | 检查输入清单，记录本次算子库；清除通过标志和执行报告 |
| `runtime/main.cpp` | 上传测试输入，运行五算子链，检查输出覆盖与边界，保存结果 |
| `runtime/istft_pipeline.h` | 管理具体算子调用、同步和计算链内拷贝计数 |
| `validate.py` | 检查执行报告，逐元素比较 CPU 与 NPU 结果，生成最终通过标志 |
| `tests/test_reference.py` | 使用解析预期值检查 CPU 参考中的关键约定 |

`tests/` 存放解析测试与测试说明；板端测试入口和数值判定代码位于上表所列路径。

## 4. CPU 参考结果如何生成

### 输入恢复与统计量

`prepare_tests.py` 生成 `[1,16,2048,336]` FP32 packed 频谱及 `[8]` FP32 统计量。默认随机种子为 `2026`，可通过准备脚本的 `--seed` 参数更改。

每组输入先按以下约定恢复为 `[8,2049,340]` 复数频谱：

1. 每个声部四平面依次为左实部、左虚部、右实部、右虚部，展开为八个通道。
2. 按 FP32 运算顺序执行 `x*stats[1]+stats[0]`。
3. 将 336 个输入帧放在完整频谱的 `[2:338]`；前后各两帧保持零。
4. Nyquist 频点 2048 补零，DC 频点的虚部置零。

### 默认 PyTorch 参考

默认调用 CPU `torch.istft`：

```python
y = torch.istft(
    z,
    n_fft=4096,
    hop_length=1024,
    win_length=4096,
    window=window,
    center=True,
    normalized=True,
    onesided=True,
    length=347136,
    return_complex=False,
)
reference = y[:, 1536:1536 + 344064]
```

其中 `window` 为 FP32 周期 Hann 窗，与上传给 NPU 的窗相同。PyTorch 的中心裁剪为 2048 点，加上显式的 1536 点裁剪，对应 NPU 相对于完整 OLA 结果的起点 `3584`。

### NumPy 参考

准备时选择 `--reference numpy`，则采用：

1. `numpy.fft.irfft(..., n=4096, norm="ortho")` 计算各帧逆变换，结果转换为 FP32。
2. 各帧乘以 Hann 窗，按帧索引递增的顺序执行 FP32 overlap-add。
3. 独立累加 Hann 窗平方，得到窗包络。
4. 从完整 OLA 结果的索引 3584 起取 344064 点，再除以对应窗包络。

参考结果不读取 NPU 输出，也不通过 NPU 的 DFT MatMul 链生成。两条参考路径均使用本次的实际 FP32 输入与窗。参考后端、版本、参数、种子和文件校验值保存在 `test_data/manifest.json`。

## 5. 六组板端测试检查什么

| 用例 | 数据特点 | 主要检查目的 |
|---|---|---|
| `tones_warmup` | 不同通道具有不同的低频、高频、DC 分量和相位 | 首次执行及基础数值正确性；虽然标记预热，仍参与数值判定 |
| `impulse` | 在不同帧放置带相位变化的频谱，形成接近脉冲的帧响应 | 逆变换相位、符号、通道顺序与帧叠加位置 |
| `tones` | 与预热相同的频谱 | 非首次运行结果及后续恢复比较的基准 |
| `noise` | 固定种子的随机频谱；`mean=0.003`、`std=0.73` | 密集频谱与非恒等反归一化 |
| `zero` | 输入频谱为零，`mean=0`、`std=1` | 检查是否残留其他输入的非零结果 |
| `tones_restore` | 在同一组设备缓冲区重新写入 tones | 检查输入变化后能否恢复一致的输出 |

`impulse` 用例也遵循 Nyquist 补零和帧裁剪约定，因此其预期值以 CPU iSTFT 参考为准，不要求输出一定为理想的单点脉冲。

所有用例复用设备缓冲区和常量，计算链为：

```text
HtDecoderPackSplit → HtDftMatmul → HtIstftMerge
                  → HtIstftWindowOla → HtIstftCompensateCrop
```

CPU 不在该链中重建或修正中间张量。测试程序只在链前上传输入与统计量，在链后回读输出。

## 6. PASS 的完整判定条件

### 执行和输出覆盖检查

每次执行前，C++ 程序将输出有效区填入 NaN 哨兵，并在有效区前后各放置 16 个 FP32 边界标记。执行完成后检查：

- ACL 查询、启动和同步均成功。
- 输出全部为有限值，未保留 NaN 哨兵。
- 输出前后边界标记未改变。
- 计算链没有应用层中间拷贝或回读。
- 每次启动 29,328 个 kernel、同步 173 次，符合固定计算链约定。

六个用例全部完成后，C++ 程序写出 `results/run_report.json`，状态为 `RUN_COMPLETE_REQUIRES_NUMERICAL_VALIDATION`。此状态只表示执行阶段完成，尚未表示数值通过。

### 执行记录与文件校验

`validate.py` 检查：

- 输入文件、参考文件与清单内记录的大小和 SHA256 相符，记录的源码未改变。
- 本次算子库与 `begin_run.py` 记录的大小和 SHA256 相符，输入清单未在运行中改变。
- 执行报告包含完整的六个用例，顺序和预热标记正确。
- 调用、同步、中间拷贝、边界检查、输入输出字节数、缓存常量及工作区大小符合约定。
- 计时值为非负有限数；计时长短不作为数值通过门限。

这些校验用于关联本次测试使用的输入、源码、算子库与执行记录。它们不替代数值比较。

### 全元素数值比较

设 `a` 为 NPU 输出，`b` 为 CPU 参考。每个用例均要求：

```text
对全部 2,752,512 个元素：
    abs(a[i] - b[i]) <= 2e-5 + 2e-5 * abs(b[i])

relative_l2_error = sqrt(sum((a-b)^2) / sum(b^2)) <= 2e-5
```

误差指标在 CPU 上以 FP64 计算，输入的 NPU 结果和参考结果均为 FP32。当参考能量为零时，只有误差也为零才将相对 L2 误差记为 0；否则视为无穷大并失败。

另有两个严格检查：

- `zero` 的全部输出数值必须为零；正零和负零均视为数值零。
- `tones_restore` 与 `tones` 的 NPU 输出必须逐位一致，确保复用缓冲区后结果可恢复。

### 生成最终标志

只有上述检查全部通过，`validate.py` 才会写入：

- `results/validation.json`：误差指标、测试边界和验证状态。
- `results/istft_pass.json`：本次通过标志及输入、输出、算子库的校验信息。
- `results/timing_runs.csv`：六个用例的分项计时。

随后打印：

```text
PASS_STANDALONE_ISTFT_ALIGNMENT
```

标准脚本在运行前清除通过标志，验证程序也会在判定开始时清除通过标志；判定失败时不会生成新的有效 PASS。验证中的 `[CHECK]` 行只表示已检查到该用例，必须看到最终 PASS 才表示所有判据都已通过。

## 7. 日志如何解读

| 字段或提示 | 含义 |
|---|---|
| `COMPLETE: six cases...` | C++ 执行完成，后续仍需数值判定 |
| `checked_elements` | 每个用例实际比较的元素数，应为 2,752,512 |
| `mismatch_count` | 超出逐元素容差的元素数，通过时必须为 0 |
| `bitwise_difference_count` | 与 CPU 参考的位表示差异数，不要求普遍为 0 |
| `max_abs_error` | 所有元素的最大绝对误差；不单独代替逐元素判据 |
| `rmse` | 均方根误差，记录指标，不另设通过门限 |
| `relative_l2_error` | 整体相对 L2 误差，必须不超过 `2e-5` |
| `guard_mismatch_count` | 输出前后边界标记变化数，必须为 0 |
| `intermediate_host_copies` / `intermediate_readbacks` | 设备计算链内的应用层中间拷贝与回读，必须为 0 |
| `PASS_STANDALONE_ISTFT_ALIGNMENT` | 全部执行与数值判据通过 |

若构建或 ACL 调用失败，Shell 脚本会停止，可能还未生成 `validate.log`。若已生成该日志但没有最终 PASS，应检查失败异常及对应的指标；不能仅凭最后一条 `[CHECK]` 判断整体通过。

## 8. CPU 解析测试与测试范围

`tests/test_reference.py` 包含五项检查：

1. `test_sources`：核对 15 个算子源码文件的 SHA256。
2. `test_zero`：全零频谱的 CPU iSTFT 结果为零。
3. `test_unpack_stats_edges_and_order`：检查 mean/std 运算、八通道排列、前后补帧、Nyquist 和 DC 虚部。
4. `test_single_frame_sine_sign_and_ola_crop`：用单频虚部构造具有解析预期的正弦帧，检查逆变换符号、窗包络及裁剪位置。
5. `test_restore_case_identity`：确认 tones 与恢复用例生成的输入及统计量相同。

板端测试验证的是所列固定形状合成频谱的最终输出。它没有逐项回读所有中间 MatMul、帧和 OLA 张量，也不验证任意幅度、其他 FFT 参数、真实音乐分离质量、模型精度或其他硬件环境。

通过标志不表示达到特定速度，也不证明相对于 CPU 更快。耗时是包含 ACL 查询、调用、调度和同步的主机墙钟时间；首次执行与输入输出传输的计时边界见工程 README。
