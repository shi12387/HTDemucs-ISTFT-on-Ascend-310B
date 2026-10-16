# 本地验证范围

本地检查不依赖 Ascend 硬件，不作为 CANN 编译成功或真实 NPU 数值通过的证明。

已执行的检查：

- 15 个 `op_host/`、`op_kernel/` 文件的 SHA256 一致性检查。
- Python 与 Shell 语法检查，以及 C++11 ACL 调用程序的 SDK stub 语法检查。
- 五组 CPU 解析测试：源码校验、全零输出、通道与统计量、虚部符号和 OLA 裁剪、输入恢复。
- 五算子源文件清单、包装代码资源 getter 及一次注册语义的 C++ 模拟检查。
- 六用例的全尺寸调用模拟，覆盖张量视图、设备地址复用、调用顺序、同步与中间拷贝计数。
- 对非 Cube 阶段使用实际 `HtDecoderPackSplit`、`HtIstftMerge`、`HtIstftWindowOla`、`HtIstftCompensateCrop` 内核主体，在带边界断言的 CPU stub 上执行。
- 对 C++ 生成的 high/low 逆 DFT 系数执行三个时间块的 CPU 四项矩阵乘法探针，与独立 NumPy `irfft` 比较，并检查末块补齐列为零；该探针不模拟 Cube 累加过程。
- 验证修改输出会触发数值失败并清除通过标志，漏写或非有限输出会被运行程序拒绝。

调用模拟中的 MatMul 以 CPU 逆 FFT 替代，使用实际拆分后的 FP16 输入，但不模拟 FP16 DFT 系数及 Cube 的硬件累加过程。因此该模拟只能检查非 Cube 内核与调用代码，不能给出 NPU 误差或 NPU 性能结论。

独立工程的最终验证应在目标板执行 `01_prepare.sh`、`02_build_and_test.sh`，并取得 `PASS_STANDALONE_ISTFT_ALIGNMENT`。真实误差与耗时以本次板端 `validate.log` 和 `run.log` 为准。
