# 第三方依赖与参考说明

本文件记录固定参数 HTDemucs iSTFT 工程所使用的开发工具、运行库和数值参考资料。

## Huawei Ascend / CANN

工程面向 CANN Toolkit 8.0.0，依赖 Ascend C、Cube MatMul API、算子注册与 tiling API、ACL 和 ACLNN。`msopgen` 用于在使用者安装的 Toolkit 环境中生成工程、构建脚本和 ACLNN 包装代码；CANN 编译及打包工具生成本地自定义算子库。

`configure_project.py` 配置五算子工程，`resource_preload_codegen.py` 为生成的包装代码添加共享资源注册入口。SDK 头文件、运行库、编译器及 CANN 工程模板不作为源码包交付。

生成文件所带的版权与许可声明应予保留，第三方组件的使用以随附条款为准。

- [CANN 文档入口](https://www.hiascend.com/cann/document)
- [CANN 8.0.0：Ascend C 自定义算子开发实践](https://www.hiascend.com/document/detail/zh/canncommercial/800/devaids/opdev/optool/atlasopdev_16_0027.html)

## PyTorch

默认 CPU 参考使用 `torch.hann_window` 和 `torch.istft`，验证合成窗、归一化逆变换、重叠相加与裁剪后的数值结果。PyTorch 只用于准备和验证，不参与 NPU 内核计算。选择 NumPy 参考时不需要 PyTorch。

- [PyTorch 官方项目](https://github.com/pytorch/pytorch)
- [torch.istft](https://docs.pytorch.org/docs/stable/generated/torch.istft.html)
- [torch.hann_window](https://docs.pytorch.org/docs/stable/generated/torch.hann_window.html)

## NumPy

NumPy 用于生成测试频谱、读取二进制张量、执行误差校验。NumPy 参考模式使用 `numpy.fft.irfft` 配合 `norm="ortho"`，并在 CPU 上计算 FP32 OLA、窗包络和裁剪参考。

- [NumPy 官方项目](https://github.com/numpy/numpy)
- [numpy.fft.irfft](https://numpy.org/doc/stable/reference/generated/numpy.fft.irfft.html)

## Demucs / HTDemucs

本工程的频谱排列、Nyquist 补零、帧恢复及裁剪接口面向 HTDemucs 频率分支。工程不包含 Demucs 模型权重、OM 模型或完整音频分离程序。

- [Demucs 官方仓库](https://github.com/facebookresearch/demucs)
- [HTDemucs 模型实现](https://github.com/facebookresearch/demucs/blob/main/demucs/htdemucs.py)
- [频谱处理实现](https://github.com/facebookresearch/demucs/blob/main/demucs/spec.py)

## 构建工具与许可范围

工程使用 Bash、Python、CMake、Make 和 g++，由使用者的系统环境提供。

本文件说明依赖关系与参考资料，不为第三方 SDK、工具、库或模型授予许可，也不是本仓库的 `LICENSE`。本仓库自身代码的授权应由单独的 `LICENSE` 明确；文件中的第三方版权或许可声明应保留。
