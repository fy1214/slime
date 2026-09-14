# 同 packed 数据逐段检查与 TE 功能核对

Job `2792145` 已提交；使用已完成 job `2791917` 的镜像、冻结 production source mount、prep 与单进程/单 GPU 入口。CPU threads=1、MAX_JOBS=1、CUDA_VISIBLE_DEVICES=0，time limit=20min。bash -n、Python AST、sbatch --test-only 已完成；没有修改正式训练。

容器内 driver：
`python3 /lustre/fsw/general_sa/shuazhang/python_space/verl_for_nvfp4_20251031/investigation/nvfp4_numeric_diagnostic_20260910_v2/stage_probe.py`

脚本复用 v1 的权重加载与统计函数，但输出到 v2。结果为 `results_2792145.json`；TE 实际源码相关定义保存到 `te_runtime_features.json`，仅导出包含功能关键字的代码上下文，无凭据。

检查项目：

1. 当前镜像 TE 版本、NVFP4BlockScaling / NVFP4Quantizer signature，recipe 是否接受 `backward_override='dequantized'` 和 `nvfp4_4over6='all'`。接受构造参数不等于 GroupedLinear 或当前 custom autograd 已使用该功能。
2. 同一个 production packed 数据，独立解码 tcgen05 swizzled block scales 与 FP4 nibbles，以 FP32 matmul + 原 per-token alpha 对照真实两次 GEMM；不使用重新量化的 oracle 输入。
3. 实际 activation quantizer 反量化结果对 v1 独立公式，以及真实 SwiGLU 输出对相同输入的 PyTorch 参考。
4. production weight scale/decode 辅助函数对 TE tensor 自带 dequantize，检查 scale/layout 理解是否一致。

官方 API 说明（在线核对，非当前镜像版本保证）：
https://nvidia.github.io/TransformerEngine/api/common.html

- `nvfp4_4over6` 在 block 内比较 map-to-4 / map-to-6 候选，按 MAE/MSE 选更小误差；scope 可选 weights/activations/all，默认 none；256/448 全局 scale bound 是另一个配置项。
- `backward_override='dequantized'` 将保存的量化操作数反量化为活动高精度 dtype 后用于 backward。与原始 high_precision 操作数不同；不等于一定在 backward 重做一次量化，也不等于把输出梯度量化为 FP4。
- 当前 Slime 的 custom autograd 直接调用 BF16 expert backward；仅设置 TE recipe 不能假定覆盖这条自定义路径。需要核对保存的是前向 rowwise 表示还是独立 columnwise 表示，以及非线性中间值是否来自实际量化前向。

本阶段只核对接口和算子差异，不切换正式训练 recipe，不声称 four-over-six 或 dequantized backward 已改善 reward。

## 完成结果

Job `2792145`：COMPLETED / 0:0，45 秒；实际读取模型权重并完成全部阶段，输出 STAGE_PROBE_COMPLETE。没有首次真实运行异常；不涉及 refit/optimizer step。

- 当前镜像 TE 为 `2.16.1+c9877beb`。
- 当前 recipe 明确有 `backward_override`，`dequantized` 构造后被保留。实际 `pytorch/module/grouped_linear.py` 的 270 行附近要求使用 fprop quantized layouts 而不 retarget；452 行附近处理反量化 weight 的 dgrad，534 行附近处理反量化 input 的 wgrad。该功能存在于 TE 路径，但 Slime custom BF16 expert backward 不经过它。
- 当前 recipe/quantizer signature 没有 `nvfp4_4over6`，recipe 和 pytorch 源码搜索也没有此符号。虽然构造时传入该参数没有报错，输出 recipe 也没有显示它，因此不能把 acceptance=true 当成成功启用；尚未证明这个旧镜像有 TE 4over6 功能。新版官方文档不能替代当前安装版本的证据。
- 24 个权重：production dequantize helper 对 TE 自带 dequantize 的最大 relative L2 `7.11e-8`，不支持全局 scale 解码明显错误的假设。
- 12 个相同 packed operands GEMM：对独立解码 + FP32 matmul + 原 alpha，relative L2 `0.003169–0.003411`（约0.32%–0.34%）。该残差仍未拆到具体累加/舍入语义，不能宣布 bitwise 一致。
- 6 个 SwiGLU：对相同输入的 FP32 silu/mul 后 BF16 cast 参考，完全一致。
- 12 个 activation quant：对独立量化公式 relative L2 `0.004737–0.008202`；对未量化输入 `0.08007–0.09531`。

这些结果把前轮端到端 expert 的约5% oracle 差异进一步限定到量化表示及数值传播；同 packed bytes GEMM 误差明显较小，SwiGLU 一致。尚未做误差逐项重放，不能断言全部5%已被量化/舍入解释完毕，也没有发现明确 kernel 公式或 scale/layout bug。

后续比较应把 forward 量化策略（标准 / 4over6）与 backward 操作数来源（原始 BF16 / 前向量化值的反量化 BF16）作为两个独立因素。先固定 forward，仅比较 backward；不能只打开一个 TE 环境变量就假设当前 custom MoE 已改为 dequantized backward。
