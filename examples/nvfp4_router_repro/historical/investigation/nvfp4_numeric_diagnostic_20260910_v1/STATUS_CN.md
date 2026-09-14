# NVFP4 数值诊断（2026-09-10）

## 提交状态

- 算子诊断 job `2791917`：COMPLETED / 0:0，59 秒；1 node、1 process、GPU 0、4 allocated CPUs、各线程池 1。
- 固定 token 完整模型参考 job `2791935`：COMPLETED / 0:0，2 分 29 秒；通过 `afterok:2791917` 串行执行，同样单进程／单 GPU。
- 均进入模型权重加载和实际计算，分别输出 OPERATOR_COMPLETE / FULL_FORWARD_COMPLETE；没有 traceback 或非有限结果。两者都不涉及 optimizer step/refit。成功退出不等于所有数值对照一致，见下文。
- 没有修改或取消正式训练链，没有创建 W&B run、optimizer update 或训练 checkpoint。

## 可复现入口

脚本：`diagnose.py`；提交：`probe.job`。容器内完整 driver：

```
python3 /lustre/fsw/general_sa/shuazhang/python_space/verl_for_nvfp4_20251031/investigation/nvfp4_numeric_diagnostic_20260910_v1/diagnose.py --phase operator
python3 /lustre/fsw/general_sa/shuazhang/python_space/verl_for_nvfp4_20251031/investigation/nvfp4_numeric_diagnostic_20260910_v1/diagnose.py --phase full
```

两条命令分别位于独立调度 job，按 dependency 串行运行。每条 job 将结果逐项写入 `results_<jobid>.json`，并输出 `<jobid>.out/.err`。

- 镜像：`/lustre/fsw/general_sa/shuazhang/images/slime-nvfp4-det-gb200-20260905-v6.sqsh`。
- 被测冻结源码：`investigation/determinism_initial_gap_20260909/seed_formal_v2/nvfp4/slime_nvfp4`，mount 为 `/root/slime`。
- 复用同 bundle 的 `jobs/nvfp4_prep_env.sh`，共享 mount `/lustre/fsw/general_sa/shuazhang`；不挂载 W&B 凭据。
- `CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=1 CUDA_LAUNCH_BLOCKING=1`。正式 NVFP4 原合同已有 CLB=1；本诊断显式保留。
- 单进程 srun 入口派生自成功诊断 `2775669/2775742`，改用当前 NVFP4 镜像及 source/prep。
- Python AST、bash -n、sbatch --test-only 已执行。调度器检查不是 GPU 测试通过证据。

提交时脚本 SHA256：

```
1f8ae0e86cd8e1028166ca2ab68730416556de359bf3d0727273103ec2aeb7e0  diagnose.py
e3750a5f484a09031c7a091d0543c991fad8853ea1cebd8ea9424423d2fc5186  probe.job
```

## 实验设计与解释边界

算子诊断读取原始 Qwen3-30B-A3B-Base HF BF16 权重，第 0/23/47 层、expert 0/17/63/127；输入首先为固定 seed 的 BF16 随机张量。测试 ragged counts `[1,17,0,129]` 和 aligned counts `[128,128,128,128]`。前向比较真实 production NVFP4 wrapper 与独立 E2M1 RTNE + E4M3 block-scale（16 elements）参考、BF16 参考；TE 权重量化单独与该独立公式比较。参考不调用生产 quantizer/dequantizer。

反向固定输入、专家权重、router probabilities 和上游随机向量；production BF16 backward 分别对照 native BF16 autograd 和明确约定的 STE（量化器 identity derivative、scale stop-gradient；非线性使用量化前向的中间值）。报告 input/router probability/fc1/fc2 的 relative L2、cosine、norm ratio、max abs。STE 是一个对照定义，不是离散量化函数的真实导数，也不是已证明更好的训练方案。

完整模型参考读取已归档 batch 的 sample 0/8，各取至多前 512 tokens，记录 token hash 与 response 起点。从同一 BF16 HF 权重逐层运行四条路径：BF16、仅权重量化、独立 W4A4 oracle、production W4A4 kernel。全部使用同一 eager attention/RMSNorm/残差/路由组合，并分别按各模式 logits 重新计算路由。记录逐层 hidden 误差、top8 route overlap、最终 logits KL/top1 agreement/entropy/NLL，以及 response-token 子集指标。第 0/23/47 层还对真实 BF16 normalized activations 做局部 backward 诊断。

这是单 GPU 独立参考模型，未复现正式 4n×4g Megatron/DeepEP/FA4/FP32 residual 的完整运行语义；不能将其 entropy 数字直接等同于 W&B entropy，也不能将通过当作正式分布式回归证明。没有 rollout reward 或 optimizer step。局部反向的上游向量固定但为随机值，不是完整模型 RL loss 梯度。

算子阶段的进程成功仅代表执行完成；没有预先以任意误差阈值宣称 correctness。必须查看各 reference 的误差量级和 finite 标记后判断，必要时解码生产 packed bytes，拆分 quantization 与 GEMM 差异。若独立参考与 kernel 差异明显，先定位参考语义／kernel，不做 reward 因果归因。若算子一致而逐层误差累积，则后续应在原正式拓扑回放相同 batch 验证；不会根据本诊断自动替换生产 backward。

## 已完成结果

两个 job 的运行时脚本 hash 均与上述记录相同。

- production backward 对 native BF16 autograd：合成输入最大 relative L2 为 5.68e-6；真实激活样本最大 1.31e-5。没有在这些局部样本中发现 BF16 backward 公式/梯度连通性异常。
- 对明确约定的 STE：真实激活 input/fc1/fc2 梯度 relative L2 为 12.5%–24.2%，cosine 为 0.971–0.993；norm ratio 约 1.003–1.020。梯度方向近似存在偏差，但这里没有观察到倍数级范数放大，也没有验证 STE 训练一定更好。真实激活只命中了两层的 expert 0，不能声称三层全部覆盖。
- production NVFP4 expert 输出相对 BF16：relative L2 15.7%–22.7%。相对独立量化 oracle 仍有 4.70%–5.93% 差异。
- TE 权重 quantizer 与独立公式的反量化值最大 relative L2 2.57%。因此不能把上述 oracle 当作已完成 bitwise 校验的金标准，也不能据 5% 差异判定 kernel bug；下一步需要逐阶段比较同一 packed bytes、scale、GEMM 及 SwiGLU，排除参考 rounding/scale 语义差异。

同一两个序列的前 512 tokens，共 665 个 response 预测位置：

| mode | response entropy | response NLL | 全位置 KL(BF16||mode) |
|---|---:|---:|---:|
| BF16 | 0.178411 | 0.153590 | 0 |
| weight_only | 0.183628 | 0.153545 | 0.059366 |
| 独立 W4A4 oracle | 0.194385 | 0.161222 | 0.127555 |
| production NVFP4 kernel | 0.185956 | 0.157261 | 0.103963 |

production NVFP4 的 response entropy 高约 4.23%；独立 W4A4 参考也有增高。支持量化前向本身能产生熵差，但只有两段短序列、参考模型与正式 stack 不同，不能据此分解 W&B 初始差距的百分比，更不能断言全部 reward 差距由精度决定。第 47 层 NVFP4 对 BF16 top8 expert 集合重合率约 89.9%。

当前判断：未复现旧式 backward 断路／BF16 backward 公式错误；确认存在量化前向差异和 surrogate-gradient 偏差。production 与独立 oracle 的残留差异尚未定位，不能宣布实现无 bug。没有提交新训练或修改生产实现。
