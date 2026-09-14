# NVFP4 训练差距调查：背景、证据、修复与后续计划

日期：2026-09-12（America/Los_Angeles）  
用途：团队技术交流。研究对象为当前 Slime / Qwen3-30B-A3B-Base、GB200 的 FP8 deterministic 与 NVFP4 实现；不泛指所有 NVFP4 实现。

## 一页结论

**当前不能把 NVFP4 的 reward 落后简单归结为 4bit 精度上限。**本次调查发现了一个确定的 router 反向缺陷，并通过原训练拓扑的完整模型对照验证修复。与此同时，量化前向本身确实带来数值扰动；当前 BF16 surrogate backward 与实际 W4A4 前向也不一致。这三件事必须分开评价。

1. **确定的实现 bug：**SGLang 推理 top-k kernel 的概率输出没有 autograd，导致训练的 router 权重失去梯度。它同时存在于被审计的 FP8 和 NVFP4 路径。
2. **完整模型已验证：**相同 checkpoint、相同归档 batch、4 nodes × 4 GPUs，仅补 router 概率的反向，48 层 router 从全部不更新变为全部更新；单步 loss、entropy 等前向指标相同。
3. **不要误读梯度范数：**补 router 后单批总梯度从 0.330 增至 0.673；此前反量化 backward 对照则让范数下降 18.4%。两者都不能单靠范数推断 reward 优劣。
4. **尚未证明：**修复后长期 reward 是否追平 BF16、router 缺陷占差距多少、dequantized backward 与 four-over-six 各能改善多少。没有提交或宣称这些长期质量结果。
5. **当前交付：**已把 router 修复写入 FP8/NVFP4 开发代码，并建立新的修复快照与边界回归测试。旧长跑及其冻结代码未中途切换，以免污染已有曲线。

## 1. 背景：为什么继续调查 NVFP4

此前 FP8 修复取得明显改善，但 NVFP4 仍表现为初始 entropy 较高、早期 grad norm 较高、reward 低于 BF16。因此问题不是“NVFP4 能不能运行”，而是要区分实现错误、量化误差、训练梯度近似与采样轨迹差异。

本轮之前已修复两个独立问题：

- **Norm / checkpoint recompute：**FP32 residual 通过 tensor 附加属性携带，在重算时丢失，导致 input RMSNorm 的 autograd 路径缺失；此前 48 层中 47 层没有 norm 梯度。改为显式传递 residual 后，48 层 norm 梯度恢复，重算前向对照通过。
- **确定性采样 seed：**原来跨 prompt 重复使用一小组固定 seed；改为按 sample 区分的确定性 seed。它涉及采样设计，不能与数值精度变量混为一谈。

Router 断梯度是此后新发现的第三个问题，不是前面 norm 修复失败或同一个 bug 的重复描述。

### 训练曲线的口径

项目为 `shawnzzz/slime-deterministic-gb200`。主要参考 runs：

- BF16：`gb200_30B_v3_bf16_4N_B256`
- FP8 deterministic：`det_seed_unique_formal_0910_v2`
- NVFP4：`nvfp4_seed_unique_formal_0910_v2`

训练指标按 `train/step`，rollout 指标按 `rollout/step` 对齐；reward 使用 `rollout/raw_reward`，不是约为零的归一化 `rollout/rewards`。下表是前轮调查的历史窗口统计，不把它当成修复后结果。

| 窗口 / 指标 | BF16 | FP8 deterministic | NVFP4 |
|---|---:|---:|---:|
| step 0 entropy | 0.718599 | 0.723869 | 0.917362 |
| step 0–9 grad norm 均值 | 0.624333 | 0.430222 | 1.294161 |
| step 100–149 grad norm 均值 | 0.209740 | 0.181511 | 0.162159 |
| step 0–9 raw reward 均值 | 0.200781 | 0.183984 | 0.153516 |
| step 100–149 raw reward 均值 | 0.425156 | 0.421094 | 0.403359 |
| step 200–219 raw reward 均值 | 0.453906 | 0.448047 | 0.427344 |

这些窗口支持“早期梯度尖峰”，而不是“梯度一直偏高”。BF16 是历史参考，以上不是多 seed、完全随机化的因果实验；各 run entropy 在各自产生的 response 上计算，输入轨迹也不同。当前 entropy coefficient 为 0，高熵不是 entropy loss 主动推动的结果。

**2026-09-12 在线复核：**上述历史窗口的 grad/reward 数字已重新核对一致。API 当时 FP8/NVFP4 最新训练 step 分别为 883/590，BF16 历史为 1536；W&B state 分别为 running/running/failed（不是根据该 state 判断失败原因）。共同的 step 300–349 窗口如下，仍全部是 router 未修复的旧 runs：

| 指标 | BF16 | FP8 deterministic | NVFP4 |
|---|---:|---:|---:|
| Raw reward | 0.497969 | 0.513438 | 0.480469 |
| Response length | 1056.41 | 1282.34 | 1381.65 |
| Entropy | 0.092256 | 0.089694 | 0.105790 |
| Grad norm | 0.155634 | 0.182374 | 0.120023 |

NVFP4 在此窗口 reward 较低但 response 更长、grad norm 更小；不能以梯度更小或答案更长等同于质量更好。本轮未刷新独立评测 accuracy，不把训练 raw reward 当成独立 benchmark accuracy。原始曲线：[BF16](https://wandb.ai/shawnzzz/slime-deterministic-gb200/runs/gb200_30B_v3_bf16_4N_B256)、[FP8](https://wandb.ai/shawnzzz/slime-deterministic-gb200/runs/det_seed_unique_formal_0910_v2)、[NVFP4](https://wandb.ai/shawnzzz/slime-deterministic-gb200/runs/nvfp4_seed_unique_formal_0910_v2)。

## 2. 调查顺序与排除逻辑

| 阶段 | 要回答的问题 | 方法 | 能支持的结论 |
|---|---|---|---|
| 指标审计 | 现象是否准确？ | 统一 metric family 的 step、reward 口径和窗口 | 梯度主要早期偏高，不能只看末值或 summary |
| Expert 算子 | W4A4 前向或 BF16 backward 是否明显算错？ | 实际模型权重、不同 token 分布，与独立参考对照 | 量化确有误差；原 backward 近似 BF16，而非实际量化前向的 STE |
| 固定 token 模型前向 | 不同采样之外还有熵差吗？ | 同两个序列，BF16 / 权重量化 / W4A4 参考 / production kernel | 固定输入下仍存在前向量化扰动 |
| Packed 表示审计 | 先前约 5% 的差异是否是 GEMM bug？ | 固定相同 packed 值和 scale，独立解码再做参考 GEMM | 未见明显 layout / scale / GEMM 公式错误 |
| 反量化 backward | 保持前向，改梯度定义有什么变化？ | 算子 autograd gate + 完整模型单步配对 | 范数改变，尚无 reward 因果证据 |
| Router 深入审计 | 全零 router 是 buffer 读取问题还是断路？ | requires_grad、tensor hook、两种 buffer、完整参数更新 | 真正的反向连接缺失；完整模型候选修复成功 |

### 2.1 Expert 与固定 token 前向

取真实模型第 0、23、47 层，experts 0、17、63、127，覆盖不均匀 token 数 `[1,17,0,129]` 和对齐 `[128,128,128,128]`。原 backward 对 BF16 参考误差很小，但与使用量化前向中间值的明确 STE 定义有可观差异；这说明两者优化近似不同，不能把 BF16 精度的计算自动等同于“与前向一致”。

独立单 GPU 手工完整模型参考使用同两个序列、每个取前 512 tokens，共 665 个 response 预测位置：

| 模式 | Response entropy |
|---|---:|
| BF16 | 0.178411 |
| 仅权重量化 | 0.183628 |
| 独立 W4A4 参考 | 0.194385 |
| Production NVFP4 kernel | 0.185956 |

Production 相对 BF16 约高 4.23%，说明固定输入下量化也会改变分布。但这是小样本、手工 eager 参考，不等于完整 Megatron / FA4 / FP32 residual / DeepEP 训练栈，不能把 4.23% 外推为正式曲线差距的解释比例。

### 2.2 为什么没有把早期 oracle 差异直接判为 kernel bug

初步 Expert 输出对独立量化参考相差约 4.7%–5.9%，但双方量化表示并非完全相同，不能据此断言 CUTLASS 错了。随后固定完全相同的 packed 操作数和 scale，逐段核查：

- 24 个权重解码对照 TE 自带 dequant，最大 relative L2 约 `7.1e-8`。
- 12 次同 packed 操作数 GEMM，对 FP32 参考的 relative L2 约 `0.32%–0.34%`。
- 6 次 SwiGLU 对照完全一致。

结论是：**被测范围内未发现明显 scale/layout/GEMM 公式 bug**。这不是全输入空间证明，也没有排除极端值或其他分布式分支；先前 5% 差异的全部传播份额未被逐项定量分解。

## 3. TE dequantized backward 与 four-over-six

实际镜像 TE 版本为 `2.16.1+c9877beb`。

**Dequantized backward：**该版本有 `backward_override="dequantized"`。其重点是反向使用前向保存的量化操作数反量化后以高精度计算，不应一概描述成“反向一定再做一次新的量化”。当前 Slime custom MoE 使用自己的 autograd，绕过 TE recipe；只改 TE 配置不会自动改变这条路径。

原始 custom backward 使用原始 BF16 权重、输入重算，其中非线性中间值也可能偏离真实量化前向。诊断候选保存真实前向 packed 输入、中间值、权重 scale 和相应前向输出，反量化为 BF16，再用 identity STE、scale stop-gradient 定义反向。

- 算子 gate：6 组前向逐位相同，84 项独立 autograd 检查，最大 relative L2 `3.02e-5`。
- 完整 4n×4g：loss、entropy 相同，grad norm `0.329986 → 0.269381`，降低 18.37%。
- 候选使用 per-expert PyTorch matmul，原版使用 grouped/chunked DeepGEMM，因此存在累计/舍入方式差异。
- 保存中间值增加显存，尚未验证正式 colocate 的预算和吞吐，不直接用于长跑。
- 该对照发生在 router 修复之前，不能直接外推为 router 修复后的组合效果。

**Four-over-six：**当前安装版本的 recipe/quantizer 签名及检查的 Python 源码中未找到对应入口。某些构造函数接受未知 kwargs，不代表功能已启用。新版 [TE 官方 recipe 文档](https://nvidia.github.io/TransformerEngine/api/common.html)提供 `nvfp4_4over6`，但新版文档不等于当前旧镜像支持，更不等于 custom kernel 已使用。后续应将它作为单独前向变量，并同时审计训练和 rollout 的表示与 kernel。

## 4. 核心发现：Router 为什么没有梯度

Router 应根据 hidden state 生成 logits，选择 top-k 专家，并用选中专家的概率加权专家输出。**离散选择的索引不求导，不意味着选中概率也可以 detach。**

当前 `_sglang_unbiased_softmax_topk_routing` 调用推理用 `sglang.jit_kernel.moe_fused_gate`。实际 runtime 源码用 `torch.empty` 分配输出，再由 Triton 写入；没有 autograd wrapper。因此 logits 虽然带梯度，返回的 `topk_weights` 不带梯度，写入 dense `routing_probs` 后仍不带梯度。

后果有两项：

1. 当前辅助 loss 系数为 0，router 权重没有其他训练梯度来源，因而不学习。
2. 通过 router logits 回传到 hidden state 的梯度分支也丢失。

专家参数和其他路径仍能训练，hidden state 变化也能改变专家选择，所以不能说整个模型不学习或路由分配永远固定。被审计的 FP8 与 NVFP4 路径都存在此问题，因此它不能独自解释 NVFP4 相对 FP8 的全部差距。

### 修复方法

保留 SGLang 推理 kernel 的原始前向、专家索引、概率舍入和 replay 捕获行为，只补概率的解析 VJP。对当前选中的专家集，若 `p = s × softmax(z_selected)`：

```
dL/dz = p × (dL/dp − sum(p × dL/dp) / s)
```

未选中位置 `p=0`；`s=0` 或归一化 top-1 单独返回严格零梯度；routing map 标记为不可微。Top-1 的数学概率为常量，但 kernel 的 1 ULP 舍入可能让通用公式留下极小伪梯度，故显式处理。修复支持当前训练所需的一阶反向，不宣称二阶导数支持。推理/no-grad 路径继续直接调用原前向。

### 算子与完整分布式证据

第一轮 gate 覆盖 1/17/128/513 tokens 与 None/1/0.5/2 scale，共 16 组。原版全部概率不带梯度，候选前向逐位一致，梯度对原生 selected-softmax 误差约 `1e-7`。

完整模型测试沿用原 4nodes×4GPUs、EP4、GBS256、32 prompts×8 responses、Adam lr `1e-6`、TIS2、full recompute、CLB1，同一 checkpoint 和归档 batch。只做一个 optimizer step，不重新采样，不混入 DQ 或 four-over-six。

| 检查 | 原版 2793006 | 候选 2793007 |
|---|---:|---:|
| 任务结果 | COMPLETED / 0:0 | COMPLETED / 0:0 |
| 耗时 | 5分23秒 | 5分28秒 |
| Router weight hook 触发记录 | 0/768 | 768/768 |
| 非零本地 router 梯度记录 | 0/768 | 720/768 |
| Router 参数非零更新记录 | 0/768 | 768/768 |
| Entropy | 0.7751415967941284 | 0.7751415967941284 |
| Loss | -0.0002020277315750718 | -0.0002020277315750718 |
| Grad norm | 0.32998561126955706 | 0.6732213038389129 |

768 是 16 rank×48 层副本记录，不是独立的 768 层。候选 rank15 本地梯度仍为零但 hook 触发；分布式更新后所有 rank 的 48 层 router 均有一致的逐层参数更新范数。没有审计该 rank 的 advantage/mask，不能擅自指定本地零值原因。这个细节不改变“原版断路、候选连接和全局更新恢复”的证据。

两组 norm/recompute gate 通过，48 个 norm 非零梯度，重算输出采样差为 0；每组 3840 条所选参数记录无缺失/非有限梯度。完整 Adam checkpoint 已保存，但没有独立 reload 验证。归档 batch 来自 FP8，因此对 NVFP4 的 train-rollout abs logprob diff 为 0.076324 是预期跨精度差异；两组相同，不能把它误判为候选破坏对齐。

## 5. 当前修复落地状态

2026-09-12 已修改两个开发 checkout 的 `slime/backends/megatron_utils/alignment/deepgemm_moe_forward.py`：

- `infix.AI/slime_sampling_seed_nvfp4_20260909`
- `infix.AI/slime_sampling_seed_fp8_20260909`

从被测正式快照另行派生 `investigation/nvfp4_router_fix_20260912_v1/code/{slime_nvfp4,slime_det}`，同样直接集成修复，不再依赖诊断 monkeypatch。开发代码与正式冻结代码有历史差异，因此后续完整模型验证使用新正式派生快照，而不是用更旧开发 checkout 代替训练基线。

新增回归 `test_router.py`：两套代码各 50 组，覆盖 0/1/17/128/513 tokens、None/0/0.5/1/2 scale、top-1/top-8；检查前向逐位一致、no-grad 推理、非选中项零梯度以及参考解析梯度。初次 job `2803058` 和增加失败定位输出的 `2803113` 均以断言失败结束，首次真实异常为 top-1、scale=2 时通用 VJP 存在 `2.38e-7` 的伪梯度；不是调度/模型初始化失败。加入严格 top-1 零梯度分支后，`2803115` **COMPLETED/0:0，47秒，FP8和NVFP4各50/50 PASS**；最大梯度 relative L2 `1.75e-7`，top-1及零缩放梯度严格为零。没有放宽检查阈值。登录节点没有运行 torch / pytest / GPU 测试。

两套开发代码、两套新正式派生快照及后续完整模型快照的三个 router 修复函数经 AST 提取核对完全一致，SHA256 为 `e67792fa914015426d76c2b77f931ea1132ae692bbd6cdf313c1c00c5136fc04`。这是修复函数文本的 hash，不是整个仓库 hash。

**继续分析已开始：**新完整模型 job `2803118` 已提交，使用直接集成的 router 修复 + 原始 BF16 expert backward，保留前轮成功的4n×4g合同、固定batch及完整审计。后续 dequantized 组已准备并完成 dry-run，待本组通过后再串行提交；两组都不再依赖 router monkeypatch。发布本版报告时，新完整模型结果尚待产出。

已有 seed-formal 长跑没有切换到修复版，所以继续产生的旧曲线不能标注为“router 已修复”。没有 push、PR、删除 checkpoint 或重写旧实验历史。

## 6. 后续研究：先做什么、如何判定

1. **工程回归：**新直接集成修复的边界 gate → 原拓扑固定 batch 验证。确认前向一致、router 更新恢复、norm/recompute 不回退、完整 Adam 保存。
2. **修复后的 backward 对照：**两组均修 router，再比较原 BF16 backward 与保存量化操作数的 dequantized backward。固定 batch 保持相同，审计梯度方向/范数、完整更新和内存，不把范数下降视为质量提升。
3. **长期质量因果对照：**在新的独立 run / checkpoint namespace 下做 fresh-start；保持模型、数据、seed 设计、训练合同一致，比较 router 修复后的 FP8/NVFP4 与 BF16。不能把旧链中途改算法的曲线作为干净对照。
4. **再研究前向精度策略：**固定 tokens 下扩大前向诊断覆盖，单独引入 four-over-six 或其他量化策略；必须记录实际版本和启用证据。

质量判据同时看 raw reward/accuracy、response length、entropy、train-rollout mismatch 与稳定性；固定 batch 单步、梯度范数、对齐差最小都不能代替新 rollout 的多步/多 seed 质量评估。当前未宣称上述长期对照已经开始或完成。

## 7. 可复查证据索引

以下为共享 workspace 内相对路径；报告分享对象只包含本文，不上传模型权重、原始 response 或凭证。

| 目录 / Job | 内容 |
|---|---|
| `nvfp4_numeric_diagnostic_20260910_v1` / 2791917、2791935 | Expert 数值、手工固定 token 模型前向 |
| `nvfp4_numeric_diagnostic_20260910_v2` / 2792145 | 相同 packed 操作数分段审计、TE runtime 功能检查 |
| `nvfp4_dq_backward_20260911_v1` / 2792416 | DQ 算子 autograd gate |
| 同上 `formal` / 2792447、2792449 | Router 未修复时的原版/DQ完整模型单步 |
| `nvfp4_router_audit_20260911_v1` / 2792992 | Router 原始断路及候选算子 gate |
| 同上 `formal` / 2793006、2793007 | 原版/候选完整模型 hook、buffer、完整更新 |
| 同上 `full_model_summary.json`、`summarize.py` | 机器可读汇总及仅 stdlib 的复算入口 |
| `nvfp4_router_fix_20260912_v1` / 2803115 | 直接集成修复、双路径50+50边界回归通过 |
| 同上 `formal` / 2803118 | 直接集成修复的完整模型回归，已提交、待结果 |

每个完整模型实验保留 snapshot、submit.sh、driver audit、Slurm 日志和独立 checkpoint。诊断均区分是否进入模型初始化、首个训练 step 和保存，不只根据调度器最终状态判断代码正确性。

## 最终判断

**已证实的 bug 应先修；尚未分离的精度代价不能先下定论。**Router 修复是必要的训练正确性工作，但不改变更新前的初始前向，因此不可能直接消除 step0 熵差。当前最合理的解释是“量化前向扰动 + 不同 backward 近似 + 已发现的实现缺陷 + 采样轨迹差异”共同影响结果，其长期 reward 贡献尚待干净对照分离。
