# 固定前向的反量化 backward 对照

## 已完成的算子验证

Job `2792416`：COMPLETED / 0:0，50 秒，GB200、正式 NVFP4 v6 镜像，1 process、GPU0、CPU/BLAS threads=1。完整执行命令见 `probe.job`，container driver 为本目录 `probe.py`。输出 `DQ_GATE_PASS`，没有真实运行异常。

- 真实 Qwen3 BF16 expert 权重，第0/23/47层，experts 0/17/63/127。
- 每层 ragged `[1,17,0,129]`、aligned `[128,128,128,128]` 两组。
- 6组 candidate/original 前向逐位相同。
- 84项 candidate/独立 autograd 梯度检查通过；最大 relative L2 `3.015631955349818e-5`，预置门槛 `1e-3`。
- 这里的“通过”是相对于明确定义的 surrogate，不是离散量化函数真实导数，也不代表 reward 已改善。

被测 candidate `dq_backward.py` SHA256：
`b52a6014c39adda1e768c844242b1a05bb202b465598230ea8cb85d2b163f038`。

候选保存实际 production forward 的 packed input/intermediate、weight scales，以及真实 FC1/FC2 输出。反向从这些 rowwise 表示解码到 BF16；非线性导数使用实际前向 gate/up，router probability 梯度使用实际前向未加权 FC2 输出。量化和 cast 使用 identity STE，scale stop-gradient。没有重新调用独立的量化公式，没有开启 four-over-six，也没有改变 CUTLASS 前向。

当前算子 gate 检查的是非 deferred router-probability 分支；正式 DeepEP deferred combine 分支仍需要完整模型验证。候选为诊断实现，使用 per-expert BF16 PyTorch matmul，生产原版使用 grouped/chunked DeepGEMM BF16 backward，因此还存在 GEMM 实现及累计/舍入方式的差异。局部 autograd 对照误差小不能自动证明所有正式形状上的数值等价。尚未优化候选显存/性能，不用于正式长跑。

## 原拓扑固定 batch 配对

新 snapshot：`formal/slime_nvfp4`，从 `investigation/determinism_initial_gap_20260909/seed_formal_v2/nvfp4/slime_nvfp4` 复制，加入两个诊断 plugin；生产目录未修改。

- 原始 backward：job `2792447`，EXP `nvfp4_dq_original_fixedbatch_0911_v1`。
- 反量化 backward：job `2792449`，EXP `nvfp4_dq_dequantized_fixedbatch_0911_v1`；`afterok:2792447` 串行。
- 两组均已完成模型初始化、首个 optimizer step 和 checkpoint 保存：2792447 COMPLETED/0:0，5分24秒；2792449 COMPLETED/0:0，5分40秒。
- 入口：`bash formal/submit.sh submit original` 与 `bash formal/submit.sh submit dequantized 2792447`；两组均先 dry-run。解析的完整 driver 参数分别保存在 `formal/original_driver_audit.txt` 和 `formal/dequantized_driver_audit.txt`。
- 复用历史成功 NVFP4 `2776093` / 当前 seed formal 的 image/mount/prep/driver，4nodes×4GPUs、EP4、GBS256、32×8、Adam lr1e-6、TIS2、CLB1、full逐层recompute。
- 同一初始 aligned BF16 checkpoint 与 NVFP4 HF 模型路径，详见 submit.sh；同一归档 `determinism_backward_audit_20260909/full_model_fixed_v1/batch.pt`，只做一次更新。不是从某组更新后的 checkpoint 开始。
- 两组均关闭 eval，CLI 移除 --use-wandb，不建立 W&B run；使用同一固定 rollout，不启动新的 SGLang 采样。
- 归档 batch 来自 FP8 rollout，故本次关闭 train/rollout diff CI 门槛（否则会把预期跨精度差异误作失败）；TIS保持。两组内部前向／entropy 应相同，仍保留原 norm/recompute gate。
- 两组最后均保存独立 full-Adam checkpoint，没有 --no-save-optim，没有 retention/prune；这是诊断单步 checkpoint，不能作为原正式训练链续跑。
- 每个 rank 记录所选 norm/router/QKV/local expert0 的 full gradient norm、finite 标记，以及前8192元素梯度样本和低精度参数更新。样本 cosine 不代表完整参数的 cosine；主梯度非零也不保证低精度参数立即变化。
- 正式单步完成后需要核对首次真实异常、norm/recompute gate、16个rank审计、前向指标一致性、梯度/更新差异和完整checkpoint；仅 Slurm COMPLETED 不能证明候选更好。

本轮未修改、取消或切换现有正式训练链。尚未提交 reward 长跑。

## 完整模型配对结果

两组均有16个rank审计文件，每组3840条参数记录，无缺失buffer和非有限值。两组原有norm/recompute gate都通过，rank0的48个norm均有非零梯度，重算输出采样最大差0。两组optimizer返回True；各自 `iter_0000000/.metadata` 与 latest=0 存在，日志显示保存正常完成。尚未独立加载checkpoint验证恢复。

| 指标（同一归档batch） | original | dequantized |
|---|---:|---:|
| entropy | 0.7751415967941284 | 0.7751415967941284 |
| loss | -0.0002020277315750718 | -0.0002020277315750718 |
| train-rollout abs logprob diff | 0.07632403820753098 | 0.07632403820753098 |
| grad norm | 0.32998561126955706 | 0.2693813294023353 |

报告的前向指标完全相同，总grad norm降低18.37%。这仅是一批固定输入的梯度变化，不是reward提升或梯度更正确的证明；较小的梯度范数没有单调质量含义。abs logprob diff非零是归档FP8 rollout对NVFP4训练的预期跨精度差异，不是此次候选破坏对齐的证据。

有非零原版buffer的所选参数，DQ/original norm比中位数：input norm .9865、QKV .9881、FC1 .9902、FC2 .9891；范围分别 .351–1.673、.390–1.100、.448–2.089、.548–2.074。统计包含DP副本/本地buffer，并非全局去重后各层独立样本。

另外，两组全部768条router main_grad记录均为0；部分其他rank的norm/QKV buffer也为0。现有审计优先读取main_grad，未同时记录.grad或autograd hook，故不能将这些零值直接判定为router断梯度；必须核对分布式/precision-aware optimizer的buffer所有权与实际梯度路径。这项未决问题使我们不能声称“完整模型所有梯度正常”。

候选显存占用明显增加（例如保存前某rank used约159GB，原版示例约89GB；非同rank对齐的峰值统计），当前只适合debug-train-only诊断。正式colocate训练的显存预算和性能尚未验证，不能直接切换长跑。
