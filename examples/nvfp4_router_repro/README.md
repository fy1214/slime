# Router autograd 修复与 NVFP4 复现包（给 Mingfa）

生产修复只有 `slime/backends/megatron_utils/alignment/deepgemm_moe_forward.py` 中约44行：保留 SGLang `moe_fused_gate` 的前向，补选中专家概率的 selected-softmax VJP；top-k map 不可微，空 token / scale=0 / 归一化top-1可用，只支持一阶反向。

**原问题：**推理 Triton kernel 的概率输出没有 autograd，aux loss=0 时48层router不更新，也丢失经router回到hidden的梯度。离散索引不求导，不等于选中概率也不求导。此问题在被测FP8和NVFP4共享路径中都存在。

## 已验证什么

| Job | 测试 | 结果 |
|---|---|---|
| 2792992 | 原版/候选router算子16组 | 原版断路，候选forward逐位一致，梯度误差约1e-7 |
| 2793006 / 2793007 | 4n×4g同batch原版/候选 | router从0/48更新到48/48；loss/entropy相同 |
| 2803115 | 直接集成修复，NVFP4和FP8各50组边界 | 100/100 PASS；含top-1严格零梯度 |
| 2803118 | 直接集成版4n×4g | 全部router更新，norm/recompute通过，full Adam保存 |
| 2803142 / 2803143 / 2803144 | 用户要求的三个NVFP4长跑段 | 前两段TIMEOUT，第三段用户取消；该实验模型已按用户要求删除 |

完整模型固定batch entropy均为0.7751415968、loss=-0.00020202773，grad norm从0.32998561到0.67322130。768条审计是16rank×48层副本，不是768个独立层；候选部分本地梯度为0，但hook触发，所有rank上48层参数都更新。完整模型没有保存全部logits进行bitwise证明。

**没有证明reward提高或追平BF16。**初始熵发生在更新前，router backward修复不能改变它。DQ backward是另一项诊断变量；旧版TE未找到four-over-six入口。不要在复现router单变量对照时混入它们。

## 为什么除了脚本还带baseline补丁

PR基于 `fy1214/slime:slime-deterministic-patch-nvfp4` 的 `81d5b66`。当时正式实验snapshot还包含尚未全部上游化的alignment/审计改动。直接使用当前checkout替代它，不是同一个实验。

- `baseline_nvfp4.patch` / `baseline_fp8.patch` 只用于**新建的复现snapshot**，重建当时原版行为，包含有意恢复的无梯度router。
- `router_fix.patch` 在修复组snapshot中重新应用生产修复。
- 这些baseline补丁**不可应用到开发/生产工作树**；`stage.py`只修改新输出目录。
- 先前fresh-start seed修复PR [#12](https://github.com/fy1214/slime/pull/12)当时未合并；baseline补丁带有被测版本，避免新训练因load(-1)失败。不要把它误作router修复内容。
- `historical/` 保存历史脚本、诊断代码和汇总，不应原样执行其中的绝对路径命令。历史状态记录是时间点快照，以本文的取消说明为准。

## 环境与输入（必须自己确认）

验证环境：GB200，4 GPU/node；完整模型4nodes×4GPUs、EP4、GBS256、32prompts×8responses、lr1e-6、TIS2、CLB1、逐层full recompute。沿用原Slurm account/partition/image/mount/缓存/数据，不能缩拓扑后声称完整模型回归通过。

原镜像：`slime-nvfp4-det-gb200-20260905-v6.sqsh`；TE为2.16.1+c9877beb。依赖镜像内匹配的Megatron、SGLang、DeepEP、CUTLASS与miniTransformer。仓库本身不包含镜像。

需要原始BF16 HF权重、NVFP4 HF模型、aligned torch_dist checkpoint、DAPO训练数据、AIME评测数据。固定batch测试还需要经授权取得的 `batch.pt`；结构为 `{"samples": [...]}`，含tokens/response_length等完整rollout字段。**本PR不上传模型、原始response/batch、凭证或W&B日志。**不能用随便造的tokens声称复现了原完整模型数值。

## 1. 安全staging（登录节点可做，不提交任务）

在本PR checkout中执行，仅stdlib，无torch导入：

```bash
python3 examples/nvfp4_router_repro/stage.py \
  --root /YOUR/SHARED/FS/router-repro-run1 \
  --tag mingfa_run1 \
  --fixed-batch /AUTHORIZED/SHARED/batch.pt
```

`--root`必须不存在且在checkout外；`--tag`必须全新。需要时设置`--shared-prefix`和`--netrc`，默认保留原集群共享目录与当前用户netrc路径。不提供batch也能stage并跑router算子gate，但不能跑固定batch模型测试。

审查生成脚本中的image、model/data、mount、account、partition和S3/W&B权限（本包不会做S3上传）。路径不支持空格。stage重写输出根和实验后缀，不会复用旧ckpts/run IDs，不会提交或取消任何job，也不会恢复已删除模型。

下文的`$REPRO`指刚生成的共享输出目录；它不是HOME/workspace根。先检查`squeue -u "$USER"`避免命名和资源冲突。GPU/torch/pytest只在Slurm任务里运行，一次一个测试job。

## 2. 算子gate

```bash
sbatch "$REPRO/investigation/nvfp4_router_audit_20260911_v1/probe.job"
# 完成并检查ROUTER_BACKWARD_GATE_PASS后：
sbatch "$REPRO/investigation/nvfp4_router_fix_20260912_v1/probe.job"
```

第二个job串行执行NVFP4/FP8各50组：tokens=0/1/17/128/513，scale=None/0/.5/1/2，topk=1/8。检查原始forward逐位一致、no-grad推理、非选中项零梯度及参考VJP。历史2803058/2803113的top-1误差在最终补丁中已修，不是通过放宽阈值绕过。

## 3. 完整模型原版/修复配对

```bash
cd "$REPRO/investigation/nvfp4_router_audit_20260911_v1/formal"
bash submit.sh dry-run original
bash submit.sh dry-run fixed
bash submit.sh submit original
# 原版完成后，用其实际job ID：
bash submit.sh submit fixed NEW_ORIGINAL_JOB_ID
```

两组相同checkpoint/batch，expert backward都是original；不生成新rollout、不记录新W&B。两组最终保留完整Adam，检查16rank/router/norm/recompute、step0日志和.metadata；不能只看Slurm COMPLETED。固定归档来自FP8，NVFP4 train-rollout diff=0.076324是本对照预期跨精度差异，CI diff门槛在两组均关闭。

直接集成修复及其后续DQ对照：

```bash
cd "$REPRO/investigation/nvfp4_router_fix_20260912_v1/formal"
bash submit.sh dry-run original
bash submit.sh submit original
# 若要单独研究DQ，在上述通过后再提交，不加入router长跑：
bash submit.sh submit dequantized NEW_ORIGINAL_JOB_ID
```

DQ候选显存较大，尚未验证colocate预算，不直接用于正式长跑。原numeric与packed审计入口也在对应`nvfp4_numeric_diagnostic_*`目录的`probe.job`；v1参数`operator`或`full`，后者仅为手工单GPU前向参考，不代替正式栈。

## 4. 三段NVFP4长跑（只有明确想重跑时才提交）

```bash
cd "$REPRO/investigation/nvfp4_router_long_20260912_v1"
export ROUTER_REPRO_GATE_JOB=NEW_DIRECT_INTEGRATED_ORIGINAL_JOB_ID
python3 submit_chain.py dry-run
python3 submit_chain.py stage
python3 submit_chain.py audit
python3 submit_chain.py release
```

Gate必须是本次第3节的直接集成版完整模型job；stage已去掉作者历史2803118的硬编码。三段各5小时，fresh-start后同一新W&B run完整Adam续跑，后两段afterany（允许TIMEOUT）。目标1600rollouts是上限，15小时不保证跑满。每10步保存；**原脚本会自动删除此新namespace中超过两代的已完成checkpoint**，保留进行中的保存。不要把SAVE改到旧实验目录。

W&B项目沿用`shawnzzz/slime-deterministic-gb200`；无该项目写权限时，修改生成的driver里的team/project为自己的，并在两组保持一致。比较使用train/step与rollout/step，raw_reward而非归一化rewards，同时看length与AIME。三个job是同一实验续跑，不是三个独立seed。

## 交付边界

本PR不自动重启已取消训练，不恢复已删除checkpoint，不更改镜像或TE版本，不将历史长跑当作质量提升证据。复现日志/输出生成在新的共享目录，不应提交回Git。
