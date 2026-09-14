# 2026-09-12 修复与后续对照交接

## 已完成

- 两套开发 checkout 直接补入 router selected-softmax 一阶 VJP；原始 forward 原样保留，推理绕过 autograd。
- 两套新正式派生快照具有同一修复；没有改正在运行的 seed-formal 冻结代码。
- 修复覆盖空 token、scale=0 和 renormalized top-1 严格零梯度。
- 最终 gate 2803115 COMPLETED/0:0，47秒，两套代码各50/50 PASS。
- 之前2803058、2803113失败是top-1的1 ULP前向舍入造成约2.38e-7伪梯度；已修分支，没有放宽阈值。
- git diff --check 与 bash -n 通过；修复三函数文本跨五份代码hash一致。
- 同事报告已上传HTML与Markdown到 s3://temp/nvfp4-investigation-20260912-v1/；HTML HTTP200且逐字节一致，签名链接见UPLOAD.md，有效至2026-09-19 08:49 PDT。报告包含W&B按metric family轴在线复核的数据和状态边界。

## 当前实验（2026-09-12更新）

2803118已COMPLETED/0:0，5分27秒。模型初始化、首步、完整Adam保存完成；48norm与16rank router hook/完整更新gate通过，loss/entropy/grad norm与2793007一致。用户随后要求router修复版NVFP4三个长跑job，已在独立目录`nvfp4_router_long_20260912_v1`提交并release：2803142 → 2803143 → 2803144（3×5h，fresh-start，后两段完整Adam续跑）。DQ组仍未提交，不将其混入此次长跑。

以下为之前提交时记录：

- Job 2803118，EXP nvfp4_routerfixed_dq_original_0912_v1，当前PENDING/Priority。
- Snapshot：本目录 formal/slime_nvfp4。
- 完整driver：formal/original_driver_audit.txt；入口 bash formal/submit.sh submit original。
- 容器、mount、原模型/数据、4n×4g、EP4、GBS256、CLB1、full recompute与前次成功2793007相同。
- 改变：router不再通过诊断monkeypatch，而在源码直接集成；expert backward保持original。
- 日志：formal/logs/2803118.{out,err}。结果：formal/results/nvfp4_routerfixed_dq_original_0912_v1。
- 当前未进入模型初始化/首次step，无真实异常可供归因；预计开始时间会随调度变化，不是保证。

## 下一项（尚未提交）

先确认2803118首步、norm/recompute gate、16rank router审计与完整Adam保存，再执行：

```
bash investigation/nvfp4_router_fix_20260912_v1/formal/submit.sh submit dequantized 2803118
```

该入口已经dry-run，driver在formal/dequantized_driver_audit.txt。提交前重新检查当前用户test jobs、namespace与scheduler dependency。保持一次只运行一个测试任务，不与无关正式训练链强制串行。

两组均源码修router；仅DQ_VARIANT=original/dequantized不同。router_full_audit不再覆盖DQ_VARIANT。对照前向指标、梯度/参数更新、显存；与前轮router未修时的DQ效果分开解释。没有新reward长跑，不从诊断单步checkpoint续接正式链。

发布到S3的v1是时间点快照。后续结果用新版本对象，不覆盖v1；旧限时链接继续对应原报告内容。
