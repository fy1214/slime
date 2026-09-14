# NVFP4 router-fixed 长跑：三个续跑段

## 已停止并删除本实验模型（用户要求）

2026-09-12美西23:07后执行：2803142、2803143此前已TIMEOUT；取消仍运行的2803144，并等待其退出COMPLETING、确认三job均不在队列。删除本目录`ckpts/nvfp4_router_fixed_formal_0912_v1`，包含iter_0000159、iter_0000169、latest指针及rollout续跑状态，删除前占用约683GiB。已确认目录不存在。本次为直接删除，没有保留回收副本；这条实验不能再从这些本地checkpoint续跑。旧长跑及其模型、诊断checkpoint、源码、日志、W&B历史和S3报告均未删除或修改。不要自动重提这条实验。

用户要求：仅NVFP4，新增一条修复版长跑，三个job，与原长跑比较。

- EXP / W&B ID：`nvfp4_router_fixed_formal_0912_v1`
- Project：`shawnzzz/slime-deterministic-gb200`
- 对照：`nvfp4_seed_unique_formal_0910_v2`
- 新初始训练，不加载旧run或诊断单步checkpoint。
- 3 × 5小时，4nodes×4GPUs，EP4、GBS256、32×8、Adam lr1e-6、TIS2、CLB1、full recompute。
- 目标1600 rollout沿用原配置；三个job共15小时不保证跑满1600步。
- 每10步保存完整Adam；沿用旧脚本，只保留本新namespace两代已完成checkpoint和进行中的保存。旧实验checkpoint不动。
- 每段复用同一个新W&B run；首段resume=never，后两段must。后两段afterany上一段，允许正常TIMEOUT后续跑；缺少latest/.metadata则失败退出，不偷偷fresh restart。
- 保留旧eval、sampling seed、训练/rollout precision与CI设置。没有DQ backward或four-over-six，没有诊断router monkeypatch。
- 新代码相对旧正式snapshot，唯一Python源文件差异为 `deepgemm_moe_forward.py`；模块hash `5aa7c36969134bdd08fcf95a0bb85d3758d59a639b24bfb0dbbcb670ac754a3c` 与新完整模型回归snapshot相同。
- 执行合同来源：历史2782759/2782760完整SubmitLine，baseline_submit.txt。只改变namespace、router源码、三段链长度，并加强续跑metadata存在检查。

## Gate与提交

- 2803115：双路径各50/50算子边界检查通过。
- 2803118：直接集成router修复的完整4n4g回归；stage/release自动要求COMPLETED/0:0、48norm非零、16rank router hook和完整参数更新、checkpoint metadata。
- 命令：`uv run --no-project --python verl_nvfp4_e2e_r3_20260831_v3/.venv/bin/python python investigation/nvfp4_router_long_20260912_v1/submit_chain.py {dry-run|stage|audit|release}`
- 三段先hold提交，逐一核对资源和实际dependency后release；job IDs与完整SubmitLine写入submission.json。
- 使用W&B技能的独立run原则与metric family step口径；新run名称在线检查无冲突，启动时never另作碰撞保护。

## 对比

2026-09-12 已完成 dry-run → stage → audit → release。三个job：2803142 → 2803143 → 2803144，每段5小时；后两段afterany实际依赖已核实。首段当前排队，尚未进入模型初始化/训练，无本次长跑真实异常。新完整模型gate 2803118 COMPLETED/0:0，5分27秒，loss/entropy/grad norm与此前候选2793007一致，16rank全部48层router完整参数更新通过。

W&B（首次初始化后可见）：https://wandb.ai/shawnzzz/slime-deterministic-gb200/runs/nvfp4_router_fixed_formal_0912_v1

按train/step比较entropy/grad，按rollout/step比较raw_reward与response_lengths，同时检查AIME和train-rollout mismatch。是router修复因子的单条长跑对照，不是三个独立seed，不把前三个job完成等同于统计显著结论。
