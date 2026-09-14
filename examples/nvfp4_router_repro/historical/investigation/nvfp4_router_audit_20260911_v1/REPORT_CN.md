# NVFP4 初始熵、梯度与 reward 差距：综合诊断

## 当前结论

不能把当前差距认定为 NVFP4 的固有精度上限。已在实际 GB200 镜像复现一个独立于已修 norm/recompute 问题的 router top-k 反向缺陷；它同时存在于当前 FP8 det 和 NVFP4 冻结代码。另有确定的量化前向扰动和 BF16 surrogate backward 偏差，但它们对长期 reward 的因果贡献尚未分离。

原4n×4g的router原版/候选对照已完成：原版48层router没有反向hook、没有参数更新；候选恢复全部48层router更新，报告的前向指标完全相同。完整分布式梯度连接修复已验证，但reward影响尚未证明。正式训练链未改动。

## 新定位的 router 反向缺陷

当前 `deepgemm_moe_forward.py::_sglang_unbiased_softmax_topk_routing` 调用推理用 `sglang.jit_kernel.moe_fused_gate`，将其输出 `topk_weights` 写入 dense routing_probs。实际 runtime 源码先 `torch.empty` 分配输出，再由 Triton 写入，没有自定义 autograd wrapper；导出代码见 `moe_fused_gate_runtime.py`。

Job `2792992`（COMPLETED/0:0，39秒）在正式 NVFP4 v6镜像直接调用当前冻结函数：即使输入logits.requires_grad=True，原版routing_probs.requires_grad=False、grad_fn=None。测试覆盖token数1/17/128/513与scale None/1/.5/2，共16组。候选给同一前向补充选定top-k后的归一化softmax解析反向，16组前向逐位相同，与原生PyTorch softmax的梯度误差约1e-7；输出 ROUTER_BACKWARD_GATE_PASS，结果见 gate.json。

对选定专家集，若输出 p=s·softmax(z_selected)，反向为
`dL/dz = p * (dL/dp - sum(p*dL/dp)/s)`；未选中位置 p=0。top-k索引不求导；这不等于可以把选中专家的概率也detach。当前配置moe_aux_loss_coeff=0，没有辅助loss提供另一条router梯度路径。

当前FP8 det的同名函数也采用同一无backward调用。由此不能声称此bug只影响NVFP4，或它解释全部NVFP4与FP8/BF16的差距。即使router权重不更新，上游hidden与专家参数仍可学习，routing选择也可随hidden变化；不能把它描述为整模型不学习或专家分配完全冻结。

## 原拓扑验证合同

- 单进程gate入口 `probe.job`；使用历史成功2792416同样的image/source/prep。没有在登录节点导入torch或运行GPU。
- 新完整模型snapshot `formal/slime_nvfp4`，从上一轮成功DQ对照snapshot派生；新增router候选和显式hook/buffer/update审计。
- 原版job `2793006`，修复候选job `2793007`，后者afterok前者，已核实scheduler dependency。两者time limit均由60min调整为20min，保留原4nodes×4GPUs、EP4、GBS256、32×8、full recompute、原模型/checkpoint、CLB1、TIS2、Adam lr1e-6；只做一个固定batch optimizer step。
- 两组均使用原始BF16 expert backward（DQ_VARIANT=original），不混入dequantized或four-over-six。
- driver参数分别在formal/original_driver_audit.txt、formal/fixed_driver_audit.txt；提交入口formal/submit.sh；镜像为slime-nvfp4-det-gb200-20260905-v6.sqsh，mount和prep沿用前次成功配对。
- 增加router weight、logits、probabilities的requires_grad和autograd hook次数/非零梯度，并同时检查main_grad、.grad和完整router参数单步更新；避免仅凭某个buffer为0就诊断断路。
- 保留原norm/recompute gate、全16rank审计和full-Adam最终保存；关闭W&B CLI、eval，归档FP8 rollout导致NVFP4 train/rollout mismatch非零，故两组均关闭该CI断言。
- 尚未提交新reward训练；实验checkpoint不用于替换正式训练chain。

## 前几轮已经建立的证据

1. 修复后的NVFP4前10步grad norm均值约1.294，BF16约.624；但100–149步NVFP4约.162、BF16约.210。应描述为早期尖峰，不能说梯度持续偏高。
2. step0高熵发生在optimizer更新之前，不能由当次backward缺陷直接导致；各run entropy还在各自采样的response上计算，混有轨迹差异。entropy_coef=0，不是entropy loss在主动鼓励高熵。
3. 独立单GPU完整模型参考、同一两个序列/665个response预测位置：BF16 entropy=.178411；仅权重量化=.183628；独立W4A4=.194385；production kernel=.185956。支持前向量化可产生熵差，不代表该比例能外推到全数据或正式训练栈。
4. 同packed操作数逐段对照：24个weight dequant与TE自带dequant最大relative L2约7.1e-8；12次GEMM相对FP32参考误差约.32%–.34%；6次SwiGLU完全一致。未发现明显scale/layout公式错误，但非bitwise全证明，未排除极端输入/分布式问题。
5. 当前TE 2.16.1+c9877beb有backward_override=dequantized，当前custom MoE不走该TE路径；未找到当前安装版本的4over6实现入口。新版文档功能不等于旧镜像已具备。
6. 实际forward操作数量化/反量化BF16 backward候选：6组前向逐位一致，84项autograd对照最大误差3.02e-5。完整4n×4g两组都完成、norm/recompute gate通过，entropy/loss相同，grad norm .329986→.269381（-18.37%）。仅说明梯度改变，不能证明方向更正确或reward更高。候选额外保存中间值，显存明显增加，尚不宜直接长跑。
7. 上一完整模型审计中两组router全部768条main_grad记录与参数更新均为0；本次原拓扑hook审计已经闭合断路证据链，见下节。

原始数据见相邻目录 nvfp4_numeric_diagnostic_20260910_v1、v2，以及 nvfp4_dq_backward_20260911_v1 的STATUS_CN.md和results JSON。W&B窗口数据来自此前在线读取，不声称本报告刷新了当前run历史。

## 新完成的完整模型 router 对照

Job 2793006 原版 COMPLETED/0:0，5分23秒；2793007 候选 COMPLETED/0:0，5分28秒。两组均到达模型初始化、首个训练/optimizer step和full-Adam checkpoint保存，没有观察到真实运行异常。完整driver见上述audit文件；本次唯一训练语义变量是router解析backward，原BF16 expert backward保持不变。未独立reload checkpoint，不声称恢复验证已完成。

| 指标 | 原版 | router候选 |
|---|---:|---:|
| router weight反向hook记录（16rank×48层） | 0/768 | 768/768 |
| router非零本地梯度记录 | 0/768 | 720/768 |
| 完整router参数非零更新记录 | 0/768 | 768/768 |
| entropy | 0.7751415967941284 | 0.7751415967941284 |
| loss | -0.0002020277315750718 | -0.0002020277315750718 |
| train-rollout abs logprob diff | 0.07632403820753098 | 0.07632403820753098 |
| grad norm | 0.32998561126955706 | 0.6732213038389129 |

768条是分布式副本记录，不是768个独立层。候选rank15的48层本地router梯度仍为0，但hook均触发，所有rank的48层完整router参数都产生相同的逐层更新范数（范围6.80e-5至1.74e-4），证明经过分布式聚合和optimizer后更新已恢复。rank15两组的本地input norm/QKV buffer也为0；本次没有审计其样本advantage/mask，故不擅自指定本地零值原因，也不声称所有rank本地梯度均非零。

两组各3840条所选参数梯度记录均无缺失/非有限值，norm/recompute gate均通过（48/48 norm非零，重算输出采样一致），full-Adam `.metadata` 与latest=0存在。gate文件中的variant=fixed指原先norm/recompute修复，不是router分组，router分组应读router_rank*.json。

单步loss/entropy等前向指标完全一致，加上算子16组bitwise前向门槛，支持候选只补反向、不改变原先路由前向。这里没有逐元素保存完整模型全部logits做bitwise证明。总梯度增加约104%，不是恶化判据：原版遗漏了router与通过router回到hidden的梯度分支；同样，DQ把范数降低18%也不等于质量改善。DQ结果来自router尚未修复的基线，不能直接外推到修复后的组合。

机器可读汇总：`full_model_summary.json`；可复算脚本：`summarize.py`（仅stdlib读取日志/JSON，无torch）。完整hook与更新结果在`formal/results/`。

## 整体归因与边界

- **初始高熵**：至少存在量化前向扰动，正式曲线还混有不同采样轨迹；本次router backward修复不改变初始前向，不可能直接消除step0熵差。尚不能判定这个幅度是NVFP4不可避免的下限。
- **早期梯度偏高**：量化前向改变loss局部几何，当前BF16重算backward又与实际量化前向不一致；实验证明backward选择显著改变梯度，但没有确定各因素的因果份额。缺失router分支本身反而压低本批次总梯度，不能用它单独解释高梯度。
- **reward持续落后**：确定存在实现缺陷，且现有FP8/NVFP4对比都受共同router缺陷污染；仍有真正W4A4前向误差和surrogate选择的影响。未跑修复后的多步/多seed reward对照，不能承诺修完追平BF16，也不能把剩余差距先归咎于4bit物理上限。
- **TE功能**：用户指出的dequantized方向成立，但当前custom MoE绕过TE recipe，不能只开recipe开关。当前旧镜像未找到4over6入口；新版[官方recipe文档](https://nvidia.github.io/TransformerEngine/api/common.html)提供该功能，不表示旧版已经支持或生产路径已经启用。

## 判断顺序

应先处理已证实的梯度连通性问题，再评价反量化backward；four-over-six作为独立前向变量最后对照。必须用固定forward对照区分梯度问题，用固定tokens/原始权重区分前向和采样轨迹，用更新后的新rollout/多seed验证reward。梯度范数更小、train-rollout diff更低、单步entropy更低，都不能单独作为训练质量改善的证据。
