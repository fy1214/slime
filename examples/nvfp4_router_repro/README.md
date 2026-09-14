# Router 修复：最小复现

生产改动只有 deepgemm_moe_forward.py 中44行：保留 SGLang top-k 前向，补选中专家概率的VJP。索引不求导；支持空token、零scale及归一化top-1，只支持一阶反向。

## 回归测试

test_router.py 是已验证的50组测试，覆盖tokens=0/1/17/128/513、scale=None/0/.5/1/2、topk=1/8，检查前向逐位一致、no-grad推理、非选中项零梯度和参考VJP。

使用已验证的Slime镜像和环境脚本，设置 ROUTER_CHECKOUT、ROUTER_IMAGE、ROUTER_MOUNTS、ROUTER_PREP、ROUTER_RESULT 后提交 gate.job。需自行指定集群account/partition/GPU资源；不要在登录节点导入torch或运行测试。原验证镜像是 slime-nvfp4-det-gb200-20260905-v6.sqsh。

## 完整模型及三个长跑job

复用自己的已验证NVFP4训练入口，只替换源码到本PR。镜像、模型、数据、seed、训练合同保持不变；不要把DQ或four-over-six混入router单变量对照。入口必须确实加载本PR，不能只设置一个被原脚本忽略的SLIME_ROOT。

submit_three.py 仅编排三个已审核的sbatch命令，**不是完整训练launcher**。输入JSON为三个argv数组，必须含sbatch、--parsable、--hold；后两段含 --dependency=afterany:{previous}。默认仅打印命令，添加 --submit --receipt 新文件路径才提交。三个job保持hold，核实资源、checkpoint合同与实际依赖后手动release，不会自动启动。

首段使用新EXP/W&B/checkpoint namespace和初始模型；后两段恢复完整Adam，缺少有效checkpoint必须退出，不能悄悄fresh-start。历史合同为4nodes×4GPUs、EP4、GBS256、32×8、Adam lr1e-6、TIS2、CLB1、full recompute，3×5小时，每10步保存。fresh-start seed前提见PR #12，不属于router修复。

历史完整模型和三段job的详细脚本仍保留在作者共享workspace的 investigation/nvfp4_router_fix_20260912_v1/formal/ 和 investigation/nvfp4_router_long_20260912_v1/，此次不再复制进修复PR。独立复现需要被授权的模型/数据/batch和对应环境；本目录不是自包含训练环境。旧长跑已取消且模型已删除，不应原样执行旧job ID/namespace。

## 验证边界

2819444：实际PR NVFP4源码与当时FP8快照各50/50通过。2793006/2793007和2803118：完整模型从0/48 router更新恢复至48/48，前向loss/entropy相同。没有证明长期reward提升，也不解释首次更新前高熵。

本次精简未改变生产修复与50组测试；新的通用gate包装和三段提交助手仅经静态检查，不声称以它们重跑过完整训练。
