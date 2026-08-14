
[run-qwen3-30B-A3B-bf16.sh](run-qwen3-30B-A3B-bf16.sh)：
使用原始的TE的GroupLinear，没有任何其他代码的sglang做rollout，做bf16训推的RL训练

[run-qwen3-30B-A3B-nvfp4.sh](run-qwen3-30B-A3B-nvfp4.sh)：
使用原始的TE的GroupLinear，sglang使用pertoken路径做的QAT rollout，做nvfp4训，bf16(nvfp4 fake quant)推的RL训练

[run-qwen3-30B-A3B-nvfp4_QAT_8.sh](run-qwen3-30B-A3B-nvfp4_QAT_8.sh):
使用miniTE的GroupLinear，op为QAT=8, sglang使用pertoken路径做的QAT rollout，做nvfp4训，bf16(nvfp4 fake quant)推的RL训练

[run-qwen3-30B-A3B-nvfp4_QAT_8_modelopt_cutlass.sh](run-qwen3-30B-A3B-nvfp4_QAT_8_modelopt_cutlass.sh)
使用miniTE的GroupLinear，op为QAT=8, sglang使用modelopt cutlass pertoken路径做的rollout，做nvfp4训，real fp4推的RL训练

[run-qwen3-30B-A3B-nvfp4_QAT_8_modelopt_trt_pertoken.sh](run-qwen3-30B-A3B-nvfp4_QAT_8_modelopt_trt_pertoken.sh)
使用miniTE的GroupLinear，op为QAT=8, sglang使用modelopt flashinfer_trt pertoken路径做的rollout，做nvfp4训，real fp4推的RL训练

- `SHOULD_REPLACE_TE_GROUPLINEAR=1` — 开启后，用miniTransformer的GroupLinear替换TE的GroupLinear
- `SGLANG_NVFP4_PERTOKEN_SCALE=1` — 用来控制所有sglang上我们新加的所有nvfp4 pertoken逻辑，关闭即没有