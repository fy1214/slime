# PR delivery validation (2026-09-13)

- Production fix commit: `bde4768` (44 insertions, original forward retained).
- New Slurm job **2819444: COMPLETED/0:0**, actual PR checkout NVFP4 gate **50/50**, staged FP8 snapshot **50/50**. Same recorded GB200/v6 image, one process at a time, GPU0, CPU/BLAS threads=1. No GPU or torch test on login node.
- Earlier delivery check 2819442 failed before test startup because staging suffix rewriting matched the `0912_v1` substring inside a YYYYMMDD directory. Corrected with a negative-digit-boundary regex; fresh v3 staging and the above job pass. This was a packaging/entrypoint failure, not a router numerical regression.
- Standard-library AST parse: 30 packaged Python files; `bash -n`: 23 shell/job scripts pass.
- Fresh staging successfully reconstructs the historical baseline and applies the fixed VJP to both arms. Four complete-model driver dry-runs (router original/fixed, directly integrated original/dequantized) and the three-segment long-run dry-run pass. All three long-run SAVE paths point under the new staging root, not the original experiment.
- No new full-model/reward run was submitted during PR delivery. Historical full-model evidence is 2793006/2793007 and 2803118, documented in README. The user's canceled long run remains canceled and its model remains deleted.
- No model/data/batch.pt/logs/credentials/signed S3 URLs in the package. Historical `.patch` files preserve diff context and original whitespace; they are provenance artifacts, not production modules to import.

The new test used the packaged `historical/.../nvfp4_router_fix_20260912_v1/test_router.py`, with its source argument pointing at the actual PR checkout for NVFP4 and the freshly staged fixed snapshot for FP8. The test source imports the requested checkout before importing Slime.
