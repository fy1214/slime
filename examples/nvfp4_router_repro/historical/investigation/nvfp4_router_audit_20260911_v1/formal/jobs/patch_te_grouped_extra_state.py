#!/usr/bin/env python3
"""Let TEGroupedLinear save a dist checkpoint when the FP8 recipe has no global meta.

Why
---
`--fp8-recipe blockwise` (Float8BlockScaling) and MXFP8 carry no persistent global FP8
meta -- no amax history, no global scales -- so TE's `get_extra_state()` returns an EMPTY
tensor.  Megatron's `_decode_extra_state` turns that into `None`:

    if state.numel() == 0:      # "No FP8 is indicated by an empty tensor"
        return

but `_split_extra_state` (the SAVE path) then indexes it unconditionally:

    state = self._decode_extra_state(state)
    extra_fp8_variables = state["extra_fp8_variables"]
    TypeError: 'NoneType' object is not subscriptable
      megatron/core/extensions/transformer_engine.py:1775
      <- _sharded_state_dict_grouped:1820 <- sharded_state_dict:1921

The LOAD path in the same file already guards for exactly this
(`... or self._decode_extra_state(state_dict[f"{prefix}_extra_state"]) is None: return`,
line ~1647) -- only the save path is missing it.  So any FP8-blockwise run with grouped
MoE experts dies at its first checkpoint (job 2765230: 10 steps, then SAVE_INTERVAL=10).

The deterministic arms are unaffected: they never pass `--fp8-format`, so
`fp8_checkpoint` is False and the function returns on its first branch.

Fix: when the decoded state is None there is nothing to split -- hand every GEMM the same
raw (empty) extra state, exactly as the `not fp8_checkpoint` branch two lines above does.

Idempotent.  Disable with SLIME_TE_EXTRA_STATE_FIX=0.
"""

import pathlib
import sys

MARKER = "NVIDIA FIX: recipes without global FP8 meta"

OLD = """            if not fp8_checkpoint:
                return [state] * self.num_gemms

            state = self._decode_extra_state(state)
            extra_states = []"""

NEW = """            if not fp8_checkpoint:
                return [state] * self.num_gemms

            decoded_state = self._decode_extra_state(state)
            if decoded_state is None and os.environ.get("SLIME_TE_EXTRA_STATE_FIX", "1") == "1":
                # NVIDIA FIX: recipes without global FP8 meta (Float8BlockScaling /
                # MXFP8) produce an empty _extra_state, which _decode_extra_state
                # returns as None.  There is nothing to split; give every GEMM the
                # same raw state, as the not-fp8_checkpoint branch above does.  The
                # load path already guards for this case (see _merge_extra_state).
                return [state] * self.num_gemms
            state = decoded_state
            extra_states = []"""


def main() -> int:
    path = pathlib.Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "/root/Megatron-LM/megatron/core/extensions/transformer_engine.py"
    )
    if not path.exists():
        print(f"TE_EXTRA_STATE_FIX: {path} not found", file=sys.stderr)
        return 1
    src = path.read_text()
    if MARKER in src:
        print("TE_EXTRA_STATE_FIX: already patched")
        return 0
    if OLD not in src:
        print("TE_EXTRA_STATE_FIX: FAILED - _split_extra_state not in the expected shape",
              file=sys.stderr)
        return 2
    if "\nimport os\n" not in src:
        src = src.replace("\nimport io\n", "\nimport io\nimport os\n", 1)
        if "\nimport os\n" not in src:
            src = "import os\n" + src
    src = src.replace(OLD, NEW, 1)

    import ast
    ast.parse(src)
    path.write_text(src)
    print("TE_EXTRA_STATE_FIX: patched (_split_extra_state tolerates an empty _extra_state)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
