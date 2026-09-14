#!/usr/bin/env python3
"""Run Megatron's parallel checkpoint writers IN-PROCESS instead of forking.

Why
---
On GB200/aarch64 every checkpoint save segfaults inside torch's zip serializer:

    crc32_16bytes -> mz_crc32 -> mz_zip_writer_add_mem_ex_v2
      -> caffe2::serialize::PyTorchStreamWriter::writeRecord

`FileSystemWriterAsync.write_preloaded_data_multiproc` forks one child per write bucket
while CUDA and several threads are live, so the child inherits an inconsistent heap
(Megatron-LM #1861).  Measured here:

    fork  + non_blocking=True   -> segfault      (jobs 2749609/2749610)
    fork  + non_blocking=False  -> segfault      (jobs 2749750/2749751)
    spawn + non_blocking=False  -> no segfault, but spawn must pickle the payload and
                                   Megatron's write bucket is not picklable:
                                   "ForkingPickler.dump ... IndexError: tuple index out of
                                   range"                       (jobs 2749702/2749703)

So the fork is the problem and spawn is not usable as-is.  This patch removes the
subprocess entirely: each bucket is written inline by the same process.  The queues,
the completion counting and the result collection are untouched, so the surrounding
logic (including `count_queue.join()`) behaves exactly as before -- `task_done()` has
already been called by the time `join()` runs.

Cost: the per-bucket writes serialise instead of running in parallel.  Measured save
time is ~27 s parallel; expect roughly N_buckets x single-bucket time.  At
SAVE_INTERVAL=10 that is a rounding error next to a ~5 min step.

Idempotent.  Disable with SLIME_CKPT_INPROC=0.
"""

import pathlib
import sys

MARKER = "_InProcWriterProc"

SHIM = '''

class _InProcWriterProc:
    """Process-compatible shim that runs the writer inline (NVIDIA, Megatron-LM #1861).

    Forking here segfaults in torch's zip serializer on aarch64; spawn cannot pickle the
    write bucket.  Running in-process removes both failure modes.
    """

    def __init__(self, target=None, kwargs=None, **_ignored):
        self._target = target
        self._kwargs = kwargs or {}
        self.exitcode = 0

    def start(self):
        self._target(**self._kwargs)

    def join(self, *args, **kwargs):
        return None

    def is_alive(self):
        return False

    def terminate(self):
        return None

'''

OLD_CALL = """                p_list.append(
                    ctx.Process(
                        target=partial(FileSystemWriterAsync.write_preloaded_data, transform_list),
                        kwargs=kwargs,
                    )
                )"""

NEW_CALL = """                _proc_cls = (
                    _InProcWriterProc
                    if os.environ.get("SLIME_CKPT_INPROC", "1") == "1"
                    else ctx.Process
                )
                p_list.append(
                    _proc_cls(
                        target=partial(FileSystemWriterAsync.write_preloaded_data, transform_list),
                        kwargs=kwargs,
                    )
                )"""


def main() -> int:
    path = pathlib.Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "/root/Megatron-LM/megatron/core/dist_checkpointing/strategies/filesystem_async.py"
    )
    if not path.exists():
        print(f"CKPT_INPROC: {path} not found", file=sys.stderr)
        return 1
    src = path.read_text()
    if MARKER in src:
        print("CKPT_INPROC: already patched")
        return 0
    if OLD_CALL not in src:
        print("CKPT_INPROC: FAILED - ctx.Process call site not found (Megatron changed?)", file=sys.stderr)
        return 2

    if "\nimport os\n" not in src:
        src = src.replace("\nimport logging\n", "\nimport logging\nimport os\n", 1)
        if "\nimport os\n" not in src:
            src = "import os\n" + src

    # insert the shim just before the class that uses it
    anchor = "\nclass FileSystemWriterAsync("
    if anchor in src:
        src = src.replace(anchor, SHIM + anchor, 1)
    else:
        src = src + SHIM

    src = src.replace(OLD_CALL, NEW_CALL, 1)
    path.write_text(src)

    import ast

    ast.parse(src)
    print("CKPT_INPROC: patched (writers now run in-process; SLIME_CKPT_INPROC=0 restores fork)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
