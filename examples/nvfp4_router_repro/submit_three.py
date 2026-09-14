"""Submit three reviewed sbatch commands; no recipe defaults or historical IDs.

Input JSON: three argv arrays beginning with sbatch --parsable --hold.
Continuation commands must contain --dependency=afterany:{previous}.
Dry-run by default. Submitted jobs remain held for manual audit/release.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('plan', type=Path)
p.add_argument('--submit', action='store_true')
p.add_argument('--receipt', type=Path)
args = p.parse_args()
plan = json.loads(args.plan.read_text())
assert len(plan) == 3
for i, cmd in enumerate(plan):
    assert isinstance(cmd, list) and all(isinstance(x, str) for x in cmd)
    assert cmd[0] == 'sbatch' and '--parsable' in cmd and '--hold' in cmd
    deps = [x for x in cmd if x.startswith('--dependency=')]
    assert deps == ([] if i == 0 else ['--dependency=afterany:{previous}'])
if not args.submit:
    for cmd in plan:
        print(shlex.join(cmd))
else:
    if args.receipt is None:
        p.error('--submit requires a new --receipt path')
    with args.receipt.open('x') as receipt:
        previous = ''
        for cmd in plan:
            actual = [x.replace('{previous}', previous) for x in cmd]
            job = subprocess.check_output(actual, text=True).strip().split(';')[0]
            if not job.isdigit():
                raise RuntimeError(f'Unexpected sbatch output: {job!r}')
            receipt.write(json.dumps({'job': job, 'command': actual}) + '\n')
            receipt.flush()
            print('HELD', job, flush=True)
            previous = job
