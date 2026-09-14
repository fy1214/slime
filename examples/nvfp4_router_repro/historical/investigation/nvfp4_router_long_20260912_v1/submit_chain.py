"""Three held Slurm segments derived from the exact old SubmitLine.

Only stdlib and scheduler commands on login node; no training imports.
"""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
EXP = 'nvfp4_router_fixed_formal_0912_v1'
BASE_EXP = 'nvfp4_seed_unique_formal_0910_v2'
OLD = str(ROOT.parent / 'determinism_initial_gap_20260909/seed_formal_v2/nvfp4')
RECEIPT = ROOT / 'submission.json'
BASE = shlex.split((ROOT / 'baseline_submit.txt').read_text().strip())
EXPORT = next(x for x in BASE if x.startswith('--export='))
ENV = dict(x.split('=', 1) for x in EXPORT.removeprefix('--export=ALL,').split(','))
ENV = {k: v.replace(OLD, str(ROOT)).replace(BASE_EXP, EXP) for k, v in ENV.items()}


def command(segment, dependency=None):
    env = dict(ENV, RESUME='0' if segment == 1 else '1',
               WANDB_RESUME_POLICY='never' if segment == 1 else 'must')
    cmd = [x for x in BASE if not x.startswith(('--export=', '--job-name='))]
    cmd[-2] = str(ROOT / 'jobs/run_nvfp4.job')
    opts = ['--hold', f'--job-name=general_sa-infix:nvfp4-router-0912-v1-s{segment:02}',
            '--export=ALL,' + ','.join(f'{k}={v}' for k, v in env.items())]
    if dependency:
        opts.append(f'--dependency=afterany:{dependency}')
    return cmd[:1] + opts + cmd[1:]


def verify_gate():
    result = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', '2803118',
                                     '--format=State,ExitCode'], text=True).strip()
    assert result.startswith('COMPLETED|0:0'), result
    base = ROOT.parent / 'nvfp4_router_fix_20260912_v1/formal'
    folder = base / 'results/nvfp4_routerfixed_dq_original_0912_v1'
    gate = json.loads((folder / 'first_step_gate.json').read_text())
    assert gate['passed'] and gate['norm_nonzero_gradients'] == 48
    routers = list(folder.glob('router_rank*.json'))
    assert len(routers) == 16
    for f in routers:
        rows = json.loads(f.read_text())['routers'].values()
        assert len(rows) == 48
        assert all(r['weight']['hook_count'] > 0 and r['parameter_update']['norm'] > 0
                   and r['parameter_update']['finite'] for r in rows)
    assert (base / 'ckpts/nvfp4_routerfixed_dq_original_0912_v1/iter_0000000/.metadata').is_file()


action = sys.argv[1] if len(sys.argv) > 1 else 'dry-run'
if action == 'dry-run':
    for s in range(1, 4):
        print(shlex.join(command(s, None if s == 1 else f'<segment{s-1}>')))
    env = dict(os.environ, **ENV, DRYRUN='1', MODE='nvfp4', SLIME_ROOT=str(ROOT / 'slime_nvfp4'))
    result = subprocess.run(['bash', str(ROOT / 'jobs/nvfp4_driver.sh')], env=env,
                            capture_output=True, text=True, check=True)
    (ROOT / 'driver_audit.txt').write_text(result.stdout + result.stderr)
elif action == 'stage':
    assert not RECEIPT.exists(), 'Submission already exists'
    for name in ['ckpts', 'results', 'wandb']:
        assert not (ROOT / name / EXP).exists(), f'{name} collision'
    jobs = subprocess.check_output(['squeue', '-u', 'shuazhang', '-h', '-o', '%j'], text=True)
    assert 'nvfp4-router-0912-v1' not in jobs, 'Job name collision'
    verify_gate()
    receipt = []
    for s in range(1, 4):
        cmd = command(s, receipt[-1]['job'] if receipt else None)
        job = subprocess.check_output(cmd, text=True).strip().split(';')[0]
        assert job.isdigit(), job
        receipt.append(dict(segment=s, job=job, command=shlex.join(cmd)))
        RECEIPT.write_text(json.dumps(receipt, indent=2))
        print('STAGED', s, job, flush=True)
elif action in ('audit', 'release'):
    receipt = json.loads(RECEIPT.read_text())
    assert len(receipt) == 3
    for i, row in enumerate(receipt):
        info = subprocess.check_output(['scontrol', 'show', 'job', row['job']], text=True)
        assert 'NumNodes=4' in info and 'TimeLimit=05:00:00' in info
        if i:
            assert f'Dependency=afterany:{receipt[i-1]["job"]}' in info
        else:
            assert 'Dependency=(null)' in info
        print(info)
    if action == 'release':
        verify_gate()
        for row in reversed(receipt):
            subprocess.run(['scontrol', 'release', row['job']], check=True)
else:
    raise ValueError(action)
