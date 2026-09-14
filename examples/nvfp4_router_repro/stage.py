"""Materialize isolated historical experiments; NEVER submits a job.

Standard-library-only staging is safe on a login node. GPU scripts are not.
"""
import argparse
import getpass
from pathlib import Path
import re
import shutil
import subprocess

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
OLD_SHARED = '/lustre/fsw/general_sa/shuazhang'
OLD_WORKSPACE = OLD_SHARED + '/python_space/verl_for_nvfp4_20251031'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True, help='NEW shared-FS output directory, outside checkout')
    p.add_argument('--tag', required=True, help='Unique W&B/experiment suffix')
    p.add_argument('--shared-prefix', default=OLD_SHARED)
    p.add_argument('--netrc', default=str(Path.home() / '.netrc'))
    p.add_argument('--fixed-batch', type=Path, help='Existing authorized batch.pt; never uploaded by this tool')
    args = p.parse_args()
    root = args.root.resolve()
    for value in [str(root), args.tag, args.shared_prefix, args.netrc]:
        if not re.fullmatch(r'[A-Za-z0-9_./-]+', value):
            p.error('Paths/tag must be shell-safe (letters, digits, underscore, slash, dot, hyphen)')
    if root.exists() or root == REPO or REPO in root.parents:
        p.error('--root must be nonexistent and outside the checkout')
    if args.fixed_batch and not args.fixed_batch.is_file():
        p.error('--fixed-batch does not exist')
    shutil.copytree(PACKAGE / 'historical', root)
    baseline = root / 'investigation/determinism_initial_gap_20260909/seed_formal_v2/nvfp4/slime_nvfp4'

    def ignore(folder, names):
        excluded = {'.git', '.venv', '__pycache__', '.pytest_cache'}
        if Path(folder) == REPO / 'examples':
            excluded.add('nvfp4_router_repro')
        return set(names) & excluded

    shutil.copytree(REPO, baseline, ignore=ignore)
    subprocess.run(['patch', '--batch', '-p1', '-d', str(baseline), '-i', str(PACKAGE / 'baseline_nvfp4.patch')], check=True)

    def snapshot(target, arm='nvfp4', fixed=False):
        # Existing target contains only the historical diagnostic plugins.
        plugins = target / 'slime_plugins'
        saved = {f.name: f.read_bytes() for f in plugins.glob('*.py')} if plugins.exists() else {}
        if arm == 'fp8':
            shutil.copytree(REPO, target, ignore=ignore, dirs_exist_ok=True)
            subprocess.run(['patch', '--batch', '-p1', '-d', str(target), '-i', str(PACKAGE / 'baseline_fp8.patch')], check=True)
        else:
            shutil.copytree(baseline, target, dirs_exist_ok=True)
        for name, data in saved.items():
            (plugins / name).write_bytes(data)
        if fixed:
            subprocess.run(['patch', '--batch', '-p1', '-d', str(target), '-i', str(PACKAGE / 'router_fix.patch')], check=True)

    for exp in ['nvfp4_dq_backward_20260911_v1', 'nvfp4_router_audit_20260911_v1']:
        snapshot(root / 'investigation' / exp / 'formal/slime_nvfp4')
    fix = root / 'investigation/nvfp4_router_fix_20260912_v1'
    snapshot(fix / 'code/slime_nvfp4', fixed=True)
    snapshot(fix / 'code/slime_det', arm='fp8', fixed=True)
    snapshot(fix / 'formal/slime_nvfp4', fixed=True)
    snapshot(root / 'investigation/nvfp4_router_long_20260912_v1/slime_nvfp4', fixed=True)

    for f in root.rglob('*'):
        if not f.is_file() or f.suffix not in {'.py', '.sh', '.job', '.md', '.txt', '.json'}:
            continue
        text = f.read_text()
        text = text.replace(OLD_WORKSPACE, str(root)).replace(OLD_SHARED, args.shared_prefix)
        text = text.replace('/home/shuazhang/.netrc', args.netrc)
        text = text.replace('-u shuazhang', '-u ' + getpass.getuser()).replace("'shuazhang'", repr(getpass.getuser()))
        # Names contain short dates; directory names contain YYYYMMDD and stay stable.
        for date in ['0910_v2', '0911_v1', '0912_v1']:
            text = re.sub(r'(?<![0-9])' + date, date + '_' + args.tag, text)
        for date in ['0911-v1', '0912-v1']:
            text = text.replace(date, date + '-' + args.tag)
        if f.name == 'submit_chain.py':
            # Never use the author's completed job as the new release gate.
            text = text.replace("'2803118'", "os.environ['ROUTER_REPRO_GATE_JOB']")
        if f.suffix == '.job':
            # Slurm does not create the parent of its output path.
            for line in text.splitlines():
                if line.startswith(('#SBATCH --output=', '#SBATCH --error=')):
                    Path(line.split('=', 1)[1]).parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    if args.fixed_batch:
        batch = root / 'investigation/determinism_backward_audit_20260909/full_model_fixed_v1/batch.pt'
        batch.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.fixed_batch, batch)
    print('STAGED (no jobs submitted):', root)
    print('Review image/model/data/mount/account/partition paths before sbatch.')
    if not args.fixed_batch:
        print('Fixed-batch model probes unavailable until an authorized batch.pt is provided.')


if __name__ == '__main__':
    main()
