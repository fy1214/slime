"""Read completed JSON/log artifacts only; no torch or distributed imports."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
summary = {}
for variant, job in [('original', 2793006), ('fixed', 2793007)]:
    folder = ROOT / 'formal/results' / f'nvfp4_router_{variant}_fixedbatch_0911_v1'
    files = list(folder.glob('router_rank*.json'))
    routers = [r for f in files for r in json.loads(f.read_text())['routers'].values()]
    if not routers:
        summary[variant] = {'job': job, 'rank_files': len(files)}
        continue
    entry = {'job': job, 'rank_files': len(files), 'router_records': len(routers)}
    for kind in ['weight', 'logits', 'probabilities']:
        entry[kind] = {key: sum(r[kind][key] for r in routers) for key in
                       ['requires_grad_true', 'requires_grad_false', 'hook_count', 'hook_nonzero']}
        entry[kind]['records_with_nonzero_hook'] = sum(r[kind]['hook_nonzero'] > 0 for r in routers)
    entry['main_grad_nonzero'] = sum(r['pre_step']['main_grad']['norm'] > 0 for r in routers)
    entry['grad_present'] = sum(r['pre_step']['grad'] is not None for r in routers)
    entry['full_parameter_update_nonzero'] = sum(r['parameter_update']['norm'] > 0 for r in routers)
    entry['update_norm_range'] = [min(r['parameter_update']['norm'] for r in routers),
                                  max(r['parameter_update']['norm'] for r in routers)]
    entry['router_stats_finite'] = all(r['pre_step']['main_grad']['finite'] and r['parameter_update']['finite'] for r in routers)
    for line in (ROOT / 'formal/logs' / f'{job}.err').read_text().splitlines():
        if 'step 0: {' in line:
            entry['metrics'] = ast.literal_eval(line.split('step 0: ', 1)[1])
    audits = [json.loads(f.read_text()) for f in folder.glob('rank*.json')]
    grads = [g for a in audits for g in a['gradients'].values()]
    entry['parameter_audit'] = {'rank_files': len(audits), 'records': len(grads),
                                'missing': sum(g['missing'] for g in grads),
                                'nonfinite': sum(not g['finite'] for g in grads if not g['missing'])}
    summary[variant] = entry
print(json.dumps(summary, indent=2))
