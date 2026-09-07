"""Run paired seeds only after all reference gates; assess history separately from plumbing."""
from pathlib import Path
import argparse,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from temporal.training import load_dataset

def assess(records):
    seeds=sorted({r['seed'] for r in records})
    if len(seeds)!=3 or len(records)!=6:raise ValueError('exactly three paired seeds required')
    grouped={mode:[r for r in records if r['mode']==mode] for mode in ('current','history')}
    if any(sorted(r['seed'] for r in rows)!=seeds for rows in grouped.values()):raise ValueError('unpaired ablation records')
    keys=('position_velocity_error','penetration','degeneration','inference_p95_ms')
    averages={mode:{k:sum(float(r[k]) for r in rows)/3 for k in keys} for mode,rows in grouped.items()}
    a,b=averages['current'],averages['history']
    def no_worse(key):return b[key]<=a[key]*1.05+1e-6
    gain=(a['position_velocity_error']-b['position_velocity_error'])/max(a['position_velocity_error'],1e-12)
    result=dict(averages=averages,relative_error_reduction=gain,
        history_benefit_passed=gain>=.1 and no_worse('penetration') and no_worse('degeneration'),
        performance_qualified=False,scope='Existing garment only; not a multi-asset generalization claim')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path);p.add_argument('--output',type=Path,default=ROOT/'.work/temporal_v4/experiments')
    p.add_argument('--assess',type=Path);p.add_argument('--epochs',type=int,default=30);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.assess:
        result=assess(json.loads(a.assess.read_text()));(a.output/'assessment.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));return
    if not a.dataset:p.error('--dataset or --assess is required')
    try:list(load_dataset(a.dataset))
    except (ValueError,KeyError,FileNotFoundError) as e:
        result=dict(quality_training_started=False,history_benefit_passed=False,blocked_reason=str(e),reference_gate_required=True)
        (a.output/'assessment.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));return
    for seed in (17,29,43):
        for mode in ('current','history'):
            folder=a.output/f'{mode}-{seed}';args=[sys.executable,str(ROOT/'tools/train_temporal.py'),'--dataset',str(a.dataset),'--output',str(folder),'--seed',str(seed),'--epochs',str(a.epochs)]
            if mode=='current':args+=['--current-only']
            if (folder/'latest.pt').exists():args+=['--resume']
            subprocess.run(args,check=True)
    print('Training complete. Held-out full-rollout evaluation is required before history benefit can be accepted.')
if __name__=='__main__':main()
