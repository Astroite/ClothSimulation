"""Verify the exact teacher-copy solution of initialization distillation on real inputs.

This is interface validation; diagnostic trajectories are not quality targets.
No optimization is needed when the student is identical with a zero final branch.
"""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from temporal.training import load_sequence,make_graph,push_history,digest
from temporal.model import TemporalTinyHood,GnnHistoryState
from temporal.format import export_model
from real_scene.fine15 import Fine15,Fine15Weights
from real_scene.tinyhood import load_tinyhood

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sequence',type=Path,required=True)
    p.add_argument('--legacy',type=Path,default=ROOT.parent/'vulkan-gnn-poc/.work/hood_data/student32x12_r1.vhood')
    p.add_argument('--output',type=Path,default=ROOT/'.work/temporal_v4/initialization-report.json');a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(17);s=load_sequence(dict(resolved_path=str(a.sequence)),'cpu')
    old=load_tinyhood(a.legacy).eval();weights=Fine15Weights.from_vhood(a.legacy);builder=Fine15(weights);rows=[]
    for current_only in (False,True):
        model=TemporalTinyHood.from_legacy(a.legacy,current_only).eval();history=GnnHistoryState();maximum=0.
        with torch.inference_mode():
            for t in range(min(8,len(s['x'])-1)):
                graph=make_graph(builder,s,s['x'][t],s['v'][t],t);h=push_history(history,builder,s,graph,s['v'][t],t)
                teacher=old(graph);student=model(graph,h);torch.testing.assert_close(student,teacher,atol=1e-5,rtol=1e-4)
                maximum=max(maximum,float((teacher-student).abs().max()))
        rows.append(dict(current_only=current_only,maximum_raw_decoder_error=maximum,frames=t+1))
    report=dict(passed=True,method='exact teacher-copy initialization: zero distillation loss before any update',quality_trained=False,
        legacy_sha256=digest(a.legacy),input_sha256=digest(a.sequence),cases=rows)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
