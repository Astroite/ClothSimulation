"""Initialize/export a v4 model without modifying the legacy weights."""
from pathlib import Path
import argparse,sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from temporal.model import TemporalTinyHood
from temporal.format import export_model,load_model
from real_scene.fine15 import Fine15Weights
import torch

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy',type=Path,default=ROOT.parent/'vulkan-gnn-poc/.work/hood_data/student32x12_r1.vhood')
    p.add_argument('--output',type=Path,default=ROOT/'.work/temporal_v4/temporal-init.vthood')
    p.add_argument('--checkpoint',type=Path);p.add_argument('--current-only',action='store_true')
    a=p.parse_args();torch.set_num_threads(4);model=TemporalTinyHood.from_legacy(a.legacy,a.current_only)
    source=a.legacy
    if a.checkpoint:
        state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(state['model']);model.current_only=bool(state['current_only']);source=a.checkpoint
    result=export_model(model,Fine15Weights.from_vhood(a.legacy),a.output,hashlib.sha256(source.read_bytes()).hexdigest())
    loaded=load_model(a.output)
    assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items())
    result['quality_trained']=a.checkpoint is not None
    result['legacy_sha256']=hashlib.sha256(a.legacy.read_bytes()).hexdigest()
    a.output.with_suffix('.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
