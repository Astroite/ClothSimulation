"""Exercise nonzero temporal weights and both ablation modes in the real Vulkan Demo."""
from pathlib import Path
import json,subprocess,sys,shutil,argparse,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import torch
from temporal.model import TemporalTinyHood
from temporal.format import export_model
from real_scene.fine15 import Fine15Weights
from validate_demo_gnn import validate

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frames',type=int,default=36);a=p.parse_args()
    legacy=ROOT.parent/'vulkan-gnn-poc/.work/hood_data/student32x12_r1.vhood';out=ROOT/'.work/temporal_v4/runtime';out.mkdir(parents=True,exist_ok=True)
    manifest=ROOT/'.work/demo/demo.json';data=json.loads(manifest.read_text());exe=ROOT/'.work/Vulkan/build-mlclothcpu/bin/mlclothcpu.exe'
    startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    report={'scope':'Nonzero synthetic temporal weights validate implementation, not trained cloth quality','cases':[],'quality_qualified':False}
    for mode in ('history','current'):
        torch.manual_seed(17);model=TemporalTinyHood.from_legacy(legacy,mode=='current')
        torch.nn.init.normal_(model.temporal_encoder[2].weight,std=1e-4)
        folder=out/mode;folder.mkdir(exist_ok=True);weights=folder/'probe.vthood';export_model(model,Fine15Weights.from_vhood(legacy),weights)
        data['temporal_gnn_model']=str(weights);tmp_manifest=manifest.parent/f'temporal-v4-{mode}.json';tmp_manifest.write_text(json.dumps(data,indent=2))
        args=[str(exe),'--demo-manifest',str(tmp_manifest),'--hybrid-algorithm','temporal','--validate-gnn','1','--dump-gnn','1','--fixed-frame-dt',str(1/60),'--frames',str(a.frames)]
        with (folder/'runtime.log').open('w') as log:subprocess.run(args,cwd=ROOT/'.work/Vulkan',startupinfo=startup,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
        source=manifest.parent/'gnn-validation';validate(source,legacy,weights)
        for name in ('gnn-integration-validation.json','performance.json'):shutil.copy2(manifest.parent/name,folder/name)
        shutil.copytree(source,folder/'dump',dirs_exist_ok=True)
        integration=json.loads((folder/'gnn-integration-validation.json').read_text());assert integration['passed']
        report['cases'].append({'mode':mode,'integration':integration,'reference':json.loads((source/'reference-report.json').read_text())})
    report['passed']=True;(out/'report.json').write_text(json.dumps(report,indent=2));print(out/'report.json')
if __name__=='__main__':main()
