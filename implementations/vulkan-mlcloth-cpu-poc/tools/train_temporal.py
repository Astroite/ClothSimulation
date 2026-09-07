"""Train matching current-only/four-history models; rejects unqualified references."""
from pathlib import Path
import argparse,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from temporal.training import load_dataset,load_sequence,rollout,digest
from temporal.model import TemporalTinyHood
from temporal.physics import TorchXPBD
from temporal.format import export_model
from real_scene.fine15 import Fine15Weights

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--legacy',type=Path,default=ROOT.parent/'vulkan-gnn-poc/.work/hood_data/student32x12_r1.vhood')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--current-only',action='store_true');p.add_argument('--seed',type=int,default=17)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--windows-per-epoch',type=int,default=16);p.add_argument('--device',default='cuda')
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    # Gate before touching weights or allocating GPU buffers.
    rows=list(load_dataset(a.dataset));train=[r for r in rows if r['split']=='train'];validation=[r for r in rows if r['split']=='validation']
    if not train or not validation:raise ValueError('train and validation sequences required')
    a.output.mkdir(parents=True,exist_ok=True);torch.manual_seed(a.seed);torch.set_num_threads(4)
    model=TemporalTinyHood.from_legacy(a.legacy,a.current_only).to(a.device);model.checkpoint_blocks=True
    weights=Fine15Weights.from_vhood(a.legacy,a.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=1e-6)
    generator=torch.Generator().manual_seed(a.seed);first=0;best=float('inf');history=[]
    contract=dict(dataset_sha256=digest(a.dataset),legacy_sha256=digest(a.legacy),current_only=a.current_only,seed=a.seed)
    latest=a.output/'latest.pt'
    if a.resume:
        state=torch.load(latest,map_location='cpu',weights_only=True)
        if state['contract']!=contract:raise ValueError('resume dataset/model/ablation mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);generator.set_state(state['sampling_rng']);torch.set_rng_state(state['torch_rng'])
        if a.device.startswith('cuda') and state.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
        first=state['epoch']+1;best=state['best'];history=state['history']
    for epoch in range(first,a.epochs):
        started=time.monotonic();model.train();total=0.
        # Freeze backbone in the initial 5 epochs; zero temporal import already matches the teacher exactly.
        for name,parameter in model.named_parameters():parameter.requires_grad_(epoch>=5 or name.startswith('temporal_encoder'))
        for _ in range(a.windows_per_epoch):
            row=train[int(torch.randint(len(train),(1,),generator=generator))];s=load_sequence(row,a.device)
            solver=TorchXPBD(s['rest'],s['triangles'],s['mass'],s['pinned'],row['config'])
            start=int(torch.randint(3,len(s['x'])-4,(1,),generator=generator));optimizer.zero_grad(set_to_none=True)
            loss,_,_=rollout(model,weights,solver,s,start)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite coupled rollout loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);optimizer.step();total+=float(loss.detach())
            del s,solver,loss
        model.eval();score=0.
        with torch.no_grad():
            for row in validation:
                s=load_sequence(row,a.device);solver=TorchXPBD(s['rest'],s['triangles'],s['mass'],s['pinned'],row['config'])
                _,pred,_=rollout(model,weights,solver,s,0,len(s['x'])-1)
                x=torch.stack([v[0] for v in pred]);v=torch.stack([v[1] for v in pred])
                score+=float(((x-s['x'][1:])/s['length_scale']).square().mean()+.1*((v-s['v'][1:])/(30*s['length_scale'])).square().mean())
        score/=len(validation);improved=score<best;best=min(best,score)
        history.append(dict(epoch=epoch,train_loss=total/a.windows_per_epoch,validation_score=score,seconds=time.monotonic()-started))
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),current_only=model.current_only,contract=contract,epoch=epoch,best=best,history=history,
            sampling_rng=generator.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if a.device.startswith('cuda') else None)
        temp=a.output/'latest.tmp';torch.save(state,temp);temp.replace(latest)
        if improved:
            torch.save(state,a.output/'best.pt');export_model(model,weights,a.output/'best.vthood',digest(a.output/'best.pt'))
        (a.output/'training.json').write_text(json.dumps(dict(contract=contract,epochs=history,quality_qualified=False),indent=2));print(json.dumps(history[-1]),flush=True)
if __name__=='__main__':main()
