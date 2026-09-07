"""Independent C++/torch fixture gate. A failed gate is a diagnostic, never training permission."""
from pathlib import Path
import argparse,json,sys,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from temporal.physics import TorchXPBD

def compare(row):
    tensor=lambda value:torch.tensor(value,dtype=torch.float32)
    rest=tensor(row['rest']);tri=torch.tensor(row['triangles'],dtype=torch.long).reshape(-1,3);mass=tensor(row['mass']);pins=torch.tensor(row['pinned'],dtype=torch.bool)
    solver=TorchXPBD(rest,tri,mass,pins,row['config']);body=row.get('collider')
    if body:
        body=dict(previous=tensor(body['previous']).reshape(-1,3),current=tensor(body['current']).reshape(-1,3),triangles=torch.tensor(body['triangles'],dtype=torch.long).reshape(-1,3),
            capsules=[{key:tensor(value) if key!='radius' else value for key,value in cap.items()} for cap in body['capsules']])
    x,v=solver.step(tensor(row['positions']),tensor(row['velocities']),row['dt'],tensor(row['pin_targets']),body)
    pe=float(torch.linalg.vector_norm(x-tensor(row['expected_positions']),dim=-1).max());ve=float(torch.linalg.vector_norm(v-tensor(row['expected_velocities']),dim=-1).max())
    return dict(name=row['name'],position_max_m=pe,velocity_max_m_s=ve,passed=pe<=1e-4 and ve<=1e-4)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=ROOT/'.work/temporal_v4/probe')
    p.add_argument('--output',type=Path,default=ROOT/'.work/temporal_v4/solver-gate.json');a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=True)
    subprocess.run([str(ROOT/'tests/build/temporal_reference_probe.exe'),str(a.directory)],check=True);torch.set_num_threads(4)
    rows=[]
    for path in sorted(a.directory.glob('fixture_*.json')):
        result=compare(json.loads(path.read_text()));rows.append(result);print(json.dumps(result),flush=True)
    gradients=subprocess.run([sys.executable,str(ROOT/'tests/test_temporal_physics.py')],capture_output=True,text=True)
    (a.directory/'gradient-tests.log').write_text(gradients.stdout+gradients.stderr)
    report=dict(reference_forward_passed=bool(rows) and all(r['passed'] for r in rows),contact_gradients_passed=gradients.returncode==0,cases=rows,
        real_asset_single_step_passed=False,scope='Synthetic fixture gate; real-asset forward parity remains separately required')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
