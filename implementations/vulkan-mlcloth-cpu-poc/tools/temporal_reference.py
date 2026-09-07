"""Checked reference loader and canonical continuous-sequence packer (no torch needed)."""
from pathlib import Path
from dataclasses import dataclass
import argparse,hashlib,json,math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SPLITS={'idle':'train','walk':'train','run':'train','sprint':'train','jump':'train','turn':'validation','complex':'test'}
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@dataclass
class ReferenceExport:
    metadata:dict
    arrays:dict

def load_reference(path):
    path=Path(path);meta=json.loads((path/'metadata.json').read_text(encoding='utf8'));blob=(path/'frames.bin').read_bytes();ranges=[]
    if meta.get('version')!=2 or meta.get('continuous') is not True or meta.get('byte_count')!=len(blob):raise ValueError('unsupported/discontinuous/truncated reference')
    def array(desc):
        dtype=desc['dtype'];offset=desc['offset'];size=desc['bytes'];shape=desc['shape']
        if dtype not in ('<f4','<u4','|u1','<u8') or not all(isinstance(d,int) and d>=0 for d in shape):raise ValueError('invalid array contract')
        dt=np.dtype(dtype)
        if not isinstance(offset,int) or not isinstance(size,int) or offset<0 or size<0 or size!=math.prod(shape)*dt.itemsize or offset+size>len(blob):raise ValueError('invalid array range')
        if size:ranges.append((offset,offset+size))
        result=np.frombuffer(blob,dtype=dt,count=math.prod(shape),offset=offset).reshape(shape).copy()
        if not np.isfinite(result).all():raise ValueError('nonfinite reference array')
        return result
    values={k:array(v) for k,v in meta['arrays'].items()};frames=meta['frames'];intervals=meta['intervals']
    if len(frames)!=len(intervals)+1 or len(frames)<2 or meta['frame_count']!=len(frames):raise ValueError('invalid frame count')
    times=np.asarray([f['time'] for f in frames]);
    if not np.allclose(times,np.arange(len(frames))/30,atol=1e-9,rtol=0):raise ValueError('noncontinuous frame timestamps')
    for key in ('x','v','body_positions','body_normals','surface_ids'):values[key]=np.stack([array(f[key]) for f in frames])
    values['pin_targets']=np.stack([np.stack([array(p) for p in it['pins']]) for it in intervals])
    for key,source in [('collider_positions','positions'),('capsules','capsules')]:
        values[key]=np.stack([np.stack([array(c[source]) for c in it['colliders']]) for it in intervals])
    for left,right in zip(sorted(ranges),sorted(ranges)[1:]):
        if left[1]>right[0]:raise ValueError('overlapping array ranges')
    f,n,_=values['x'].shape;m=values['body_positions'].shape[1];c=values['collider_positions'].shape[2]
    expected={'rest':(n,3),'mass':(n,),'pinned':(n,),'v':(f,n,3),'body_positions':(f,m,3),'body_normals':(f,m,3),
              'surface_ids':(f,m),'pin_targets':(f-1,8,n,3),'collider_positions':(f-1,9,c,3)}
    if any(values[key].shape!=shape for key,shape in expected.items()):raise ValueError('reference shape mismatch')
    if (values['mass']<=0).any() or not np.isin(values['pinned'],[0,1]).all():raise ValueError('invalid mass or pins')
    for key,count in [('triangles',n),('collider_triangles',c)]:
        t=values[key]
        if t.ndim!=2 or t.shape[1]!=3 or (t>=count).any():raise ValueError('invalid reference topology')
    ids=values['surface_ids']
    if len(np.unique(ids[0]))!=m or not (ids==ids[0]).all():raise ValueError('unstable surface identity')
    cp=values['collider_positions'];caps=values['capsules']
    if not np.array_equal(cp[:-1,-1],cp[1:,0]) or not np.array_equal(caps[:-1,-1],caps[1:,0]):raise ValueError('discontinuous body interval')
    return ReferenceExport(meta,values)

def pack_reference(path,output):
    reference=load_reference(path);output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**reference.arrays)
    config=dict(reference.metadata['physics_config']);config['iterations']=2
    return dict(clip=reference.metadata['clip'],split=SPLITS[reference.metadata['clip']],speed=reference.metadata['speed'],
        in_place=reference.metadata['in_place'],path=output.name,sha256=digest(output),config=config,
        source_metadata_sha256=digest(Path(path)/'metadata.json'),source_frames_sha256=digest(Path(path)/'frames.bin'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gate',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    gate=json.loads(a.gate.read_text());gate_path=a.output/'reference-gate.json'
    if gate_path.resolve()!=a.gate.resolve():gate_path.write_text(json.dumps(gate,indent=2))
    rows=[]
    for i,path in enumerate(a.reference):rows.append(pack_reference(path,a.output/f'sequence-{i:03}.npz'))
    meta=dict(version=1,reference_gate=gate_path.name,reference_gate_sha256=digest(gate_path),sequences=rows)
    (a.output/'dataset.json').write_text(json.dumps(meta,indent=2));print(json.dumps(dict(sequences=len(rows),training_allowed=bool(gate.get('training_allowed')),manifest=str(a.output/'dataset.json'))))
if __name__=='__main__':main()
